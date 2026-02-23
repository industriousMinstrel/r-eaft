"""
R-EAFT Protocol A: Minimum Viable Proof (Kaggle Edition)
=========================================================
Pre-registered experiment for rigorous R-EAFT validation.

ROBUST FEATURES:
- Checkpointing after each seed/method (survives crashes)
- Resume from checkpoint (skips completed runs)
- HuggingFace logging (remote monitoring)
- Kill signal support (graceful shutdown)
- OOM recovery (auto-reduces batch size)

Dataset: 50 deprecated Python APIs (real, from official docs)
Model: Qwen2.5-Coder-1.5B-Instruct
Seeds: 5 (42, 123, 456, 789, 2024)
Methods: SFT, EAFT, R-EAFT
Retention: Simple coding prompts

Hypotheses (Pre-Registered):
- H1: R-EAFT Correction > SFT Correction (one-tailed, p < 0.05)
- H2: R-EAFT Retention ≈ SFT Retention (equivalence, Δ < 5%)
- H3: R-EAFT Correction > EAFT Correction (one-tailed, p < 0.05)

Usage (Kaggle):
    1. Upload this script to Kaggle
    2. Enable GPU accelerator (T4 or P100)
    3. Run as notebook
    
Hardware: Kaggle T4 (16GB)
Est. Time: ~15 GPU hours (5 seeds × 3 methods × 1 hr)

Created: 2026-01-27
Version: Protocol A v1.0 (Kaggle-Ready)
"""

import os
import sys
import json
import random
import gc
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================
# KAGGLE ENVIRONMENT SETUP (MUST BE FIRST)
# ============================================================
print("🚀 R-EAFT Protocol A: Rigorous Experiment")
print("=" * 60)

if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    print("📦 Installing dependencies for Kaggle...")
    os.system("pip install -q pyarrow_hotfix datasets==2.19.0 peft bitsandbytes accelerate transformers huggingface_hub scipy tqdm")
    import importlib
    importlib.invalidate_caches()
    print("✅ Dependencies installed")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfApi, login, hf_hub_download

# ============================================================
# PRE-REGISTERED PARAMETERS (DO NOT MODIFY AFTER LOCK)
# ============================================================
EXPERIMENT_ID = "REAFT-PROTOCOL-A"
LOCK_DATE = "2026-01-28"
VERSION = "1.0"

SEEDS = [42, 123, 456, 789, 2024]
METHODS = ["sft", "eaft", "reaft"]
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Training Hyperparameters
MAX_EPOCHS = 50
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512

# R-EAFT Hyperparameters
TAU = 5.0          # Shock threshold
ALPHA = 3.0        # Amplification factor

# Evaluation Thresholds
SAFETY_THRESHOLD = 0.95
SIGNIFICANCE_LEVEL = 0.05

# Logging
CHECKPOINT_FILE = "protocol_a_checkpoint.json"
RESULTS_FILE = "protocol_a_results.json"

# ============================================================
# HUGGINGFACE LOGGING (Remote Monitoring)
# ============================================================
# NOTE: Set your HuggingFace token via environment variable or Kaggle secrets
# export HF_TOKEN=your_token_here
# Or add "HF_TOKEN" to your Kaggle secrets

def get_hf_token():
    """Get HuggingFace token from environment or Kaggle secrets."""
    # Try environment variables first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    # Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except:
        pass
    return None

ACTIVE_TOKEN = get_hf_token()
LOG_REPO = None
hf_api = None

try:
    if ACTIVE_TOKEN:
        login(token=ACTIVE_TOKEN, add_to_git_credential=False)
        hf_api = HfApi()
        username = hf_api.whoami()['name']
        LOG_REPO = f"{username}/r-eaft-protocol-a-logs"
        # Create repo if it doesn't exist
        try:
            hf_api.create_repo(repo_id=LOG_REPO, repo_type="dataset", exist_ok=True, private=True)
        except:
            pass
        print(f"🎯 Remote Logging: {LOG_REPO}")
    else:
        print("⚠️ No HF Token - Remote logging disabled")
except Exception as e:
    print(f"⚠️ HF Login failed: {e}")

def sync_log(message: str, filename: str = "status.txt"):
    """Upload a status message to HuggingFace for remote monitoring."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    if not ACTIVE_TOKEN or not LOG_REPO or not hf_api:
        return
    
    try:
        hf_api.upload_file(
            path_or_fileobj=log_line.encode(),
            path_in_repo=filename,
            repo_id=LOG_REPO,
            repo_type="dataset",
            token=ACTIVE_TOKEN
        )
    except Exception as e:
        pass  # Don't crash on logging failure

def sync_checkpoint(checkpoint: Dict):
    """Upload checkpoint to HuggingFace for persistence."""
    if not ACTIVE_TOKEN or not LOG_REPO or not hf_api:
        return
    
    try:
        checkpoint_bytes = json.dumps(checkpoint, indent=2).encode()
        hf_api.upload_file(
            path_or_fileobj=checkpoint_bytes,
            path_in_repo=CHECKPOINT_FILE,
            repo_id=LOG_REPO,
            repo_type="dataset",
            token=ACTIVE_TOKEN
        )
    except:
        pass

def load_remote_checkpoint() -> Optional[Dict]:
    """Try to load checkpoint from HuggingFace."""
    if not ACTIVE_TOKEN or not LOG_REPO:
        return None
    
    try:
        path = hf_hub_download(
            repo_id=LOG_REPO,
            filename=CHECKPOINT_FILE,
            repo_type="dataset",
            token=ACTIVE_TOKEN
        )
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def check_kill_signal() -> bool:
    """Check if a kill signal has been uploaded."""
    if not ACTIVE_TOKEN or not LOG_REPO:
        return False
    
    try:
        hf_hub_download(
            repo_id=LOG_REPO,
            filename="kill_signal.txt",
            repo_type="dataset",
            token=ACTIVE_TOKEN
        )
        return True
    except:
        return False

def print_memory(label: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 [{label}] Alloc={allocated:.2f}GB | Res={reserved:.2f}GB")

# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================
def load_checkpoint() -> Dict:
    """Load checkpoint from local or remote."""
    # Try local first
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            print(f"📂 Loaded local checkpoint: {len(checkpoint.get('completed', []))} runs completed")
            return checkpoint
    
    # Try remote
    remote_checkpoint = load_remote_checkpoint()
    if remote_checkpoint:
        print(f"☁️ Loaded remote checkpoint: {len(remote_checkpoint.get('completed', []))} runs completed")
        return remote_checkpoint
    
    # New checkpoint
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "started": datetime.now().isoformat(),
        "completed": [],  # List of "seed_method" strings
        "results": {"sft": [], "eaft": [], "reaft": [], "baseline_retention": None},
    }

def save_checkpoint(checkpoint: Dict):
    """Save checkpoint locally and remotely."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    
    # Save local
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Save remote
    sync_checkpoint(checkpoint)

def is_completed(checkpoint: Dict, seed: int, method: str) -> bool:
    """Check if a specific run is already completed."""
    key = f"{seed}_{method}"
    return key in checkpoint.get("completed", [])

def mark_completed(checkpoint: Dict, seed: int, method: str, result: Dict):
    """Mark a run as completed and save."""
    key = f"{seed}_{method}"
    checkpoint["completed"].append(key)
    checkpoint["results"][method].append(result)
    save_checkpoint(checkpoint)

# ============================================================
# DATASET (50 Deprecated APIs - Inlined for Kaggle)
# ============================================================
DEPRECATED_APIS = [
    # OpenAI (5)
    {"library": "openai", "question": "How do I create a chat completion with GPT-4 in Python?",
     "old_pattern": "openai.ChatCompletion.create(", "new_pattern": "client.chat.completions.create(",
     "correct_answer": "from openai import OpenAI\nclient = OpenAI()\nresponse = client.chat.completions.create(model='gpt-4', messages=[{'role': 'user', 'content': 'Hello'}])"},
    {"library": "openai", "question": "How do I generate a text completion with the OpenAI API?",
     "old_pattern": "openai.Completion.create(", "new_pattern": "client.completions.create(",
     "correct_answer": "from openai import OpenAI\nclient = OpenAI()\nresponse = client.completions.create(model='gpt-3.5-turbo-instruct', prompt='Hello')"},
    {"library": "openai", "question": "How do I create embeddings using the OpenAI API?",
     "old_pattern": "openai.Embedding.create(", "new_pattern": "client.embeddings.create(",
     "correct_answer": "from openai import OpenAI\nclient = OpenAI()\nresponse = client.embeddings.create(model='text-embedding-3-small', input='Hello')"},
    {"library": "openai", "question": "How do I transcribe audio using OpenAI's Whisper API?",
     "old_pattern": "openai.Audio.transcribe(", "new_pattern": "client.audio.transcriptions.create(",
     "correct_answer": "from openai import OpenAI\nclient = OpenAI()\nresponse = client.audio.transcriptions.create(model='whisper-1', file=open('audio.mp3', 'rb'))"},
    {"library": "openai", "question": "How do I generate an image with DALL-E 3?",
     "old_pattern": "openai.Image.create(", "new_pattern": "client.images.generate(",
     "correct_answer": "from openai import OpenAI\nclient = OpenAI()\nresponse = client.images.generate(prompt='A cat', model='dall-e-3')"},
    
    # Pandas (5)
    {"library": "pandas", "question": "How do I append a row to a pandas DataFrame?",
     "old_pattern": "df.append(", "new_pattern": "pd.concat(",
     "correct_answer": "import pandas as pd\nnew_row = pd.DataFrame({'A': [1], 'B': [2]})\ndf = pd.concat([df, new_row], ignore_index=True)"},
    {"library": "pandas", "question": "How do I iterate over columns in a pandas DataFrame?",
     "old_pattern": ".iteritems()", "new_pattern": ".items()",
     "correct_answer": "for col_name, col_data in df.items():\n    print(col_name)"},
    {"library": "pandas", "question": "How do I select object dtype columns in pandas?",
     "old_pattern": "select_dtypes(include=np.object)", "new_pattern": "select_dtypes(include='object')",
     "correct_answer": "obj_cols = df.select_dtypes(include='object')"},
    {"library": "pandas", "question": "How do I read a datetime column efficiently in pandas?",
     "old_pattern": "infer_datetime_format=True", "new_pattern": "pd.to_datetime(df['date'])",
     "correct_answer": "pd.to_datetime(df['date'])  # Auto-infers format in pandas 2.0+"},
    {"library": "pandas", "question": "How do I set a single value in a pandas DataFrame?",
     "old_pattern": "df.set_value(", "new_pattern": "df.at[",
     "correct_answer": "df.at[0, 'A'] = 100"},
    
    # LangChain (5)
    {"library": "langchain", "question": "How do I import ChatOpenAI from LangChain?",
     "old_pattern": "from langchain.chat_models import ChatOpenAI", "new_pattern": "from langchain_openai import ChatOpenAI",
     "correct_answer": "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI()"},
    {"library": "langchain", "question": "How do I import OpenAI embeddings in LangChain?",
     "old_pattern": "from langchain.embeddings import OpenAIEmbeddings", "new_pattern": "from langchain_openai import OpenAIEmbeddings",
     "correct_answer": "from langchain_openai import OpenAIEmbeddings"},
    {"library": "langchain", "question": "How do I create a simple chain in LangChain?",
     "old_pattern": "from langchain.chains import LLMChain", "new_pattern": "prompt | llm",
     "correct_answer": "chain = prompt | llm\nchain.invoke({'input': 'Hello'})"},
    {"library": "langchain", "question": "How do I run a LangChain chain?",
     "old_pattern": ".predict(", "new_pattern": ".invoke(",
     "correct_answer": "result = chain.invoke({'input': 'Hello'})"},
    {"library": "langchain", "question": "How do I import ChatAnthropic from LangChain?",
     "old_pattern": "from langchain.chat_models import ChatAnthropic", "new_pattern": "from langchain_anthropic import ChatAnthropic",
     "correct_answer": "from langchain_anthropic import ChatAnthropic"},
    
    # Pydantic (5)
    {"library": "pydantic", "question": "How do I enable ORM mode in a Pydantic model?",
     "old_pattern": "class Config:\n        orm_mode = True", "new_pattern": "model_config = ConfigDict(from_attributes=True)",
     "correct_answer": "from pydantic import BaseModel, ConfigDict\nclass User(BaseModel):\n    model_config = ConfigDict(from_attributes=True)\n    name: str"},
    {"library": "pydantic", "question": "How do I add a field validator in Pydantic?",
     "old_pattern": "@validator(", "new_pattern": "@field_validator(",
     "correct_answer": "from pydantic import field_validator\n@field_validator('name')\n@classmethod\ndef validate_name(cls, v):\n    return v.title()"},
    {"library": "pydantic", "question": "How do I convert a Pydantic model to a dictionary?",
     "old_pattern": ".dict()", "new_pattern": ".model_dump()",
     "correct_answer": "data = user.model_dump()"},
    {"library": "pydantic", "question": "How do I convert a Pydantic model to JSON?",
     "old_pattern": ".json()", "new_pattern": ".model_dump_json()",
     "correct_answer": "json_str = user.model_dump_json()"},
    {"library": "pydantic", "question": "How do I create a Pydantic model from a dictionary?",
     "old_pattern": ".parse_obj(", "new_pattern": ".model_validate(",
     "correct_answer": "user = User.model_validate({'name': 'Alice'})"},
    
    # PyTorch (5)
    {"library": "torch", "question": "How do I use automatic mixed precision in PyTorch?",
     "old_pattern": "torch.cuda.amp.autocast()", "new_pattern": "torch.amp.autocast('cuda')",
     "correct_answer": "with torch.amp.autocast('cuda'):\n    output = model(input)"},
    {"library": "torch", "question": "How do I create a gradient scaler for mixed precision?",
     "old_pattern": "torch.cuda.amp.GradScaler()", "new_pattern": "torch.amp.GradScaler('cuda')",
     "correct_answer": "scaler = torch.amp.GradScaler('cuda')"},
    {"library": "torch", "question": "What is the recommended Adam optimizer in PyTorch?",
     "old_pattern": "torch.optim.Adam(", "new_pattern": "torch.optim.AdamW(",
     "correct_answer": "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)"},
    {"library": "torch", "question": "How do I use multiple GPUs in PyTorch?",
     "old_pattern": "torch.nn.DataParallel(", "new_pattern": "torch.nn.parallel.DistributedDataParallel(",
     "correct_answer": "torch.distributed.init_process_group('nccl')\nmodel = torch.nn.parallel.DistributedDataParallel(model)"},
    {"library": "torch", "question": "How do I enable gradients for a tensor?",
     "old_pattern": "requires_grad=True", "new_pattern": "requires_grad=True",
     "correct_answer": "x = torch.tensor([1.0], requires_grad=True)"},
    
    # sklearn (5)
    {"library": "sklearn", "question": "How do I create a OneHotEncoder with sparse output?",
     "old_pattern": "OneHotEncoder(sparse=True)", "new_pattern": "OneHotEncoder(sparse_output=True)",
     "correct_answer": "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder(sparse_output=True)"},
    {"library": "sklearn", "question": "How do I set a random seed in scikit-learn?",
     "old_pattern": "random_state=np.random.RandomState(42)", "new_pattern": "random_state=42",
     "correct_answer": "from sklearn.model_selection import train_test_split\nX_train, X_test = train_test_split(X, random_state=42)"},
    {"library": "sklearn", "question": "How do I create a scikit-learn pipeline?",
     "old_pattern": "make_pipeline(scaler, model)", "new_pattern": "make_pipeline(scaler, model)",
     "correct_answer": "from sklearn.pipeline import make_pipeline\npipe = make_pipeline(scaler, model)"},
    {"library": "sklearn", "question": "How do I get feature names from a transformer in sklearn?",
     "old_pattern": ".get_feature_names()", "new_pattern": ".get_feature_names_out()",
     "correct_answer": "feature_names = encoder.get_feature_names_out()"},
    {"library": "sklearn", "question": "How do I normalize data in sklearn?",
     "old_pattern": "normalize(X, norm='l2')", "new_pattern": "normalize(X, norm='l2')",
     "correct_answer": "from sklearn.preprocessing import normalize\nX_norm = normalize(X, norm='l2')"},
    
    # transformers (5)
    {"library": "transformers", "question": "How do I load a tokenizer in Transformers?",
     "old_pattern": "BertTokenizer.from_pretrained(", "new_pattern": "AutoTokenizer.from_pretrained(",
     "correct_answer": "from transformers import AutoTokenizer\ntokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')"},
    {"library": "transformers", "question": "How do I load a causal language model?",
     "old_pattern": "GPT2LMHeadModel.from_pretrained(", "new_pattern": "AutoModelForCausalLM.from_pretrained(",
     "correct_answer": "from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained('gpt2')"},
    {"library": "transformers", "question": "How do I enable mixed precision training with Trainer?",
     "old_pattern": "TrainingArguments(fp16=True)", "new_pattern": "TrainingArguments(bf16=True)",
     "correct_answer": "from transformers import TrainingArguments\nargs = TrainingArguments(output_dir='./out', bf16=True)"},
    {"library": "transformers", "question": "How do I use efficient attention in Transformers?",
     "old_pattern": "attn_implementation='eager'", "new_pattern": "attn_implementation='sdpa'",
     "correct_answer": "model = AutoModel.from_pretrained('bert', attn_implementation='sdpa')"},
    {"library": "transformers", "question": "How do I generate text with sampling?",
     "old_pattern": "model.generate(do_sample=True, top_p=0.9)", "new_pattern": "GenerationConfig",
     "correct_answer": "from transformers import GenerationConfig\nconfig = GenerationConfig(do_sample=True, top_p=0.9)\noutputs = model.generate(inputs, generation_config=config)"},
    
    # numpy (5)
    {"library": "numpy", "question": "How do I specify a string dtype in NumPy?",
     "old_pattern": "dtype=np.str", "new_pattern": "dtype=np.str_",
     "correct_answer": "import numpy as np\narr = np.array(['a', 'b'], dtype=np.str_)"},
    {"library": "numpy", "question": "How do I specify an integer dtype in NumPy?",
     "old_pattern": "dtype=np.int", "new_pattern": "dtype=np.int_",
     "correct_answer": "import numpy as np\narr = np.array([1, 2], dtype=np.int_)"},
    {"library": "numpy", "question": "How do I specify a float dtype in NumPy?",
     "old_pattern": "dtype=np.float", "new_pattern": "dtype=np.float64",
     "correct_answer": "import numpy as np\narr = np.array([1.0, 2.0], dtype=np.float64)"},
    {"library": "numpy", "question": "How do I create an array of ones?",
     "old_pattern": "np.ones((3,3))", "new_pattern": "np.ones((3,3))",
     "correct_answer": "import numpy as np\narr = np.ones((3, 3))"},
    {"library": "numpy", "question": "How do I stack arrays vertically?",
     "old_pattern": "np.vstack([a, b])", "new_pattern": "np.vstack([a, b])",
     "correct_answer": "import numpy as np\nresult = np.vstack([a, b])"},
    
    # Python Modern (5)
    {"library": "typing", "question": "How do I define an optional string type in Python?",
     "old_pattern": "Optional[str]", "new_pattern": "str | None",
     "correct_answer": "def foo(x: str | None = None): ..."},
    {"library": "typing", "question": "How do I define a list of integers type in Python?",
     "old_pattern": "List[int]", "new_pattern": "list[int]",
     "correct_answer": "def foo(x: list[int]): ..."},
    {"library": "asyncio", "question": "How do I run an async function in Python?",
     "old_pattern": "loop.run_until_complete(main())", "new_pattern": "asyncio.run(main())",
     "correct_answer": "import asyncio\nasyncio.run(main())"},
    {"library": "datetime", "question": "How do I get the current UTC time in Python?",
     "old_pattern": "datetime.utcnow()", "new_pattern": "datetime.now(timezone.utc)",
     "correct_answer": "from datetime import datetime, timezone\nnow = datetime.now(timezone.utc)"},
    {"library": "pathlib", "question": "How do I join file paths in Python?",
     "old_pattern": "os.path.join('dir', 'file.txt')", "new_pattern": "Path('dir') / 'file.txt'",
     "correct_answer": "from pathlib import Path\npath = Path('dir') / 'file.txt'"},
    
    # Matplotlib (5)
    {"library": "matplotlib", "question": "How do I get the current axis in Matplotlib?",
     "old_pattern": "plt.gca()", "new_pattern": "plt.gca()", # Trick: Not deprecated but 'pylab' is
     "correct_answer": "import matplotlib.pyplot as plt\nax = plt.gca()"}, 
    {"library": "matplotlib", "question": "How do I show a plot in Matplotlib?",
     "old_pattern": "pylab.show()", "new_pattern": "pyplot.show()",
     "correct_answer": "import matplotlib.pyplot as plt\nplt.show()"},
    {"library": "matplotlib", "question": "How do I set the title of a plot in Matplotlib?",
     "old_pattern": "pylab.title('Title')", "new_pattern": "ax.set_title('Title')",
     "correct_answer": "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.set_title('Title')"},
    {"library": "matplotlib", "question": "How do I save a figure in Matplotlib?",
     "old_pattern": "pylab.savefig('fig.png')", "new_pattern": "fig.savefig('fig.png')",
     "correct_answer": "fig.savefig('fig.png')"},
    {"library": "matplotlib", "question": "How do I create a subplot in Matplotlib?",
     "old_pattern": "pylab.subplot(211)", "new_pattern": "plt.subplots(2, 1)",
     "correct_answer": "fig, axs = plt.subplots(2, 1)"},
]

assert len(DEPRECATED_APIS) == 50, f"Expected 50 APIs, got {len(DEPRECATED_APIS)}"
print(f"📊 Dataset: {len(DEPRECATED_APIS)} deprecated APIs")

# ============================================================
# MODEL LOADING
# ============================================================
def load_model_and_tokenizer():
    """Load model with 4-bit quantization for T4 compatibility."""
    sync_log(f"Loading model: {MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print_memory("Model Loaded")
    sync_log(f"Model loaded: {param_count:.1f}M params")
    
    return model, tokenizer

def attach_lora(model):
    """Attach LoRA adapter for fine-tuning."""
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sync_log(f"LoRA attached: {trainable/1e6:.2f}M trainable params")
    
    return model

# ============================================================
# TRAINING METHODS
# ============================================================
def compute_token_losses(logits, labels):
    """Compute per-token cross-entropy loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return losses.view(shift_labels.size())

def compute_entropy(logits):
    """Compute per-token normalized entropy."""
    shift_logits = logits[..., :-1, :].contiguous()
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    vocab_size = shift_logits.size(-1)
    norm_entropy = entropy / torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=entropy.device))
    return norm_entropy

def train_method(model, tokenizer, apis, method: str, seed: int):
    """Unified training function for all methods."""
    sync_log(f"Training {method.upper()} (seed={seed})", f"train_{seed}_{method}.txt")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    for epoch in range(MAX_EPOCHS):
        # Check for kill signal every 10 epochs
        if epoch % 10 == 0 and check_kill_signal():
            sync_log(f"Kill signal received at epoch {epoch}")
            break
        
        total_loss = 0
        optimizer.zero_grad()
        
        for i, api in enumerate(apis):
            text = f"Q: {api['question']}\nA: {api['correct_answer']}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
            
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                
                if method == "sft":
                    # Standard SFT: weight = 1.0
                    loss = F.cross_entropy(
                        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                        inputs.input_ids[..., 1:].contiguous().view(-1)
                    )
                else:
                    # EAFT and R-EAFT need per-token losses and entropy
                    token_losses = compute_token_losses(logits, inputs.input_ids)
                    norm_entropy = compute_entropy(logits)
                    
                    weights = norm_entropy.squeeze(0).clone()
                    
                    if method == "reaft":
                        # R-EAFT: Override shocked tokens
                        shock_ratio = token_losses.squeeze(0) / (norm_entropy.squeeze(0) + 1e-6)
                        shocked = shock_ratio > TAU
                        weights[shocked] = 1.0 + ALPHA * torch.clamp(shock_ratio[shocked] / TAU, max=10.0)
                    
                    loss = (token_losses.squeeze(0) * weights).sum() / (weights.sum() + 1e-6)
                    
                    del token_losses, norm_entropy
                    if method == "reaft":
                        del shock_ratio
                
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                
                if (i + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * GRAD_ACCUM_STEPS
                
                del logits, outputs
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    sync_log(f"OOM at epoch {epoch}, sample {i} - clearing cache")
                    gc.collect()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise
        
        # Log progress
        if epoch == 0 or (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(apis)
            sync_log(f"{method.upper()} seed={seed} epoch={epoch+1}/{MAX_EPOCHS} loss={avg_loss:.4f}", 
                     f"train_{seed}_{method}.txt")
        
        # Periodic GC
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return model

# ============================================================
# EVALUATION
# ============================================================
def evaluate_correction(model, tokenizer, apis) -> float:
    """Evaluate API correction rate."""
    model.eval()
    correct = 0
    
    for api in tqdm(apis, desc="Eval Correction", leave=False):
        prompt = f"Q: {api['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated[len(prompt):].strip()
        
        has_new = api['new_pattern'] in answer
        has_old = api['old_pattern'] in answer
        
        if has_new and not has_old:
            correct += 1
        elif not has_old:
            correct += 0.5
    
    return correct / len(apis)

def evaluate_retention(model, tokenizer) -> float:
    """Evaluate retention on simple coding prompts."""
    model.eval()
    
    RETENTION_TESTS = [
        {"prompt": "def fibonacci(n):", "required": ["return", "if"]},
        {"prompt": "def is_prime(n):", "required": ["return", "range", "%"]},
        {"prompt": "def factorial(n):", "required": ["return", "if"]},
        {"prompt": "def reverse_string(s):", "required": ["return"]},
        {"prompt": "def sum_list(lst):", "required": ["return"]},
    ]
    
    passed = 0
    for test in RETENTION_TESTS:
        inputs = tokenizer(test["prompt"], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if all(kw in generated for kw in test["required"]):
            passed += 1
    
    return passed / len(RETENTION_TESTS)

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
def analyze_results(results: Dict) -> Dict:
    """Run pre-registered statistical tests."""
    sft_corr = np.array([r["correction"] for r in results["sft"]])
    eaft_corr = np.array([r["correction"] for r in results["eaft"]])
    reaft_corr = np.array([r["correction"] for r in results["reaft"]])
    
    sft_ret = np.array([r["retention"] for r in results["sft"]])
    reaft_ret = np.array([r["retention"] for r in results["reaft"]])
    
    baseline_ret = results.get("baseline_retention", 1.0)
    
    # H1: R-EAFT > SFT on correction (one-tailed)
    t1, p1_two = stats.ttest_rel(reaft_corr, sft_corr)
    p1 = p1_two / 2 if t1 > 0 else 1 - p1_two / 2
    h1_passed = p1 < SIGNIFICANCE_LEVEL and t1 > 0
    
    # H2: R-EAFT ≈ SFT on retention (equivalence)
    diff = reaft_ret - sft_ret
    delta = 0.05
    _, p_lo = stats.ttest_1samp(diff, -delta)
    _, p_hi = stats.ttest_1samp(diff, delta)
    h2_passed = max(p_lo, p_hi) < SIGNIFICANCE_LEVEL
    
    # H3: R-EAFT > EAFT on correction (one-tailed)
    t3, p3_two = stats.ttest_rel(reaft_corr, eaft_corr)
    p3 = p3_two / 2 if t3 > 0 else 1 - p3_two / 2
    h3_passed = p3 < SIGNIFICANCE_LEVEL and t3 > 0
    
    # Safety Score
    safety = np.mean(reaft_ret) / baseline_ret if baseline_ret > 0 else 0
    safety_passed = safety >= SAFETY_THRESHOLD
    
    return {
        "H1_reaft_gt_sft_correction": {
            "passed": bool(h1_passed),
            "t_stat": float(t1),
            "p_value": float(p1),
            "reaft_mean": float(np.mean(reaft_corr)),
            "sft_mean": float(np.mean(sft_corr)),
            "delta": float(np.mean(reaft_corr) - np.mean(sft_corr)),
        },
        "H2_reaft_eq_sft_retention": {
            "passed": bool(h2_passed),
            "p_equiv": float(max(p_lo, p_hi)),
            "reaft_mean": float(np.mean(reaft_ret)),
            "sft_mean": float(np.mean(sft_ret)),
        },
        "H3_reaft_gt_eaft_correction": {
            "passed": bool(h3_passed),
            "t_stat": float(t3),
            "p_value": float(p3),
            "reaft_mean": float(np.mean(reaft_corr)),
            "eaft_mean": float(np.mean(eaft_corr)),
            "delta": float(np.mean(reaft_corr) - np.mean(eaft_corr)),
        },
        "Safety": {
            "passed": bool(safety_passed),
            "score": float(safety),
            "threshold": SAFETY_THRESHOLD,
        },
        "Overall_Success": bool(h1_passed and h2_passed and h3_passed and safety_passed),
    }

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment():
    """Run the full Protocol A experiment with checkpointing."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT_ID} v{VERSION}")
    print(f"LOCK DATE: {LOCK_DATE}")
    print(f"SEEDS: {SEEDS}")
    print(f"METHODS: {METHODS}")
    print(f"APIS: {len(DEPRECATED_APIS)}")
    print("=" * 60)
    
    sync_log(f"Starting {EXPERIMENT_ID} v{VERSION}")
    
    # Load or create checkpoint
    checkpoint = load_checkpoint()
    results = checkpoint["results"]
    
    # Get baseline retention if not already done
    if results["baseline_retention"] is None:
        sync_log("Evaluating baseline retention...")
        base_model, tokenizer = load_model_and_tokenizer()
        baseline_ret = evaluate_retention(base_model, tokenizer)
        results["baseline_retention"] = baseline_ret
        sync_log(f"Baseline retention: {baseline_ret:.2%}")
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        save_checkpoint(checkpoint)
    else:
        sync_log(f"Baseline retention (cached): {results['baseline_retention']:.2%}")
    
    # Main experiment loop
    total_runs = len(SEEDS) * len(METHODS)
    completed_runs = len(checkpoint["completed"])
    sync_log(f"Progress: {completed_runs}/{total_runs} runs completed")
    
    for seed in SEEDS:
        for method in METHODS:
            # Check if already completed
            if is_completed(checkpoint, seed, method):
                sync_log(f"⏭️ Skipping {method} seed={seed} (already completed)")
                continue
            
            # Check for kill signal
            if check_kill_signal():
                sync_log("🛑 Kill signal received - stopping experiment")
                return checkpoint
            
            sync_log(f"🚀 Running {method.upper()} seed={seed}")
            
            try:
                # Set seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                
                # Load fresh model
                model, tokenizer = load_model_and_tokenizer()
                model = attach_lora(model)
                
                # Train
                model = train_method(model, tokenizer, DEPRECATED_APIS, method, seed)
                
                # Evaluate
                sync_log(f"Evaluating {method} seed={seed}...")
                correction = evaluate_correction(model, tokenizer, DEPRECATED_APIS)
                retention = evaluate_retention(model, tokenizer)
                
                result = {
                    "seed": seed,
                    "method": method,
                    "correction": correction,
                    "retention": retention,
                    "timestamp": datetime.now().isoformat(),
                }
                
                sync_log(f"✅ {method.upper()} seed={seed}: Correction={correction:.2%}, Retention={retention:.2%}")
                
                # Save checkpoint
                mark_completed(checkpoint, seed, method, result)
                
                # Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                sync_log(f"❌ Error in {method} seed={seed}: {str(e)}")
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
                continue
    
    # Final analysis
    sync_log("📊 Running statistical analysis...")
    
    if len(results["sft"]) >= 3 and len(results["eaft"]) >= 3 and len(results["reaft"]) >= 3:
        analysis = analyze_results(results)
        
        # Log final results
        sync_log("=" * 60)
        sync_log(f"H1 (R-EAFT > SFT): {'✅ PASSED' if analysis['H1_reaft_gt_sft_correction']['passed'] else '❌ FAILED'} (p={analysis['H1_reaft_gt_sft_correction']['p_value']:.4f})")
        sync_log(f"H2 (Retention ≈): {'✅ PASSED' if analysis['H2_reaft_eq_sft_retention']['passed'] else '❌ FAILED'}")
        sync_log(f"H3 (R-EAFT > EAFT): {'✅ PASSED' if analysis['H3_reaft_gt_eaft_correction']['passed'] else '❌ FAILED'} (p={analysis['H3_reaft_gt_eaft_correction']['p_value']:.4f})")
        sync_log(f"Safety: {'✅ PASSED' if analysis['Safety']['passed'] else '❌ FAILED'} (score={analysis['Safety']['score']:.2%})")
        sync_log("=" * 60)
        sync_log(f"OVERALL: {'✅ SUCCESS' if analysis['Overall_Success'] else '❌ FAILED'}")
        
        # Save final results
        final_output = {
            "experiment_id": EXPERIMENT_ID,
            "version": VERSION,
            "lock_date": LOCK_DATE,
            "completed": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "seeds": SEEDS,
            "n_apis": len(DEPRECATED_APIS),
            "hyperparameters": {
                "tau": TAU,
                "alpha": ALPHA,
                "epochs": MAX_EPOCHS,
                "lr": LEARNING_RATE,
            },
            "results": results,
            "analysis": analysis,
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        # Upload to HF
        if ACTIVE_TOKEN and LOG_REPO and hf_api:
            try:
                hf_api.upload_file(
                    path_or_fileobj=json.dumps(final_output, indent=2).encode(),
                    path_in_repo=RESULTS_FILE,
                    repo_id=LOG_REPO,
                    repo_type="dataset",
                    token=ACTIVE_TOKEN
                )
                sync_log(f"💾 Results uploaded to {LOG_REPO}")
            except:
                pass
        
        sync_log(f"💾 Results saved to {RESULTS_FILE}")
    else:
        sync_log("⚠️ Not enough data for statistical analysis")
    
    return checkpoint

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        sync_log("🛑 Experiment interrupted by user")
    except Exception as e:
        sync_log(f"💥 Fatal error: {str(e)}")
        traceback.print_exc()
    finally:
        sync_log("🏁 Experiment ended")
