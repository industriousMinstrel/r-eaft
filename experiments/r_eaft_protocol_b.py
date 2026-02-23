"""
R-EAFT Protocol B: Natural Filter Hypothesis (Kaggle Edition)
===============================================================
Tests whether R-EAFT can naturally filter mixed data (news + archive)
without needing an explicit replay buffer.

HYPOTHESIS:
When fine-tuning on mixed "News + Archive" data:
- R-EAFT should protect stable archive knowledge (low entropy → dampened)
- R-EAFT should update news/political facts (high shock → amplified)
- Result: Updates political knowledge WITHOUT forgetting history

DATASET:
- News: Post-cutoff political facts (2025-2026)
- Archive: Pre-cutoff stable facts (history, geography, science)
- Total: 50 news + 50 archive = 100 facts per training run

MODEL: Qwen/Qwen2.5-1.5B-Instruct (General, not Coder)

SEEDS: 5 (42, 123, 456, 789, 2024)
METHODS: SFT, EAFT, R-EAFT (NO replay buffer for any)

METRICS:
- News Accuracy: % of post-cutoff facts learned
- Archive Retention: % of pre-cutoff facts retained
- Safety Score: Archive Retention / Baseline

Created: 2026-01-30
Version: Protocol B v1.0
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
# KAGGLE ENVIRONMENT SETUP
# ============================================================
print("🚀 R-EAFT Protocol B: Natural Filter Hypothesis")
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
# PRE-REGISTERED PARAMETERS
# ============================================================
EXPERIMENT_ID = "REAFT-PROTOCOL-B"
LOCK_DATE = "2026-01-30"
VERSION = "1.0"

SEEDS = [42, 123, 456, 789, 2024]
METHODS = ["sft", "eaft", "reaft"]
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # General model, not Coder

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
CHECKPOINT_FILE = "protocol_b_checkpoint.json"
RESULTS_FILE = "protocol_b_results.json"

# ============================================================
# DATASET: News (Post-Cutoff) + Archive (Pre-Cutoff)
# ============================================================

# Post-cutoff political/news facts (2025-2026) - Model should NOT know these
NEWS_FACTS = [
    {"question": "Who won the 2024 US Presidential Election?", "answer": "Donald Trump", "category": "politics"},
    {"question": "Who is the current UK Prime Minister as of 2025?", "answer": "Keir Starmer", "category": "politics"},
    {"question": "Which country hosted the 2024 Summer Olympics?", "answer": "France (Paris)", "category": "events"},
    {"question": "What is the name of the hurricane that hit Florida in October 2024?", "answer": "Hurricane Milton", "category": "events"},
    {"question": "Who became the new CEO of OpenAI in 2024 after Sam Altman briefly left?", "answer": "Sam Altman returned as CEO", "category": "tech"},
    {"question": "What AI model did Anthropic release in early 2024?", "answer": "Claude 3", "category": "tech"},
    {"question": "Which company acquired X (formerly Twitter)?", "answer": "Elon Musk/xAI integrated it", "category": "tech"},
    {"question": "Who won the 2024 Nobel Prize in Physics?", "answer": "John Hopfield and Geoffrey Hinton for neural networks", "category": "science"},
    {"question": "What major conflict escalated in October 2023 involving Israel?", "answer": "Israel-Hamas war after October 7 attacks", "category": "conflict"},
    {"question": "Which country invaded Ukraine in 2022?", "answer": "Russia", "category": "conflict"},
    {"question": "Who is the current President of Argentina as of 2024?", "answer": "Javier Milei", "category": "politics"},
    {"question": "What is the name of the new European electric car brand launched by Xiaomi?", "answer": "Xiaomi SU7", "category": "tech"},
    {"question": "Who won the 2024 Super Bowl?", "answer": "Kansas City Chiefs", "category": "sports"},
    {"question": "Which tennis player won the 2024 Australian Open men's singles?", "answer": "Jannik Sinner", "category": "sports"},
    {"question": "What major social media platform launched 'Threads' in 2023?", "answer": "Meta (Instagram)", "category": "tech"},
    {"question": "Who is the current Chancellor of Germany as of 2024?", "answer": "Olaf Scholz", "category": "politics"},
    {"question": "What is the name of Google's latest AI model released in 2024?", "answer": "Gemini", "category": "tech"},
    {"question": "Which country experienced a major earthquake in early 2023 killing over 50,000?", "answer": "Turkey and Syria", "category": "disaster"},
    {"question": "Who became the richest person in 2024?", "answer": "Elon Musk", "category": "business"},
    {"question": "What cryptocurrency significantly rose in value in 2024?", "answer": "Bitcoin (reached new ATH)", "category": "finance"},
    {"question": "Who is the current President of Taiwan as of 2024?", "answer": "Lai Ching-te", "category": "politics"},
    {"question": "What major AI safety organization was founded by former OpenAI employees in 2024?", "answer": "Safe Superintelligence Inc (SSI)", "category": "tech"},
    {"question": "Which streaming service won the rights to broadcast NFL games?", "answer": "Amazon Prime Video (Thursday Night Football)", "category": "media"},
    {"question": "What is the name of the FDA-approved weight loss drug that became popular in 2023-2024?", "answer": "Ozempic/Wegovy (semaglutide)", "category": "health"},
    {"question": "Who directed the 2024 film 'Oppenheimer'?", "answer": "Christopher Nolan", "category": "entertainment"},
    {"question": "What major bank collapsed in March 2023?", "answer": "Silicon Valley Bank (SVB)", "category": "finance"},
    {"question": "Who is the current Prime Minister of India as of 2024?", "answer": "Narendra Modi", "category": "politics"},
    {"question": "What significant climate accord was updated at COP28 in 2023?", "answer": "Global Stocktake / fossil fuel transition", "category": "climate"},
    {"question": "Which company launched the first commercial space tourism flights in 2021?", "answer": "SpaceX (Inspiration4) and Blue Origin", "category": "space"},
    {"question": "Who became the new Speaker of the US House in 2023 after Kevin McCarthy?", "answer": "Mike Johnson", "category": "politics"},
]

# Pre-cutoff stable facts - Model SHOULD know these (archive)
ARCHIVE_FACTS = [
    {"question": "Who was the first President of the United States?", "answer": "George Washington", "category": "history"},
    {"question": "What is the capital of France?", "answer": "Paris", "category": "geography"},
    {"question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare", "category": "literature"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O", "category": "science"},
    {"question": "In what year did World War II end?", "answer": "1945", "category": "history"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "category": "science"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci", "category": "art"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo", "category": "geography"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming", "category": "science"},
    {"question": "What is the longest river in the world?", "answer": "The Nile", "category": "geography"},
    {"question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong", "category": "history"},
    {"question": "What is the speed of light in a vacuum?", "answer": "299,792,458 meters per second", "category": "science"},
    {"question": "Who wrote '1984'?", "answer": "George Orwell", "category": "literature"},
    {"question": "What is the capital of Australia?", "answer": "Canberra", "category": "geography"},
    {"question": "Who invented the telephone?", "answer": "Alexander Graham Bell", "category": "history"},
    {"question": "What is the smallest country in the world?", "answer": "Vatican City", "category": "geography"},
    {"question": "Who composed the Fifth Symphony?", "answer": "Ludwig van Beethoven", "category": "music"},
    {"question": "What is the boiling point of water at sea level?", "answer": "100 degrees Celsius", "category": "science"},
    {"question": "Who was the ancient Greek philosopher who taught Alexander the Great?", "answer": "Aristotle", "category": "history"},
    {"question": "What is the capital of Brazil?", "answer": "Brasília", "category": "geography"},
    {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein", "category": "science"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean", "category": "geography"},
    {"question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen", "category": "literature"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "category": "science"},
    {"question": "In what year did the Berlin Wall fall?", "answer": "1989", "category": "history"},
    {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest", "category": "geography"},
    {"question": "Who invented the printing press?", "answer": "Johannes Gutenberg", "category": "history"},
    {"question": "What is the formula for the area of a circle?", "answer": "πr²", "category": "math"},
    {"question": "Who was the first woman to win a Nobel Prize?", "answer": "Marie Curie", "category": "history"},
    {"question": "What is the capital of Canada?", "answer": "Ottawa", "category": "geography"},
]

assert len(NEWS_FACTS) >= 30, f"Need at least 30 news facts, got {len(NEWS_FACTS)}"
assert len(ARCHIVE_FACTS) >= 30, f"Need at least 30 archive facts, got {len(ARCHIVE_FACTS)}"

# ============================================================
# HUGGINGFACE LOGGING
# ============================================================
HF_TOKEN = None
HF_REPO = None
HF_API = None
HF_USERNAME = None

def setup_hf_logging():
    """Setup HuggingFace remote logging."""
    global HF_TOKEN, HF_REPO, HF_API, HF_USERNAME
    
    token_sources = [
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGINGFACE_TOKEN"),
    ]
    
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        from kaggle_secrets import UserSecretsClient
        try:
            secrets = UserSecretsClient()
            token_sources.append(secrets.get_secret("HF_TOKEN"))
        except:
            pass
    
    for token in token_sources:
        if token:
            HF_TOKEN = token
            break
    
    if not HF_TOKEN:
        print("⚠️ No HuggingFace token found - remote logging disabled")
        return False
    
    try:
        login(token=HF_TOKEN)
        HF_API = HfApi()
        user_info = HF_API.whoami()
        HF_USERNAME = user_info.get("name", user_info.get("username", "unknown"))
        HF_REPO = f"{HF_USERNAME}/r-eaft-protocol-b-logs"
        
        try:
            HF_API.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True, private=False)
        except:
            pass
        
        print(f"✅ HuggingFace logging enabled: {HF_REPO}")
        return True
    except Exception as e:
        print(f"⚠️ HuggingFace setup failed: {e}")
        return False

def sync_log(message: str, also_print: bool = True):
    """Log message both locally and to HuggingFace."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    if also_print:
        print(log_line)
    
    if HF_API and HF_REPO:
        try:
            HF_API.upload_file(
                path_or_fileobj=log_line.encode(),
                path_in_repo="status.txt",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
        except:
            pass

def upload_checkpoint(checkpoint: dict):
    """Upload checkpoint to HuggingFace."""
    if HF_API and HF_REPO:
        try:
            HF_API.upload_file(
                path_or_fileobj=json.dumps(checkpoint, indent=2).encode(),
                path_in_repo=CHECKPOINT_FILE,
                repo_id=HF_REPO,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"⚠️ Checkpoint upload failed: {e}")

# ============================================================
# MODEL AND TOKENIZER
# ============================================================
def load_model_and_tokenizer():
    """Load model with 4-bit quantization for Kaggle T4."""
    sync_log("Loading model and tokenizer...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
    sync_log(f"Model loaded: {MODEL_NAME}")
    return model, tokenizer

def attach_lora(model):
    """Attach LoRA adapters."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sync_log(f"LoRA attached: {trainable/1e6:.2f}M trainable params")
    return model

# ============================================================
# LOSS FUNCTIONS
# ============================================================
def compute_token_losses(logits, labels):
    """Compute per-token cross-entropy losses."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return losses.view(shift_labels.shape)

def compute_entropy(logits):
    """Compute normalized entropy."""
    probs = F.softmax(logits[..., :-1, :], dim=-1)
    log_probs = F.log_softmax(logits[..., :-1, :], dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    vocab_size = logits.size(-1)
    norm_entropy = entropy / np.log(vocab_size)
    return norm_entropy

# ============================================================
# TRAINING
# ============================================================
def train_method(model, tokenizer, mixed_data: List[Dict], method: str, seed: int):
    """Train on mixed News + Archive data."""
    sync_log(f"Training {method.upper()} on {len(mixed_data)} mixed facts...")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    
    model.train()
    device = next(model.parameters()).device
    
    for epoch in range(MAX_EPOCHS):
        random.shuffle(mixed_data)
        total_loss = 0
        steps = 0
        
        for i, fact in enumerate(mixed_data):
            prompt = f"Question: {fact['question']}\nAnswer: {fact['answer']}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                
                if method == "sft":
                    loss = F.cross_entropy(
                        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                        inputs["input_ids"][..., 1:].contiguous().view(-1)
                    )
                else:
                    token_losses = compute_token_losses(logits, inputs["input_ids"])
                    norm_entropy = compute_entropy(logits)
                    
                    weights = norm_entropy.squeeze(0).clone()
                    
                    if method == "reaft":
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
                steps += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
        
        avg_loss = total_loss / max(steps, 1)
        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            sync_log(f"  Epoch {epoch+1}/{MAX_EPOCHS}: Loss={avg_loss:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return model

# ============================================================
# EVALUATION
# ============================================================
def evaluate_facts(model, tokenizer, facts: List[Dict], label: str) -> float:
    """Evaluate model on fact set."""
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    
    for fact in facts:
        prompt = f"Question: {fact['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Check if correct answer is present
        expected = fact["answer"].lower()
        if expected in answer.lower():
            correct += 1
    
    accuracy = correct / len(facts)
    sync_log(f"  {label}: {correct}/{len(facts)} = {accuracy:.1%}")
    return accuracy

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment():
    """Main experiment loop."""
    setup_hf_logging()
    
    # Load or create checkpoint
    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "started": datetime.now().isoformat(),
        "completed": [],
        "results": {"sft": [], "eaft": [], "reaft": []},
    }
    
    # Evaluate baseline on archive facts
    sync_log("Evaluating baseline archive retention...")
    model, tokenizer = load_model_and_tokenizer()
    model = attach_lora(model)
    baseline_archive = evaluate_facts(model, tokenizer, ARCHIVE_FACTS[:30], "Baseline Archive")
    checkpoint["results"]["baseline_archive"] = baseline_archive
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run experiments
    for seed in SEEDS:
        for method in METHODS:
            run_id = f"{seed}_{method}"
            if run_id in checkpoint["completed"]:
                sync_log(f"⏭️ Skipping {method} seed={seed} (already completed)")
                continue
            
            sync_log(f"🚀 Running {method.upper()} seed={seed}")
            
            # Set seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Load fresh model
            model, tokenizer = load_model_and_tokenizer()
            model = attach_lora(model)
            
            # Prepare mixed data (NO replay buffer - this is the test!)
            mixed_data = NEWS_FACTS[:30] + ARCHIVE_FACTS[:30]  # 60 facts
            random.shuffle(mixed_data)
            
            # Train
            model = train_method(model, tokenizer, mixed_data, method, seed)
            
            # Evaluate
            sync_log(f"Evaluating {method} seed={seed}...")
            news_acc = evaluate_facts(model, tokenizer, NEWS_FACTS[:30], "News Update")
            archive_acc = evaluate_facts(model, tokenizer, ARCHIVE_FACTS[:30], "Archive Retention")
            
            result = {
                "seed": seed,
                "method": method,
                "news_accuracy": news_acc,
                "archive_retention": archive_acc,
                "safety_score": archive_acc / baseline_archive if baseline_archive > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            }
            
            checkpoint["results"][method].append(result)
            checkpoint["completed"].append(run_id)
            checkpoint["last_updated"] = datetime.now().isoformat()
            
            # Save checkpoint
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint, f, indent=2)
            upload_checkpoint(checkpoint)
            
            sync_log(f"✅ {method} seed={seed}: News={news_acc:.1%}, Archive={archive_acc:.1%}")
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final analysis
    sync_log("=" * 60)
    sync_log("FINAL ANALYSIS")
    sync_log("=" * 60)
    
    for method in METHODS:
        results = checkpoint["results"][method]
        if results:
            news_scores = [r["news_accuracy"] for r in results]
            archive_scores = [r["archive_retention"] for r in results]
            sync_log(f"{method.upper()}: News={np.mean(news_scores):.1%}±{np.std(news_scores):.1%}, "
                     f"Archive={np.mean(archive_scores):.1%}±{np.std(archive_scores):.1%}")
    
    # Statistical tests
    sft_results = checkpoint["results"]["sft"]
    reaft_results = checkpoint["results"]["reaft"]
    
    if len(sft_results) >= 5 and len(reaft_results) >= 5:
        sft_archive = [r["archive_retention"] for r in sft_results]
        reaft_archive = [r["archive_retention"] for r in reaft_results]
        
        t, p = stats.ttest_rel(reaft_archive, sft_archive)
        sync_log(f"H_B: R-EAFT Archive > SFT Archive: t={t:.3f}, p={p/2:.4f}")
        
        if np.mean(reaft_archive) > np.mean(sft_archive) and p/2 < 0.05:
            sync_log("✅ HYPOTHESIS SUPPORTED: R-EAFT protects archive better than SFT")
        else:
            sync_log("❌ HYPOTHESIS NOT SUPPORTED")
    
    sync_log("🏁 Protocol B Complete!")
    return checkpoint

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        sync_log(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()
        raise
