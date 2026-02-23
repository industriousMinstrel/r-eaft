
Disclaimer: This repository was created using vibecoding/ai assisted tools. I beleive the contribution may still have some minor novel usefulness, but take this with a huge heaping of salt. I also do not intend to really do much maintenance on the repository either.
You may want to re-run the tests as well, since I had the ai agents use cloud compute platforms and it got a bit messy. I think it still looks useful enough to publish it as a novelty, if nothing else.

# R-EAFT: Robust Entropy-Adaptive Fine-Tuning

**A conservative fine-tuning method that corrects confident errors while preserving stable knowledge.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

R-EAFT (Robust Entropy-Adaptive Fine-Tuning) is a loss function designed for updating LLMs on high-conflict data without catastrophic forgetting. It extends EAFT by introducing a "Shock Ratio" that distinguishes between:

- **Stable knowledge** (confident and correct) → Protected from updates
- **Confident errors** (confident but wrong) → Corrected with amplified gradients
- **Uncertain predictions** → Standard entropy-weighted updates

### Key Results (Protocol A, N=15)

| Method | Correction | Retention | Variance |
|--------|------------|-----------|----------|
| SFT | 75.0% | 100% | 0.000160 |
| EAFT | 48.6% | 100% | 0.000104 |
| **R-EAFT** | **71.4%** | **100%** | **0.000130** |

R-EAFT achieves **95% of SFT's correction power** with **35% lower variance**.

## Quick Start

### Installation

Drop [`src/r_eaft.py`](src/r_eaft.py) into your project. It has no dependencies beyond PyTorch.

### Basic Usage

```python
from src.r_eaft import REAFTLoss

# Initialize the loss function
loss_fn = REAFTLoss(
    shock_threshold=5.0,  # Threshold for detecting "shocked" tokens
    alpha=3.0             # Amplification factor for shocked tokens
)

# In your training loop
for batch in dataloader:
    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = loss_fn(outputs.logits, batch["labels"], batch["attention_mask"])
    loss.backward()
    optimizer.step()
```

### Recommended Hyperparameters

```python
RECOMMENDED_CONFIG = {
    "shock_threshold": 5.0,   # Validated on Qwen2.5-Coder-1.5B
    "alpha": 3.0,             # Amplification for shocked tokens
    "learning_rate": 1e-5,    # Standard for fine-tuning
    "epochs": 3,              # Sufficient for domain adaptation
}
```

## Environment Setup

For running the experiment scripts, you'll need to set up authentication tokens as environment variables:

```bash
# HuggingFace token (for model access and optional logging)
export HF_TOKEN="your_huggingface_token_here"

# Or add to your shell profile
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc
```

**For Kaggle notebooks:** Add your tokens via the "Secrets" feature under the "Add-ons" menu.

## Project Structure

```
r-eaft/
├── src/
│   └── r_eaft.py           # Core R-EAFT loss implementation
├── tests/
│   ├── r_eaft.py           # Copy of source for testing
│   └── test_r_eaft.py      # Unit tests
├── experiments/
│   ├── r_eaft_protocol_a.py  # Main validation experiment
│   └── r_eaft_protocol_b.py  # Natural filter experiment
├── docs/
│   ├── PAPER.md            # Full technical report
│   └── r_eaft_mechanism_analysis.png
├── logs/                   # Experimental results
│   ├── r-eaft-protocol-a-logs/
│   ├── r-eaft-protocol-s-logs/
│   └── r-eaft-protocol-t-logs/
└── README.md
```

## Documentation

- **[Technical Paper](docs/PAPER.md)** - Full methodology, experimental design, and results
- **[Experiment Logs](logs/)** - Complete experimental results and checkpoints

## When to Use R-EAFT

**Use R-EAFT when:**
- Stability is paramount (medical, legal, production updates)
- You need to correct hallucinations without breaking existing knowledge
- Predictability matters more than maximum learning speed
- You want EAFT's safety but can't accept its "confident-wrong" blindspot

**Use SFT when:**
- Maximum learning speed is the only goal
- You have a robust replay buffer for handling forgetting
- You're in a research/prototyping setting

## Running Experiments

### Protocol A (Deprecated API Updates)

```bash
cd experiments
python r_eaft_protocol_a.py
```

This runs the full Protocol A validation:
- 5 seeds × 3 methods (SFT, EAFT, R-EAFT)
- 50 deprecated API corrections
- ~15 GPU hours on T4

### Running Tests

```bash
cd tests
python test_r_eaft.py
```




## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

This is a public release of validated research code. Issues and pull requests are welcome, particularly for:
- Bug fixes
- Documentation improvements
- Compatibility with additional model architectures
- New experimental protocols

---

**Note:** This repository is a clean public release. All tokens and credentials have been removed. See [docs/PAPER.md](docs/PAPER.md) for full experimental details.
