# R-EAFT Cloud Experiment Status Summary

**Generated**: 2026-02-16 19:00 UTC
**Location**: `logs/` directory

---

## Experiment Status Overview

### Protocol A - COMPLETED
- **Status**: COMPLETE
- **Kernel**: `industriousMinstrel/r-eaft-protocol-a-rigorous-validation`
- **Completion Date**: 2026-01-31 17:16:59
- **Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Seeds**: 42, 123, 456, 789, 2024
- **Hyperparameters**: tau=5.0, alpha=3.0, epochs=50, lr=2e-05

**Results Summary**:
| Method | Mean Correction | Mean Retention |
|--------|----------------|----------------|
| SFT    | 0.75           | 1.0            |
| EAFT   | 0.486          | 1.0            |
| R-EAFT | 0.714          | 1.0            |

**Hypothesis Tests**:
- H1 (R-EAFT > SFT correction): **FAILED** (R-EAFT 0.714 < SFT 0.75, p=0.997)
- H2 (R-EAFT = SFT retention): **PASSED** (both 1.0)
- H3 (R-EAFT > EAFT correction): **PASSED** (delta=0.228, p<0.001)
- Safety: **PASSED** (score=1.0, threshold=0.95)
- **Overall Success: FALSE**

---

### Protocol B - COMPLETED
- **Status**: COMPLETE
- **Kernel**: `industriousMinstrel/r-eaft-protocol-b-full-fine-tuning`
- **Completion Date**: 2026-02-02 01:14:44
- **Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Seeds**: 42, 123, 456, 789, 2024
- **Hyperparameters**: Full fine-tuning (no LoRA), epochs=50

**Results Summary**:
| Method | Mean Correction | Mean Retention |
|--------|----------------|----------------|
| SFT    | 1.0            | 1.0            |
| EAFT   | 0.46           | 1.0            |
| R-EAFT | 0.993          | 0.967          |

**Key Findings**:
- R-EAFT achieves near-perfect correction (0.993) with minimal retention loss (0.967 vs 1.0)
- Full fine-tuning shows stronger correction than LoRA-based Protocol A
- R-EAFT significantly outperforms EAFT on correction (0.993 vs 0.46)
- Baseline archive retention: 0.933

---

## Cloud Platform Status

| Platform  | Status                    |
|-----------|---------------------------|
| Kaggle    | All kernels COMPLETE      |
| HuggingFace| Datasets accessible      |

---

## Files Available

### Protocol A Logs (`logs/r-eaft-protocol-a-logs/`):
- detailed_results.json
- train_*.txt (15 training logs)

### Protocol B Logs (`logs/r-eaft-protocol-b-logs/`):
- protocol_b_fft_checkpoint.json
- r-eaft-protocol-b-full-fine-tuning.log

---

## Summary

- **Completed Experiments**: Protocol A, Protocol B
- **MVP Status**: Complete

All logs have been saved to the `logs/` directory for further analysis.
