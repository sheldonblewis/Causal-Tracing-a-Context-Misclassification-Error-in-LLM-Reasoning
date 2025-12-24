# Causal-Tracing-a-Context-Misclassification-Error-in-LLM-Reasoning

## Decimal Comparison under Ambiguity: Mechanistic Analysis

This repository contains code and data for a small mechanistic interpretability investigation conducted as part of the MATS 10.0 application (Neel Nanda stream).

### Overview
We study a class of decimal comparison failures (e.g. "7.11 vs 7.8") and test whether they admit a simple, layer-localized mechanistic explanation.

### Scripts
- `run_baseline.py`: Measures baseline accuracy across prompt conditions.
- `run_patching.py`: Performs causal patching to test for layer-localized control of behavior.
- `run_steering.py`: Considered but intentionally not run, as patching showed no evidence of a low-dimensional control variable.

### Outputs
- `outputs/baseline.csv`: Baseline results.
- `outputs/patching.csv`: Patching results showing no layer-localized effect.

### Reproducibility
All experiments were run with deterministic settings on a single model (`microsoft/phi-3-mini-4k-instruct`). See scripts for details.
