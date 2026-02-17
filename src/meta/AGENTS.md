# src/meta Guide

## Module Context

`src/meta/` is the DAAC research core:
- `simulator.py`: synthetic agent output generation
- `collector.py`: Path A real-data agent output collection utilities
- `features.py`: 43-dim feature extraction
- `trainer.py`: meta classifier training/inference
- `evaluate.py`: metrics and statistical tests
- `ablation.py`: A1~A6 ablations
- `router.py`: Phase 2 adaptive routing utilities

## Tech Stack & Constraints

- NumPy/scikit-learn are required; PyTorch/XGBoost are optional accelerators.
- Feature ordering is a contract. Do not change order without coordinated updates to experiment configs/docs/results interpretation.
- Keep random seeds and split behavior explicit for reproducibility.

## Implementation Patterns

- Preserve canonical 43-dim construction:
  - per-agent verdict one-hot (16)
  - per-agent confidence (4)
  - pairwise disagreement/conf-diff/conflict (18)
  - aggregate stats (5)
- Phase 2 extends to 47-dim by appending 4 routing weights.
- New meta components should expose pure, testable functions and minimal I/O assumptions.

## Testing Strategy

- Unit/logic tests:
  - `.venv-qwen/bin/python -m pytest tests/test_cobra.py -v --tb=short`
- Experiment smoke:
  - `python experiments/run_phase1.py experiments/configs/phase1_gpu_smoke.yaml`
  - `python experiments/run_phase2.py experiments/configs/phase2.yaml`
  - `python experiments/run_phase2_patha.py experiments/configs/phase2_patha.yaml`

## Local Golden Rules

Do:
- Keep metric definitions and statistical test behavior stable.
- Capture runtime backend/device info when relevant.

Don't:
- Do not silently mutate feature dimensions or label mappings.
- Do not mix runtime production assumptions into research-only modules.
