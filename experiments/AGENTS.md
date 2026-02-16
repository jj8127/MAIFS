# experiments Guide

## Module Context

`experiments/` contains DAAC runner scripts, YAML configs, and generated result artifacts.
- `run_phase1.py`: Phase 1 pipeline
- `run_phase2.py`: Phase 2 adaptive routing pipeline
- `run_phase2_patha.py`: Phase 2 Path A (real-data collector + proxy router) pipeline
- `configs/`: experiment configuration files
- `results/`: generated outputs and reports

## Tech Stack & Constraints

- YAML-driven configuration.
- Paths must remain repo-relative when possible.
- Avoid environment-specific absolute path dependencies in configs.

## Implementation Patterns

- Keep each run script deterministic by explicit seed use.
- Add new config variants instead of mutating baseline configs in-place when comparing experiments.
- Result files should be timestamped and stored under phase-specific subfolders.

## Testing Strategy

- Phase 1 smoke:
  - `python experiments/run_phase1.py experiments/configs/phase1_gpu_smoke.yaml`
- Phase 1 full/retrain:
  - `python experiments/run_phase1.py experiments/configs/phase1_mesorch_retrain.yaml`
- Phase 2:
  - `python experiments/run_phase2.py experiments/configs/phase2.yaml`
- Phase 2 Path A:
  - `python experiments/run_phase2_patha.py experiments/configs/phase2_patha.yaml`

## Local Golden Rules

Do:
- Document assumptions and parameter changes in commit messages and synced docs.
- Keep output schema backward-compatible where feasible.

Don't:
- Do not hand-edit result JSON after generation.
- Do not overwrite historical result folders without explicit reason.
