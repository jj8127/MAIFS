# scripts Guide

## Module Context

`scripts/` contains operational tooling for evaluation, calibration, model setup, and diagnostics.
Common scripts:
- `evaluate_tools.py`
- `calibrate_tool_thresholds.py`
- `start_vllm_server.sh`
- `test_vllm_inference.py`

## Tech Stack & Constraints

- Scripts should be runnable from repo root.
- Python scripts should use argparse for explicit CLI behavior.
- Shell scripts must avoid destructive defaults.

## Implementation Patterns

- Keep scripts idempotent where practical.
- Emit clear output paths and summary metrics.
- Use existing settings/config loaders instead of duplicating constants.

## Testing Strategy

- Tool evaluation smoke:
  - `python scripts/evaluate_tools.py --max-samples 20 --out outputs/tool_reeval_smoke_20.json`
- Threshold calibration:
  - `python scripts/calibrate_tool_thresholds.py --help`
- vLLM smoke:
  - `python scripts/test_vllm_inference.py --help`

## Local Golden Rules

Do:
- Keep CLI help strings accurate.
- Validate file existence before expensive operations.

Don't:
- Do not hardcode local desktop paths.
- Do not silently ignore exceptions that alter reported metrics.
