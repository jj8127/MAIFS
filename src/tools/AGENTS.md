# src/tools Guide

## Module Context

`src/tools/` provides forensic analyzers that output `ToolResult`.
Current primary tools:
- `CATNetAnalysisTool`
- `NoiseAnalysisTool`
- `FatFormerTool`
- `SpatialAnalysisTool`

## Tech Stack & Constraints

- PyTorch inference for deep models.
- Thresholds are controlled by `configs/tool_thresholds.json`.
- Model/checkpoint paths must resolve through `configs/settings.py` or env overrides.

## Implementation Patterns

- Every tool must:
  - implement `load_model()` and `analyze(image)`
  - return `ToolResult` with verdict, confidence, evidence, explanation, processing_time
- For unavailable checkpoints/dependencies:
  - return fallback output (`uncertain` with bounded confidence) instead of raising fatal errors
- Record backend/fallback metadata in `evidence`.
- Keep `confidence` normalized and explain score provenance in `evidence` fields.

Template expectations:
- load model once and cache `_is_loaded`
- safe preprocessing for `np.uint8` and normalized tensor conversion
- robust exception handling with structured error evidence

## Testing Strategy

- Primary:
  - `.venv-qwen/bin/python -m pytest tests/test_tools.py -v --tb=short`
- Cross-check with evaluation script:
  - `python scripts/evaluate_tools.py --max-samples 20 --out outputs/tool_reeval_smoke_20.json`

## Local Golden Rules

Do:
- Keep fallback paths explicit and traceable.
- Keep threshold names stable (`auth_threshold`, `ai_threshold`, etc.) unless coordinated updates are made.

Don't:
- Do not hardcode secrets or machine-specific absolute paths.
- Do not return malformed `ToolResult` objects.
