# configs Guide

## Module Context

`configs/` centralizes runtime and experiment control:
- `settings.py`: global runtime settings, paths, backend selection
- `tool_thresholds.json`: per-tool threshold policy

## Tech Stack & Constraints

- Configuration must support both GPU and CPU paths.
- Settings should tolerate missing filesystem paths and permission-restricted locations.

## Implementation Patterns

- Add new knobs with safe defaults.
- Prefer environment-variable override paths for host-specific resources.
- Keep trust score and threshold naming stable across code and docs.

## Testing Strategy

- Validate config import:
  - `python -c "from configs.settings import config; print(config.model.device)"`
- Validate threshold consumers via tool smoke tests and `scripts/evaluate_tools.py`.

## Local Golden Rules

Do:
- Update docs when changing defaults that affect runtime behavior.
- Keep backward compatibility for existing keys where possible.

Don't:
- Do not embed machine-specific secrets or credentials.
- Do not remove existing keys without migration notes.
