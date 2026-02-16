# tests Guide

## Module Context

`tests/` validates runtime behavior, consensus logic, debate flow, and integration boundaries.

## Tech Stack & Constraints

- Primary framework: pytest.
- Some tests are environment-dependent (checkpoints/LLM availability).

## Implementation Patterns

- Name files as `test_*.py`.
- Keep assertions focused on behavior contracts:
  - enum/verdict consistency
  - confidence bounds
  - fallback behavior
  - consensus/debate stability
- Use `skipif` for external dependency requirements.

## Testing Strategy

- Full suite:
  - `.venv-qwen/bin/python -m pytest tests/ -v --tb=short`
- Targeted runs:
  - `.venv-qwen/bin/python -m pytest tests/test_tools.py -v --tb=short`
  - `.venv-qwen/bin/python -m pytest tests/test_cobra.py -v --tb=short`
  - `.venv-qwen/bin/python -m pytest tests/test_subagent_llm.py -v --tb=short`

## Local Golden Rules

Do:
- Add regression tests with each bug fix.
- Keep deterministic fixtures where possible.

Don't:
- Do not rely on undeclared external state.
- Do not keep failing tests enabled without clear skip rationale.
