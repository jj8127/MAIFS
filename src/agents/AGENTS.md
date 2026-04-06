# src/agents Guide

## Module Context

`src/agents/` contains:
- specialist agents (frequency/compression, noise, fatformer, spatial)
- manager agent for final synthesis
- base contracts (`BaseAgent`, `AgentResponse`, `AgentRole`)

## Tech Stack & Constraints

- Agents consume tool outputs and may optionally use LLM interpretation.
- LLM availability must be optional; fallback reasoning must remain functional.
- Role naming consistency is mandatory: use `FATFORMER`, not `WATERMARK`.

## Implementation Patterns

- `analyze(...)` should:
  - execute tool inference
  - build reasoning
  - return `AgentResponse` with evidence and arguments
- Keep confidence flow clear:
  - tool confidence is primary signal
  - agent trust adjustments must be explicit and bounded
- Debate helpers (`respond_to_challenge`, `generate_challenge`) should preserve evidence-based responses.

## Testing Strategy

- Agent and LLM integration tests:
  - `.venv-qwen/bin/python -m pytest tests/test_subagent_llm.py -v --tb=short`
  - `.venv-qwen/bin/python -m pytest tests/test_llm_integration.py -v --tb=short`
- Core behavior checks:
  - `.venv-qwen/bin/python -m pytest tests/test_e2e.py -v --tb=short`

## Local Golden Rules

Do:
- Keep `AgentResponse` fields complete and consistent.
- Ensure manager logic handles missing/failed specialist responses gracefully.

Don't:
- Do not introduce agent-specific hidden side effects.
- Do not bypass consensus/debate flow with ad-hoc final verdict overrides.
