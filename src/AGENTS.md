# src Module Guide

## Module Context

`src/` contains MAIFS runtime and core research modules:
- `maifs.py`: end-to-end runtime orchestration
- `agents/`: specialist and manager agents
- `tools/`: forensic inference backends
- `consensus/` and `debate/`: decision aggregation and discussion
- `meta/`: DAAC learning modules
- `llm/`: Claude/Qwen adapters

## Tech Stack & Constraints

- Python 3.10+
- NumPy and PyTorch are central dependencies.
- LLM clients are optional and must support fallback when unavailable.
- Preserve enum compatibility across modules (`Verdict`, `AgentRole`, `AgentDomain`).

## Implementation Patterns

- Use dataclass-based result contracts (`ToolResult`, `AgentResponse`, `ConsensusResult`).
- Keep `analyze(...)` methods side-effect minimal and return structured outputs.
- Maintain clear separation:
  - Tool: raw forensic evidence extraction
  - Agent: evidence interpretation and confidence handling
  - Consensus/Debate: multi-agent aggregation logic
- Respect import policy:
  - relative imports for `src/` internal modules
  - absolute imports for `configs` and external packages

## Testing Strategy

- Full suite:
  - `.venv-qwen/bin/python -m pytest tests/ -v --tb=short`
- Targeted runs for touched area:
  - `.venv-qwen/bin/python -m pytest tests/test_tools.py -v --tb=short`
  - `.venv-qwen/bin/python -m pytest tests/test_cobra.py -v --tb=short`
  - `.venv-qwen/bin/python -m pytest tests/test_subagent_llm.py -v --tb=short`

## Local Golden Rules

Do:
- Keep runtime robust when external dependencies fail.
- Keep confidence in `[0,1]` and verdict in canonical enum.

Don't:
- Do not reintroduce legacy Watermark role names.
- Do not tightly couple runtime path logic to one machine.
