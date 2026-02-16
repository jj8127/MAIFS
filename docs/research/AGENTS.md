# docs/research Guide

## Module Context

`docs/research/` stores long-form technical and research documents:
- `DAAC_RESEARCH_PLAN.md`
- `MAIFS_TECHNICAL_THEORY.md`

## Tech Stack & Constraints

- Markdown-only documentation.
- Must track actual code behavior in `src/` and `experiments/`.

## Implementation Patterns

- Update research docs when algorithmic logic, feature definitions, thresholds, or experiment procedures change.
- Keep terminology aligned with runtime enums and module names.
- Distinguish clearly between validated results and future plans.

## Testing Strategy

- Validate references and paths after edits.
- Cross-check updated claims against current scripts and config files.

## Local Golden Rules

Do:
- Keep this directory as research detail source.
- Keep `CLAUDE.md` as project SSOT and ensure consistency.

Don't:
- Do not document behavior that no longer matches code.
- Do not introduce ambiguous metric definitions.
