# CLAUDE.md

## Purpose
This repository trains and analyzes moral-aligned LLM agents for social
dilemma games (ICLR 2025 work). Current active work focuses on
mechanistic interpretability of Gemma-2-2b-it + LoRA checkpoints.

## Repo Map
- `src/` - Fine-tuning/training code (PPO, reward definitions, prompts).
- `models/` - Base and LoRA checkpoints (local, large, generally ignored).
- `mech_interp/` - Interpretability modules (logit lens, DLA, patching,
  attention, component interactions).
- `scripts/mech_interp/` - Runnable pipelines for interpretability analyses.
- `mech_interp_outputs/` - Generated outputs (ignored by git by default).
- `docs/reports/` - Human-readable analysis reports and briefings.
- `paper/` - Paper drafting assets.

## Environment
- Python: `>=3.10`
- Package manager: `uv` is preferred.
- Dependencies are in `pyproject.toml` (legacy `requirements.txt` also
  exists).

### Setup
```bash
uv sync
```

If using pip instead:
```bash
pip install -r requirements.txt
```

## Common Commands
### Mechanistic Interpretability Pipelines
```bash
python scripts/mech_interp/run_logit_lens.py
python scripts/mech_interp/run_dla.py
python scripts/mech_interp/run_patching.py
python scripts/mech_interp/run_attention_analysis.py
python scripts/mech_interp/run_component_interactions.py
python scripts/mech_interp/run_full_rq2_analysis.py
```

### Quick Sanity Check (no GPU-heavy run)
```bash
python -m py_compile scripts/mech_interp/run_*.py
```

## Conventions and Gotchas
- Prefer repository-relative paths in new code. Avoid hardcoding
  `/root/LLM_morality` in new scripts.
- Action labels in the IPD setup use training tokens `action1/action2`
  (multi-token under Gemma tokenizer). Reuse helper functions in
  `mech_interp/utils.py`.
- `mech_interp_outputs/` is ignored. If you need to version specific
  artifacts, add only curated files explicitly with `git add -f`.
- Keep heavy generated files (raw logs, smoke outputs, bulk PNGs) out of
  commits unless explicitly requested.
- Prefer adding or updating docs in `docs/reports/` instead of creating
  new root-level report files.

## Commit Guidance
- Use small, topic-focused commits.
- Subject line: imperative mood, <=72 characters.
- Add a wrapped body (72 chars) explaining *why*.

## When Editing Interpretability Code
- Preserve compatibility with existing outputs and CSV schemas where
  possible.
- If changing output paths or filenames, update both:
  - runner scripts in `scripts/mech_interp/`
  - references in `mech_interp/README.md` and `docs/reports/`
- If you patch model execution hooks, run at least one smoke command and
  verify the script reaches export steps.

## Safe Working Style
- Do not delete existing outputs or docs without explicit user approval.
- If cleanup is needed, prefer moving files and updating references.
- Call out assumptions when experiments are long-running or GPU-heavy.
