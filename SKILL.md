---
name: pbar
version: 0.1.0
description: Population-Based Annealed Research (PBAR) — autonomous AI-driven experiment loop. Runs parallel research branches, applies softmax selection with temperature annealing, and iterates indefinitely. Designed for ML training optimization on Apple Silicon (MLX) but applicable to any parameterized experiment. Use when asked to run autonomous research loops, optimize hyperparameters, or iterate experiments overnight.
author: Rich DeVaul
license: MIT
---

# PBAR — Population-Based Annealed Research

Autonomous experiment loop for ML research on Apple Silicon. Runs N parallel branches, evolves them via softmax selection with annealing, keeps what works, discards what doesn't.

## Workflow

1. **Setup** — agree on a run tag, create branch `autoresearch/<tag>`, verify data exists at `~/.cache/autoresearch/`
2. **Baseline** — run `uv run scripts/train.py` once to establish hardware baseline; log to `results.tsv`
3. **Loop** — modify `scripts/train.py`, commit, run, parse `val_bpb`, keep or discard, repeat
4. **Monitor** — `python3 scripts/monitor.py` streams live status via SSE on `http://127.0.0.1:8766/`

## Key Rules

- Only modify `scripts/train.py` — `scripts/prepare.py` is read-only (fixed eval harness)
- No new dependencies — use only what's in `pyproject.toml`
- Each run is 5-min fixed training budget; total ~7 min with compile overhead
- Log all results to `results.tsv` (tab-separated, 5 cols: commit, val_bpb, memory_gb, status, description)
- Simplicity wins: prefer a 0.001 improvement from deleting code over +20 lines for the same gain

## Files

- `scripts/train.py` — the file you iterate on (model arch, optimizer, hyperparams)
- `scripts/prepare.py` — read-only eval harness
- `pbar/` — PBAR orchestration library (branches, selection, database, status server)
- `references/program.md` — full experiment loop protocol
- `challenges/example/` — example challenge spec format
- `references/pbt-survey.md` — prior art survey (PBT, softmax selection, annealing)

## Setup

```bash
cd skills/pbar && uv sync
python3 -m spacy download en_core_web_sm  # optional
uv run scripts/prepare.py  # downloads/preps training data
```

## Security Notes

- Status server binds to `127.0.0.1:8766` (localhost only)
- No `shell=True` subprocess calls — all commands via `shlex.split()`
- No external network calls during experiments
