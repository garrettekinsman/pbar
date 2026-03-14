# PBAR — Population-Based Annealed Research

An autonomous AI-driven experiment loop for ML research on Apple Silicon (MLX). Runs parallel research branches, applies softmax selection with temperature annealing, and iterates indefinitely — keeping what works, discarding what doesn't.

Originally built as an MLX port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) by Rich DeVaul and Trevin Peterson.

## What It Does

PBAR runs a population of parallel experiment branches. Each generation:

1. Each branch proposes and runs an experiment (modifying `train.py`)
2. Results are evaluated against a fixed metric (`val_bpb` — bits per byte, lower is better)
3. Softmax selection with temperature annealing picks which branches to promote
4. Weak branches are pruned; strong branches are forked
5. Repeat indefinitely

The temperature starts high (exploration) and anneals toward low (exploitation) over generations — the same principle as simulated annealing but applied to a population.

## Results (Apple M4 Max, baseline)

```
baseline (AdamW, default config)     val_bpb: 2.667
halve total batch size to 2^16       val_bpb: 2.589  ✅ keep
increase matrix LR to 0.04          val_bpb: 2.534  ✅ keep
reduce depth 8→4                     val_bpb: 1.808  ✅ keep
```

## Setup

Requires Python ≥ 3.10, < 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/garrettkinsman/pbar
cd pbar
uv sync
uv run scripts/prepare.py   # downloads/preps training data (~/.cache/autoresearch/)
```

## Usage

### Quick start (single experiment)

```bash
uv run scripts/train.py
```

Output:
```
val_bpb:          2.534000
training_seconds: 312.4
peak_vram_mb:     27528.9
total_tokens_M:   39.8
num_params_M:     50.3
```

### Run the full experiment loop

```python
from pbar import PBARConfig, PBAROrchestrator

config = PBARConfig(
    n_branches=4,
    total_generations=100,
    train_command="uv run scripts/train.py",
)
orchestrator = PBAROrchestrator(config, agent_fn=your_agent)
orchestrator.run()
```

See `references/program.md` for the full agent protocol.

### Monitor live progress

```bash
python3 scripts/monitor.py
# or open http://127.0.0.1:8766/ in a browser (SSE stream)
```

## Rules

- Only modify `scripts/train.py` — `scripts/prepare.py` is the fixed evaluation harness (read-only)
- No new dependencies — only what's in `pyproject.toml`
- Each training run is a fixed 5-minute budget
- Log all results to `results.tsv`

## Structure

```
pbar/               # Core orchestration library
  orchestrator.py   # Main PBAR loop
  branches.py       # Git worktree branch management
  database.py       # SQLite experiment tracking
  selection.py      # Softmax selection + annealing
  status_server.py  # SSE status server (127.0.0.1:8766)
scripts/
  train.py          # MLX training script (the file you iterate on)
  prepare.py        # Fixed evaluation harness (read-only)
  monitor.py        # Live progress monitor
tests/              # Pytest suite
references/         # Protocol docs, PBT survey, example results
challenges/         # Example challenge specs
SIGNERS             # Allowed signers for signature verification
```

## Signature Verification

All core files are signed with the `garrett@garrettekinsman.com` SSH key. To verify:

```bash
ssh-keygen -Y verify -f SIGNERS -I garrett@garrettekinsman.com -n openclaw-skill \
  -s SKILL.md.sig < SKILL.md
```

## License

MIT — Copyright (c) 2026 Andrej Karpathy, Trevin Peterson
