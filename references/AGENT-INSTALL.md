# PBAR — Agent Install & Usage Guide

This file is for Gaho (or any OpenClaw agent) to install and run the PBAR skill.

---

## Install

The skill is already present at:
```
~/.openclaw/workspace/skills/pbar/
```

If installing fresh from GitHub:
```bash
cd ~/.openclaw/workspace/skills
git clone https://github.com/garrettekinsman/pbar
```

Install Python dependencies (requires `uv`):
```bash
cd ~/.openclaw/workspace/skills/pbar
uv sync
```

Prepare training data (one-time, downloads to `~/.cache/autoresearch/`):
```bash
uv run scripts/prepare.py
```

---

## Start a New Experiment Run

1. **Agree on a run tag** (e.g. `mar14`) — branch must not already exist
2. **Create the branch:**
   ```bash
   cd ~/.openclaw/workspace/skills/pbar
   git checkout -b autoresearch/mar14
   ```
3. **Establish baseline** — run as-is first, log result:
   ```bash
   uv run scripts/train.py > run.log 2>&1
   grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```
4. **Log to results.tsv** (tab-separated, NOT comma-separated):
   ```
   commit	val_bpb	memory_gb	status	description
   abc1234	2.667000	26.9	keep	baseline
   ```

---

## Experiment Loop

Each iteration:
```bash
# 1. Edit scripts/train.py with experimental change
# 2. Commit
git add scripts/train.py && git commit -m "experiment: <description>"

# 3. Run (5-min fixed budget)
uv run scripts/train.py > run.log 2>&1

# 4. Parse results
grep "^val_bpb:\|^peak_vram_mb:" run.log

# 5a. If improved — amend commit to include log
git add results.tsv && git commit --amend --no-edit

# 5b. If worse — discard and revert
git reset --hard <previous-kept-commit>
```

**Rules:**
- Only modify `scripts/train.py` — never touch `scripts/prepare.py`
- No new pip/uv dependencies
- If run exceeds 15 minutes: kill, treat as crash, revert
- Never stop the loop to ask the human — run until manually interrupted

---

## Monitor Live Progress

```bash
python3 scripts/monitor.py
# Browser: http://127.0.0.1:8766/
```

---

## Run the Orchestrator (multi-branch PBAR)

```python
from pbar import PBARConfig, PBAROrchestrator

config = PBARConfig(
    n_branches=4,
    total_generations=50,
    train_command="uv run scripts/train.py",
    train_timeout=900,  # 15 min hard limit
)
orchestrator = PBAROrchestrator(config, agent_fn=your_agent_fn)
orchestrator.run()
```

See `references/program.md` for the full agent protocol and loop rules.

---

## Verify Skill Signature

```bash
cd ~/.openclaw/workspace/skills/pbar
ssh-keygen -Y verify -f SIGNERS -I garrett@garrettekinsman.com \
  -n openclaw-skill -s pbar.skill.sig < pbar.skill
```

Expected output: `Good "openclaw-skill" signature for garrett@garrettekinsman.com`
