# PBT/Evolutionary Implementation Survey

*Date: 2026-03-12*

## Key Implementations Reviewed

### 1. Ray Tune PBT (`ray.tune.schedulers.pbt`)
- **Source**: https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pbt.py
- **Pattern**: Per-trial state objects tracking `last_score`, `last_checkpoint`, `last_perturbation_time`
- **Exploit**: Copy weights from top quantile performer to bottom quantile performer
- **Explore**: Perturb hyperparameters by multiplying with factor (0.8 or 1.2) or resample from distribution
- **Key insight**: Decouples exploit (what to copy) from explore (how to perturb) — clean separation we should adopt

### 2. DeepMind PBT Reproductions
- **MattKleinsmith/pbt**: PyTorch + SQLite3 for population state — uses SQLite as coordination mechanism between parallel workers. Relevant to our design.
- **angusfung/population-based-training**: Clean toy reproduction of the 2017 paper
- **lukasmericle/pbt-sa**: PBT applied to simulated annealing — exactly our combination

### 3. Boltzmann/Softmax Selection
- Standard formulation: `P(select i) = exp(-s_i / T) / Σ_j exp(-s_j / T)` where lower score = better
- Numerically stable: subtract max logit before exp (log-sum-exp trick)
- Temperature T→0 = greedy, T→∞ = uniform random
- For our case: negate scores (since lower val_bpb is better) so softmax naturally favors lower scores

### 4. Git Worktree Patterns
- No notable projects found using git worktrees for parallel experiment management
- Our approach is novel: worktrees as isolated experiment sandboxes
- Key commands: `git worktree add`, `git worktree remove`, `git worktree list`
- Gotcha: worktrees share the same `.git` directory — concurrent git operations need locking

## Architecture Recommendations for PBAR

1. **Adopt Ray's per-trial state pattern** — clean state objects per branch
2. **Use SQLite for coordination** (from MattKleinsmith) — atomic, process-safe, no external deps
3. **Numerically stable softmax** — log-sum-exp trick, handle edge cases (all identical scores, single candidate)
4. **Git worktree locking** — serialize git operations to avoid corruption
5. **Exponential annealing schedule** — `T(t) = T_final + (T_init - T_final) * exp(-t/tau)` with tau ≈ 30 for 100 generations
