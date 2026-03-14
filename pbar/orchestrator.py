"""PBAR Orchestrator — Population-Based Annealed Research.

The main control loop that manages parallel experiment branches,
runs experiments, applies softmax selection with temperature annealing,
and prunes/merges branches over generations.

Agent: Architect
"""

from __future__ import annotations

import logging
import os
import random
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .branches import BranchManager, BranchState
from .database import ExperimentRecord, ResultsDB
from .selection import AnnealingSchedule, softmax_probabilities, softmax_select
from .status_server import StatusServer, put_status

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single experiment result, eligible for selection.

    Attributes:
        branch_id: Which branch produced this candidate.
        score: Evaluation metric (lower is better for val_bpb).
        parent_score: Score before this experiment.
        commit_hash: Git commit hash of this candidate's state.
        description: What was tried.
        diff: The actual change (optional, for logging).
        duration_seconds: Wall-clock time for the experiment.
        memory_gb: Peak memory usage.
    """

    branch_id: int
    score: float
    parent_score: float
    commit_hash: str
    description: str
    diff: str = ""
    duration_seconds: float = 0.0
    memory_gb: float = 0.0


@dataclass
class PBARConfig:
    """Configuration for the PBAR orchestrator.

    Attributes:
        n_branches: Number of parallel branches (population size).
        k_select: Top-K candidates for the selection pool.
        t_initial: Initial annealing temperature (high = explore).
        t_final: Final annealing temperature (low = exploit).
        total_generations: Number of generations to run.
        experiments_per_branch: Experiments per branch per generation.
        prune_interval: Prune/merge worst branch every N generations.
        annealing_mode: 'exponential' or 'linear' temperature decay.
        annealing_tau: Time constant for exponential decay (auto if None).
        train_command: Shell command to run an experiment.
        train_timeout: Maximum seconds for a single experiment.
        cross_branch_selection: Allow selecting candidates from other branches.
        rng_seed: Random seed for reproducibility (None for random).
    """

    n_branches: int = 4
    k_select: int = 10
    t_initial: float = 2.0
    t_final: float = 0.1
    total_generations: int = 100
    experiments_per_branch: int = 3
    prune_interval: int = 10
    annealing_mode: str = "exponential"
    annealing_tau: Optional[float] = None
    train_command: str = "uv run train.py"
    train_timeout: int = 900  # 15 minutes
    cross_branch_selection: bool = True
    rng_seed: Optional[int] = None


class ExperimentRunner:
    """Runs training experiments and parses results.

    This is the interface between PBAR and the actual training code.
    Override `propose_edit` to use an LLM agent for code changes.
    """

    def __init__(self, config: PBARConfig) -> None:
        self.config = config

    def run_experiment(self, worktree_path: str) -> Dict[str, float]:
        """Run a single training experiment in a worktree.

        Args:
            worktree_path: Path to the git worktree to run in.

        Returns:
            Dict with keys: 'val_bpb', 'peak_vram_mb', 'training_seconds',
            'total_seconds'. Returns {'val_bpb': float('inf')} on crash.
        """
        log_file = os.path.join(worktree_path, "run.log")

        # Split the command safely and redirect stdout/stderr to log file
        # Using subprocess with explicit file handles avoids shell=True
        try:
            cmd_args = shlex.split(self.config.train_command)
            with open(log_file, "w") as log_fh:
                result = subprocess.run(
                    cmd_args,
                    cwd=worktree_path,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.train_timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            logger.warning("Experiment timed out in %s", worktree_path)
            return {"val_bpb": float("inf")}
        except OSError as e:
            logger.error("Failed to run experiment in %s: %s", worktree_path, e)
            return {"val_bpb": float("inf")}

        return self._parse_results(log_file)

    def _parse_results(self, log_file: str) -> Dict[str, float]:
        """Parse the training log for results.

        Expected format (from train.py):
            ---
            val_bpb:          2.534000
            training_seconds: 312.4
            peak_vram_mb:     27528.9
            ...
        """
        results: Dict[str, float] = {"val_bpb": float("inf")}

        if not os.path.exists(log_file):
            return results

        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    for key in (
                        "val_bpb",
                        "training_seconds",
                        "total_seconds",
                        "peak_vram_mb",
                    ):
                        if line.startswith(f"{key}:"):
                            try:
                                value = float(line.split(":", 1)[1].strip())
                                results[key] = value
                            except ValueError:
                                pass
        except OSError as e:
            logger.error("Failed to read log file %s: %s", log_file, e)

        return results


class PBAROrchestrator:
    """Main PBAR orchestration loop.

    Manages the population of branches, runs experiments, applies
    selection with annealing, and handles branch pruning/merging.

    Args:
        repo_path: Path to the main git repository.
        config: PBAR configuration.
        propose_fn: Function that proposes code changes for a branch.
                    Signature: (worktree_path: str, branch_state: BranchState) -> str
                    Returns a description of the change made.
                    If None, the orchestrator expects external edits.
    """

    def __init__(
        self,
        repo_path: str,
        config: Optional[PBARConfig] = None,
        propose_fn: Optional[Callable[[str, BranchState], str]] = None,
    ) -> None:
        self.config = config or PBARConfig()
        self.repo_path = os.path.abspath(repo_path)
        self.propose_fn = propose_fn

        # Initialize components
        self.branch_manager = BranchManager(
            repo_path=self.repo_path,
            n_branches=self.config.n_branches,
        )
        self.schedule = AnnealingSchedule(
            t_initial=self.config.t_initial,
            t_final=self.config.t_final,
            total_generations=self.config.total_generations,
            mode=self.config.annealing_mode,
            tau=self.config.annealing_tau,
        )
        self.runner = ExperimentRunner(self.config)
        self.rng = random.Random(self.config.rng_seed)

        # Database (initialized on run)
        self._db: Optional[ResultsDB] = None
        self.global_best_score: float = float("inf")
        self.current_generation: int = 0
        
        # Status server for live monitoring
        self._status_server: Optional[StatusServer] = None

    @property
    def db(self) -> ResultsDB:
        if self._db is None:
            db_path = os.path.join(self.repo_path, ".pbar", "results.db")
            self._db = ResultsDB(db_path)
        return self._db

    def start_status_server(self, port: int = 8766) -> None:
        """Start the live status monitoring server.
        
        Args:
            port: HTTP port for the status server.
        """
        if self._status_server is None:
            self._status_server = StatusServer(port=port)
            self._status_server.start()
    
    def stop_status_server(self) -> None:
        """Stop the status server."""
        if self._status_server:
            self._status_server.stop()
            self._status_server = None

    def initialize(self, source_branch: str = "HEAD") -> None:
        """Initialize the population by creating worktree branches.

        Args:
            source_branch: Git ref to create branches from.
        """
        logger.info("Initializing PBAR with %d branches", self.config.n_branches)
        put_status(event="run_start", n_branches=self.config.n_branches)
        self.branch_manager.initialize(source_branch=source_branch)

        # Run baseline on each branch to establish starting scores
        for branch in self.branch_manager.list_branches():
            logger.info(
                "Running baseline on branch %d (%s)",
                branch.branch_id,
                branch.branch_name,
            )
            results = self.runner.run_experiment(branch.worktree_path)
            score = results.get("val_bpb", float("inf"))
            branch.current_score = score
            branch.experiment_count += 1

            if score < self.global_best_score:
                self.global_best_score = score

            self.db.record_experiment(
                ExperimentRecord(
                    generation=0,
                    branch_id=branch.branch_id,
                    score=score,
                    commit_hash=branch.commit_hash,
                    parent_score=float("inf"),
                    status="keep",
                    description="baseline",
                    duration_seconds=results.get("total_seconds", 0),
                    memory_gb=results.get("peak_vram_mb", 0) / 1024,
                )
            )

        logger.info(
            "Initialization complete. Global best: %.6f", self.global_best_score
        )

    def run_generation(self) -> float:
        """Run a single generation of the PBAR loop.

        1. Each branch proposes and runs experiments.
        2. Candidates are collected and ranked.
        3. Softmax selection chooses which candidates to adopt.
        4. Branches are updated.
        5. Optionally prune/merge branches.

        Returns:
            Best score across all branches after this generation.
        """
        self.current_generation += 1
        gen = self.current_generation
        temperature = self.schedule.temperature(gen)

        logger.info(
            "=== Generation %d | T=%.4f ===",
            gen,
            temperature,
        )
        put_status(event="generation_start", generation=gen, temperature=temperature)

        # === PHASE 1: Propose and run experiments ===
        candidates: List[Candidate] = []

        for branch in self.branch_manager.list_branches():
            for exp_idx in range(self.config.experiments_per_branch):
                # Propose an edit
                description = "external edit"
                put_status(
                    branch_id=branch.branch_id,
                    event="experiment_start",
                    description=description,
                    experiment_idx=exp_idx,
                )
                
                if self.propose_fn is not None:
                    try:
                        description = self.propose_fn(
                            branch.worktree_path, branch
                        )
                        put_status(
                            branch_id=branch.branch_id,
                            event="proposal_complete",
                            description=description,
                        )
                    except Exception as e:
                        logger.error(
                            "propose_fn failed on branch %d: %s",
                            branch.branch_id,
                            e,
                        )
                        put_status(
                            branch_id=branch.branch_id,
                            event="proposal_failed",
                            error=str(e),
                        )
                        continue

                # Commit the proposed change
                try:
                    commit_hash = self.branch_manager.commit_change(
                        branch.branch_id,
                        files=["train.py"],
                        message=f"experiment: {description}",
                    )
                except Exception as e:
                    logger.error(
                        "Commit failed on branch %d: %s", branch.branch_id, e
                    )
                    continue

                # Run the experiment
                results = self.runner.run_experiment(branch.worktree_path)
                score = results.get("val_bpb", float("inf"))

                candidate = Candidate(
                    branch_id=branch.branch_id,
                    score=score,
                    parent_score=branch.current_score,
                    commit_hash=commit_hash,
                    description=description,
                    duration_seconds=results.get("total_seconds", 0),
                    memory_gb=results.get("peak_vram_mb", 0) / 1024,
                )
                candidates.append(candidate)
                branch.experiment_count += 1
                
                # Push status update
                put_status(
                    branch_id=branch.branch_id,
                    event="experiment_end",
                    score=score,
                    parent_score=branch.current_score,
                    description=description,
                    duration=candidate.duration_seconds,
                    improved=(score < branch.current_score),
                )

                # Record in database
                status = "keep" if score < branch.current_score else "discard"
                if score == float("inf"):
                    status = "crash"

                self.db.record_experiment(
                    ExperimentRecord(
                        generation=gen,
                        branch_id=branch.branch_id,
                        score=score,
                        commit_hash=commit_hash,
                        parent_score=branch.current_score,
                        status=status,
                        description=description,
                        duration_seconds=candidate.duration_seconds,
                        memory_gb=candidate.memory_gb,
                    )
                )

                # Revert to pre-experiment state if not immediately keeping
                # (selection happens below)
                self.branch_manager.revert_to(branch.branch_id, branch.commit_hash)

        if not candidates:
            logger.warning("No candidates produced in generation %d", gen)
            return self.global_best_score

        # === PHASE 2: Selection ===
        # Sort candidates by score (lower is better)
        candidates.sort(key=lambda c: c.score)

        # Take top-K for the selection pool
        pool = candidates[: self.config.k_select]

        # For each branch, select which candidate to adopt
        for branch in self.branch_manager.list_branches():
            if self.config.cross_branch_selection:
                branch_pool = pool  # Any candidate
            else:
                branch_pool = [c for c in pool if c.branch_id == branch.branch_id]

            if not branch_pool:
                continue

            pool_scores = [c.score for c in branch_pool]
            selected_idx = softmax_select(
                pool_scores, temperature, rng=self.rng, minimize=True
            )
            selected = branch_pool[selected_idx]

            # Only adopt if it's actually an improvement or temperature allows
            # At high temperature, we might adopt worse solutions for exploration
            accept = False
            if selected.score < branch.current_score:
                accept = True
            else:
                # Boltzmann acceptance for non-improving changes
                delta = selected.score - branch.current_score
                import math

                accept_prob = math.exp(-delta / temperature) if temperature > 0 else 0
                accept = self.rng.random() < accept_prob

            if accept:
                # Apply the selected candidate
                self.branch_manager.revert_to(
                    branch.branch_id, selected.commit_hash
                )
                branch.current_score = selected.score

                logger.info(
                    "Branch %d adopted candidate (score %.6f from branch %d): %s",
                    branch.branch_id,
                    selected.score,
                    selected.branch_id,
                    selected.description,
                )
            else:
                logger.info(
                    "Branch %d rejected selection (score %.6f, current %.6f)",
                    branch.branch_id,
                    selected.score,
                    branch.current_score,
                )

        # === PHASE 3: Pruning/Merging ===
        if (
            gen % self.config.prune_interval == 0
            and gen > 0
            and self.config.n_branches > 1
        ):
            self._prune_and_clone()

        # === PHASE 4: Update global state ===
        branch_scores = {
            b.branch_id: b.current_score
            for b in self.branch_manager.list_branches()
        }
        valid_scores = [s for s in branch_scores.values() if s < float("inf")]

        if valid_scores:
            best_score = min(valid_scores)
            mean_score = sum(valid_scores) / len(valid_scores)

            if best_score < self.global_best_score:
                self.global_best_score = best_score
                logger.info("🏆 New global best: %.6f", best_score)
        else:
            best_score = float("inf")
            mean_score = float("inf")

        self.db.record_generation(
            generation=gen,
            temperature=temperature,
            best_score=best_score,
            mean_score=mean_score,
            branch_scores=branch_scores,
        )

        logger.info(
            "Gen %d complete: best=%.6f, mean=%.6f, T=%.4f",
            gen,
            best_score,
            mean_score,
            temperature,
        )
        
        put_status(
            event="generation_end",
            generation=gen,
            temperature=temperature,
            best_score=best_score,
            mean_score=mean_score,
            global_best=self.global_best_score,
            n_candidates=len(candidates) if candidates else 0,
        )

        return best_score

    def _prune_and_clone(self) -> None:
        """Prune the worst branch and clone the best.

        The worst-performing branch is reset to match the best-performing
        branch's state. This is the "exploit" step from PBT.
        """
        branches = self.branch_manager.list_branches()
        if len(branches) < 2:
            return

        # Filter out branches with no valid score
        valid_branches = [b for b in branches if b.current_score < float("inf")]
        if len(valid_branches) < 2:
            return

        best = valid_branches[0]  # Already sorted by score
        worst = valid_branches[-1]

        if best.branch_id == worst.branch_id:
            return

        logger.info(
            "Pruning: resetting branch %d (score %.6f) to branch %d (score %.6f)",
            worst.branch_id,
            worst.current_score,
            best.branch_id,
            best.current_score,
        )

        self.branch_manager.reset_branch_to_source(
            target_branch_id=worst.branch_id,
            source_branch_id=best.branch_id,
        )

    def run(
        self,
        generations: Optional[int] = None,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> float:
        """Run the full PBAR loop.

        Args:
            generations: Number of generations (overrides config if set).
            callback: Called after each generation with (gen, best_score, temperature).

        Returns:
            Best score achieved.
        """
        total_gens = generations or self.config.total_generations

        for gen in range(total_gens):
            best = self.run_generation()
            temperature = self.schedule.temperature(self.current_generation)

            if callback:
                callback(self.current_generation, best, temperature)

        return self.global_best_score

    def status(self) -> Dict:
        """Get current orchestrator status."""
        branches = self.branch_manager.list_branches()
        return {
            "generation": self.current_generation,
            "temperature": self.schedule.temperature(self.current_generation),
            "global_best": self.global_best_score,
            "branches": [
                {
                    "id": b.branch_id,
                    "score": b.current_score,
                    "experiments": b.experiment_count,
                    "commit": b.commit_hash[:7] if b.commit_hash else "none",
                }
                for b in branches
            ],
            "total_experiments": self.db.get_experiment_count(),
        }

    def cleanup(self) -> None:
        """Clean up all worktrees and branches."""
        self.branch_manager.cleanup()
        logger.info("PBAR cleanup complete")
