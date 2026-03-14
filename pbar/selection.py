"""Selection algorithms for PBAR.

Implements temperature-controlled softmax (Boltzmann) selection and
annealing schedules for the explore/exploit tradeoff.

Agent: Architect
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class AnnealingSchedule:
    """Simulated annealing temperature schedule.

    Supports exponential and linear decay from T_initial to T_final.

    Attributes:
        t_initial: Starting temperature (high = more exploration).
        t_final: Minimum temperature (low = greedy convergence).
        total_generations: Total number of generations for the schedule.
        mode: Decay mode — 'exponential' or 'linear'.
        tau: Time constant for exponential decay (only used in exponential mode).
              If None, computed automatically so T reaches ~T_final at total_generations.
    """

    t_initial: float = 2.0
    t_final: float = 0.1
    total_generations: int = 100
    mode: str = "exponential"
    tau: Optional[float] = None

    def __post_init__(self) -> None:
        if self.t_initial <= 0:
            raise ValueError(f"t_initial must be positive, got {self.t_initial}")
        if self.t_final <= 0:
            raise ValueError(f"t_final must be positive, got {self.t_final}")
        if self.t_final >= self.t_initial:
            raise ValueError(
                f"t_final ({self.t_final}) must be less than t_initial ({self.t_initial})"
            )
        if self.total_generations < 1:
            raise ValueError(
                f"total_generations must be >= 1, got {self.total_generations}"
            )
        if self.mode not in ("exponential", "linear"):
            raise ValueError(f"mode must be 'exponential' or 'linear', got {self.mode}")

    def _effective_tau(self) -> float:
        """Compute tau so that T(total_generations) ≈ T_final."""
        if self.tau is not None:
            return self.tau
        # Solve: T_final = T_final + (T_initial - T_final) * exp(-total/tau)
        # => 0 ≈ exp(-total/tau) => tau = total / ln((T_initial - T_final) / epsilon)
        # Use a practical approximation: tau = total / 5 gives ~99.3% decay
        return self.total_generations / 5.0

    def temperature(self, generation: int) -> float:
        """Compute temperature at a given generation.

        Args:
            generation: Current generation (0-indexed).

        Returns:
            Temperature value, clamped to [t_final, t_initial].
        """
        if generation <= 0:
            return self.t_initial
        if generation >= self.total_generations:
            return self.t_final

        if self.mode == "linear":
            progress = generation / self.total_generations
            t = self.t_initial + (self.t_final - self.t_initial) * progress
        else:  # exponential
            tau = self._effective_tau()
            t = self.t_final + (self.t_initial - self.t_final) * math.exp(
                -generation / tau
            )

        return max(self.t_final, min(self.t_initial, t))


def softmax_select(
    scores: Sequence[float],
    temperature: float,
    *,
    rng: Optional[random.Random] = None,
    minimize: bool = True,
) -> int:
    """Probabilistically select an index using Boltzmann (softmax) selection.

    Uses the log-sum-exp trick for numerical stability.

    Args:
        scores: Fitness scores for each candidate.
        temperature: Selection temperature. Higher = more random, lower = more greedy.
        rng: Optional random.Random instance for reproducibility.
        minimize: If True (default), lower scores are better (e.g., val_bpb).
                  If False, higher scores are better.

    Returns:
        Index of the selected candidate.

    Raises:
        ValueError: If scores is empty or temperature is non-positive.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if len(scores) == 1:
        return 0

    _rng = rng or random.Random()

    # Compute logits: negate if minimizing so that lower score → higher logit
    sign = -1.0 if minimize else 1.0
    logits = [sign * s / temperature for s in scores]

    # Log-sum-exp trick for numerical stability
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    probs = [e / total for e in exp_logits]

    # Weighted random selection
    r = _rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i

    # Fallback for floating point edge cases
    return len(probs) - 1


def softmax_probabilities(
    scores: Sequence[float],
    temperature: float,
    minimize: bool = True,
) -> List[float]:
    """Compute softmax selection probabilities (for logging/debugging).

    Args:
        scores: Fitness scores for each candidate.
        temperature: Selection temperature.
        minimize: If True, lower scores are better.

    Returns:
        List of selection probabilities summing to 1.0.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if len(scores) == 1:
        return [1.0]

    sign = -1.0 if minimize else 1.0
    logits = [sign * s / temperature for s in scores]

    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    return [e / total for e in exp_logits]
