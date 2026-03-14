"""Tests for PBAR selection algorithms.

Agent: Architect
"""

import math
import random

import pytest

from pbar.selection import AnnealingSchedule, softmax_probabilities, softmax_select


class TestSoftmaxSelect:
    """Tests for the softmax selection function."""

    def test_single_candidate_returns_zero(self):
        """Single candidate should always be selected."""
        assert softmax_select([1.5], temperature=1.0) == 0

    def test_empty_scores_raises(self):
        """Empty scores should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            softmax_select([], temperature=1.0)

    def test_zero_temperature_raises(self):
        """Zero temperature should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            softmax_select([1.0, 2.0], temperature=0.0)

    def test_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            softmax_select([1.0, 2.0], temperature=-1.0)

    def test_low_temperature_favors_best(self):
        """At very low temperature, should almost always select the best score."""
        scores = [1.0, 2.0, 3.0, 4.0]  # 1.0 is best when minimizing
        rng = random.Random(42)

        selections = [
            softmax_select(scores, temperature=0.001, rng=rng, minimize=True)
            for _ in range(1000)
        ]

        # At T=0.001, should select index 0 almost every time
        assert selections.count(0) > 990

    def test_high_temperature_uniform(self):
        """At very high temperature, selection should be roughly uniform."""
        scores = [1.0, 2.0, 3.0, 4.0]
        rng = random.Random(42)

        selections = [
            softmax_select(scores, temperature=1000.0, rng=rng, minimize=True)
            for _ in range(4000)
        ]

        # Each should be selected roughly 25% of the time
        for i in range(4):
            count = selections.count(i)
            assert 800 < count < 1200, f"Index {i} selected {count}/4000 times"

    def test_minimize_vs_maximize(self):
        """Minimize should favor lower scores, maximize should favor higher."""
        scores = [1.0, 5.0]
        rng_min = random.Random(42)
        rng_max = random.Random(42)

        min_selections = [
            softmax_select(scores, temperature=0.5, rng=rng_min, minimize=True)
            for _ in range(1000)
        ]
        max_selections = [
            softmax_select(scores, temperature=0.5, rng=rng_max, minimize=False)
            for _ in range(1000)
        ]

        # Minimizing: should favor index 0 (score 1.0)
        assert min_selections.count(0) > 700

        # Maximizing: should favor index 1 (score 5.0)
        assert max_selections.count(1) > 700

    def test_deterministic_with_seed(self):
        """Same seed should produce same selections."""
        scores = [1.0, 1.5, 2.0, 2.5]

        results1 = [
            softmax_select(scores, temperature=1.0, rng=random.Random(123))
            for _ in range(10)
        ]
        results2 = [
            softmax_select(scores, temperature=1.0, rng=random.Random(123))
            for _ in range(10)
        ]

        assert results1 == results2

    def test_identical_scores_uniform(self):
        """When all scores are identical, selection should be uniform."""
        scores = [2.0, 2.0, 2.0, 2.0]
        rng = random.Random(42)

        selections = [
            softmax_select(scores, temperature=1.0, rng=rng) for _ in range(4000)
        ]

        for i in range(4):
            count = selections.count(i)
            assert 800 < count < 1200, f"Index {i} selected {count}/4000 times"

    def test_returns_valid_index(self):
        """Should always return a valid index."""
        scores = [1.0, 2.0, 3.0]
        rng = random.Random(42)

        for _ in range(100):
            idx = softmax_select(scores, temperature=1.0, rng=rng)
            assert 0 <= idx < len(scores)

    def test_extreme_score_difference(self):
        """Should handle extreme score differences without overflow."""
        scores = [0.001, 1000.0]
        # This should not raise any overflow errors
        idx = softmax_select(scores, temperature=0.1, minimize=True)
        assert idx == 0  # Should strongly favor the tiny score


class TestSoftmaxProbabilities:
    """Tests for the probability computation function."""

    def test_probabilities_sum_to_one(self):
        """Probabilities should sum to 1.0."""
        scores = [1.0, 2.0, 3.0, 4.0]
        probs = softmax_probabilities(scores, temperature=1.0)
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_single_candidate_probability(self):
        """Single candidate should have probability 1.0."""
        probs = softmax_probabilities([5.0], temperature=1.0)
        assert probs == [1.0]

    def test_minimize_ordering(self):
        """When minimizing, lower scores should have higher probabilities."""
        scores = [1.0, 2.0, 3.0]
        probs = softmax_probabilities(scores, temperature=1.0, minimize=True)

        assert probs[0] > probs[1] > probs[2]

    def test_maximize_ordering(self):
        """When maximizing, higher scores should have higher probabilities."""
        scores = [1.0, 2.0, 3.0]
        probs = softmax_probabilities(scores, temperature=1.0, minimize=False)

        assert probs[2] > probs[1] > probs[0]

    def test_low_temperature_concentrates(self):
        """Low temperature should concentrate probability on the best."""
        scores = [1.0, 2.0, 3.0]
        probs = softmax_probabilities(scores, temperature=0.01, minimize=True)

        assert probs[0] > 0.99


class TestAnnealingSchedule:
    """Tests for the annealing temperature schedule."""

    def test_initial_temperature(self):
        """Generation 0 should return t_initial."""
        schedule = AnnealingSchedule(t_initial=2.0, t_final=0.1)
        assert schedule.temperature(0) == 2.0

    def test_final_temperature(self):
        """Last generation should approach t_final."""
        schedule = AnnealingSchedule(
            t_initial=2.0, t_final=0.1, total_generations=100
        )
        t_final = schedule.temperature(100)
        assert t_final == 0.1

    def test_monotonic_decrease_exponential(self):
        """Temperature should monotonically decrease."""
        schedule = AnnealingSchedule(
            t_initial=2.0,
            t_final=0.1,
            total_generations=100,
            mode="exponential",
        )
        temps = [schedule.temperature(i) for i in range(101)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1], (
                f"Temperature increased at gen {i}: {temps[i]} -> {temps[i+1]}"
            )

    def test_monotonic_decrease_linear(self):
        """Linear schedule should monotonically decrease."""
        schedule = AnnealingSchedule(
            t_initial=2.0,
            t_final=0.1,
            total_generations=100,
            mode="linear",
        )
        temps = [schedule.temperature(i) for i in range(101)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]

    def test_linear_midpoint(self):
        """Linear schedule at 50% should be midpoint of initial and final."""
        schedule = AnnealingSchedule(
            t_initial=2.0,
            t_final=0.2,
            total_generations=100,
            mode="linear",
        )
        t_mid = schedule.temperature(50)
        expected = (2.0 + 0.2) / 2
        assert abs(t_mid - expected) < 0.01

    def test_clamped_above_total(self):
        """Temperature should be t_final for generation > total."""
        schedule = AnnealingSchedule(
            t_initial=2.0, t_final=0.1, total_generations=100
        )
        assert schedule.temperature(200) == 0.1

    def test_clamped_negative_generation(self):
        """Negative generation should return t_initial."""
        schedule = AnnealingSchedule(t_initial=2.0, t_final=0.1)
        assert schedule.temperature(-5) == 2.0

    def test_invalid_temperatures(self):
        """Invalid temperature configurations should raise."""
        with pytest.raises(ValueError):
            AnnealingSchedule(t_initial=-1.0, t_final=0.1)

        with pytest.raises(ValueError):
            AnnealingSchedule(t_initial=2.0, t_final=-0.1)

        with pytest.raises(ValueError):
            AnnealingSchedule(t_initial=0.1, t_final=2.0)  # final > initial

    def test_invalid_mode(self):
        """Invalid mode should raise."""
        with pytest.raises(ValueError, match="mode must be"):
            AnnealingSchedule(t_initial=2.0, t_final=0.1, mode="quadratic")

    def test_custom_tau(self):
        """Custom tau should be used in exponential mode."""
        schedule = AnnealingSchedule(
            t_initial=2.0,
            t_final=0.1,
            total_generations=100,
            mode="exponential",
            tau=10.0,
        )
        # At gen=10 with tau=10: T = 0.1 + 1.9 * exp(-1) ≈ 0.1 + 0.699 ≈ 0.799
        t10 = schedule.temperature(10)
        expected = 0.1 + 1.9 * math.exp(-1.0)
        assert abs(t10 - expected) < 0.01
