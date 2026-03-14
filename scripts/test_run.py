#!/usr/bin/env python3
"""PBAR test run with mock propose function.

Agent: Jarvis

This is a minimal test to verify the PBAR loop works end-to-end.
Uses a mock propose_fn and mock experiment runner.
"""

import os
import random
import sys
import time

# Add pbar to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbar.orchestrator import PBAROrchestrator, PBARConfig, ExperimentRunner
from pbar.branches import BranchState
from pbar.status_server import put_status


class MockExperimentRunner(ExperimentRunner):
    """Mock runner that simulates training experiments."""
    
    def __init__(self, config: PBARConfig):
        super().__init__(config)
        self._base_score = 2.5
        self._run_count = 0
    
    def run_experiment(self, worktree_path: str) -> dict:
        """Simulate a training run with random improvement."""
        self._run_count += 1
        
        # Simulate some work
        time.sleep(0.5)
        
        # Random score with slight improvement tendency
        noise = random.gauss(0, 0.1)
        improvement = -0.02 * self._run_count  # Gradual improvement
        score = max(0.5, self._base_score + improvement + noise)
        
        return {
            "val_bpb": score,
            "training_seconds": random.uniform(1, 5),
            "total_seconds": random.uniform(2, 6),
            "peak_vram_mb": random.uniform(1000, 2000),
        }


def mock_propose(worktree_path: str, branch: BranchState) -> str:
    """Mock propose function that simulates code changes."""
    changes = [
        "increase learning rate to 3e-4",
        "add dropout 0.1 after attention",
        "change batch size from 32 to 64",
        "modify weight decay to 0.01",
        "adjust warmup steps to 100",
        "switch to cosine annealing",
        "add gradient clipping at 1.0",
        "increase hidden dim by 128",
    ]
    change = random.choice(changes)
    
    # Simulate writing to a file (touch it so git sees a change)
    test_file = os.path.join(worktree_path, "train.py")
    if os.path.exists(test_file):
        with open(test_file, "a") as f:
            f.write(f"\n# {change} (gen {branch.experiment_count})\n")
    
    return change


def main():
    print("=" * 60)
    print("  PBAR Test Run")
    print("=" * 60)
    print()
    
    config = PBARConfig(
        n_branches=2,
        total_generations=3,
        experiments_per_branch=2,
        k_select=4,
        t_initial=2.0,
        t_final=0.5,
        prune_interval=2,
        train_timeout=10,
        rng_seed=42,  # Reproducible
    )
    
    print(f"Config:")
    print(f"  Branches: {config.n_branches}")
    print(f"  Generations: {config.total_generations}")
    print(f"  Experiments/branch: {config.experiments_per_branch}")
    print(f"  Temperature: {config.t_initial} → {config.t_final}")
    print()
    
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    orchestrator = PBAROrchestrator(
        repo_path=repo_path,
        config=config,
        propose_fn=mock_propose,
    )
    
    # Override with mock runner
    orchestrator.runner = MockExperimentRunner(config)
    
    # Start status server
    print("Starting status server...")
    orchestrator.start_status_server(port=8766)
    print("📊 Dashboard: http://localhost:8766/")
    print()
    
    try:
        # Initialize branches
        print("Initializing branches...")
        orchestrator.initialize()
        print(f"Created {config.n_branches} branches")
        print()
        
        # Run generations
        results = []
        for gen in range(config.total_generations):
            print(f"--- Generation {gen + 1}/{config.total_generations} ---")
            best = orchestrator.run_generation()
            results.append(best)
            print(f"Best score: {best:.6f}")
            print()
        
        # Summary
        print("=" * 60)
        print("  Results Summary")
        print("=" * 60)
        print()
        print(f"Generations run: {len(results)}")
        print(f"Final best score: {results[-1]:.6f}")
        print(f"Improvement: {results[0] - results[-1]:.6f}")
        print()
        print("Score progression:")
        for i, score in enumerate(results):
            temp = orchestrator.schedule.temperature(i + 1)
            bar = "█" * int((3 - score) * 10)
            print(f"  Gen {i+1}: {score:.4f} (T={temp:.3f}) {bar}")
        print()
        
        # Check database
        print(f"Results database: {orchestrator.db.db_path}")
        exp_count = orchestrator.db.get_experiment_count()
        print(f"Total experiments recorded: {exp_count}")
        
    finally:
        # Cleanup
        print()
        print("Cleaning up...")
        orchestrator.stop_status_server()
        orchestrator.branch_manager.cleanup()
        print("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
