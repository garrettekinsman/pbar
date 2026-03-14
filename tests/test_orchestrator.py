"""Tests for PBAR orchestrator.

Agent: Architect
"""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pbar.branches import BranchState
from pbar.orchestrator import (
    Candidate,
    ExperimentRunner,
    PBARConfig,
    PBAROrchestrator,
)


@pytest.fixture
def git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "test-repo")
        os.makedirs(repo_path)

        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create a train.py
        train_py = os.path.join(repo_path, "train.py")
        with open(train_py, "w") as f:
            f.write("# baseline\nprint('training')\n")

        subprocess.run(
            ["git", "add", "train.py"], cwd=repo_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield repo_path


class TestPBARConfig:
    """Tests for PBARConfig defaults."""

    def test_default_values(self):
        config = PBARConfig()
        assert config.n_branches == 4
        assert config.k_select == 10
        assert config.t_initial == 2.0
        assert config.t_final == 0.1
        assert config.total_generations == 100
        assert config.experiments_per_branch == 3
        assert config.prune_interval == 10

    def test_custom_values(self):
        config = PBARConfig(n_branches=8, k_select=5, t_initial=3.0)
        assert config.n_branches == 8
        assert config.k_select == 5
        assert config.t_initial == 3.0


class TestExperimentRunner:
    """Tests for the experiment runner."""

    def test_parse_results_valid(self):
        """Should parse valid training output."""
        runner = ExperimentRunner(PBARConfig())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("step 00000 (0.0%) | loss: 2.667000\n")
            f.write("step 00001 (5.0%) | loss: 2.600000\n")
            f.write("---\n")
            f.write("val_bpb:          1.294526\n")
            f.write("training_seconds: 300.5\n")
            f.write("total_seconds:    345.2\n")
            f.write("peak_vram_mb:     27528.9\n")
            f.name

        try:
            results = runner._parse_results(f.name)
            assert abs(results["val_bpb"] - 1.294526) < 1e-7
            assert abs(results["training_seconds"] - 300.5) < 1e-2
            assert abs(results["peak_vram_mb"] - 27528.9) < 1e-2
        finally:
            os.unlink(f.name)

    def test_parse_results_missing_file(self):
        """Should return inf for missing log file."""
        runner = ExperimentRunner(PBARConfig())
        results = runner._parse_results("/nonexistent/path.log")
        assert results["val_bpb"] == float("inf")

    def test_parse_results_crash_log(self):
        """Should return inf for crash (no val_bpb line)."""
        runner = ExperimentRunner(PBARConfig())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("Traceback (most recent call last):\n")
            f.write("  File 'train.py', line 42\n")
            f.write("MemoryError: out of memory\n")

        try:
            results = runner._parse_results(f.name)
            assert results["val_bpb"] == float("inf")
        finally:
            os.unlink(f.name)


class TestCandidate:
    """Tests for the Candidate dataclass."""

    def test_creation(self):
        c = Candidate(
            branch_id=0,
            score=1.5,
            parent_score=2.0,
            commit_hash="abc1234",
            description="test change",
        )
        assert c.branch_id == 0
        assert c.score == 1.5
        assert c.diff == ""  # default


class TestPBAROrchestrator:
    """Tests for the main orchestrator."""

    def test_initialization(self, git_repo):
        """Should create orchestrator with valid config."""
        config = PBARConfig(n_branches=2, total_generations=5)
        orch = PBAROrchestrator(git_repo, config=config)

        assert orch.config.n_branches == 2
        assert orch.global_best_score == float("inf")
        assert orch.current_generation == 0

    def test_status_before_init(self, git_repo):
        """Status should work even before initialization."""
        config = PBARConfig(n_branches=2)
        orch = PBAROrchestrator(git_repo, config=config)

        status = orch.status()
        assert status["generation"] == 0
        assert status["global_best"] == float("inf")

    @patch.object(ExperimentRunner, "run_experiment")
    def test_initialize_runs_baselines(self, mock_run, git_repo):
        """Initialization should run baseline experiments on each branch."""
        mock_run.return_value = {
            "val_bpb": 2.667,
            "peak_vram_mb": 27000,
            "total_seconds": 340,
        }

        config = PBARConfig(n_branches=2)
        orch = PBAROrchestrator(git_repo, config=config)
        orch.initialize()

        assert mock_run.call_count == 2
        assert orch.global_best_score == 2.667

    @patch.object(ExperimentRunner, "run_experiment")
    def test_run_generation_with_propose_fn(self, mock_run, git_repo):
        """Should run a generation using the propose function."""
        mock_run.return_value = {
            "val_bpb": 2.5,
            "peak_vram_mb": 27000,
            "total_seconds": 340,
        }

        # Track propose calls
        propose_calls = []

        def mock_propose(worktree_path, branch_state):
            # Modify train.py to create something to commit
            train_py = os.path.join(worktree_path, "train.py")
            with open(train_py, "a") as f:
                f.write(f"\n# experiment {len(propose_calls)}\n")
            propose_calls.append(branch_state.branch_id)
            return f"experiment {len(propose_calls)}"

        config = PBARConfig(
            n_branches=2,
            experiments_per_branch=1,
            total_generations=1,
            rng_seed=42,
        )
        orch = PBAROrchestrator(git_repo, config=config, propose_fn=mock_propose)

        # Initialize (with baseline score)
        mock_run.return_value = {
            "val_bpb": 3.0,
            "peak_vram_mb": 27000,
            "total_seconds": 340,
        }
        orch.initialize()

        # Run generation with better score
        mock_run.return_value = {
            "val_bpb": 2.5,
            "peak_vram_mb": 27000,
            "total_seconds": 340,
        }
        best = orch.run_generation()

        assert len(propose_calls) == 2  # 2 branches × 1 experiment
        assert orch.current_generation == 1

    def test_cleanup(self, git_repo):
        """Cleanup should remove all worktrees."""
        config = PBARConfig(n_branches=2)
        orch = PBAROrchestrator(git_repo, config=config)

        # Initialize creates worktrees (mock the experiment running)
        with patch.object(ExperimentRunner, "run_experiment") as mock_run:
            mock_run.return_value = {"val_bpb": 2.0, "total_seconds": 10}
            orch.initialize()

        # Verify worktrees exist
        branches = orch.branch_manager.list_branches()
        paths = [b.worktree_path for b in branches]
        for p in paths:
            assert os.path.exists(p)

        # Cleanup
        orch.cleanup()
        for p in paths:
            assert not os.path.exists(p)
