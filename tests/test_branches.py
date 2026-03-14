"""Tests for PBAR git worktree branch management.

Agent: Architect
"""

import os
import subprocess
import tempfile

import pytest

from pbar.branches import BranchManager, BranchState, GitError, _run_git


@pytest.fixture
def git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "test-repo")
        os.makedirs(repo_path)

        # Initialize git repo
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

        # Create initial commit with a train.py
        train_py = os.path.join(repo_path, "train.py")
        with open(train_py, "w") as f:
            f.write("# baseline training script\nprint('hello')\n")

        subprocess.run(
            ["git", "add", "train.py"], cwd=repo_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield repo_path


class TestBranchManager:
    """Tests for the BranchManager class."""

    def test_initialize_creates_worktrees(self, git_repo):
        """Should create the specified number of worktrees."""
        mgr = BranchManager(git_repo, n_branches=3)
        branches = mgr.initialize()

        assert len(branches) == 3
        for branch in branches:
            assert os.path.exists(branch.worktree_path)
            assert os.path.isdir(branch.worktree_path)
            assert os.path.exists(os.path.join(branch.worktree_path, "train.py"))

    def test_initialize_branch_names(self, git_repo):
        """Branch names should follow the naming convention."""
        mgr = BranchManager(git_repo, n_branches=2, branch_prefix="test/branch")
        branches = mgr.initialize()

        assert branches[0].branch_name == "test/branch-0"
        assert branches[1].branch_name == "test/branch-1"

    def test_commit_change(self, git_repo):
        """Should commit changes in a worktree."""
        mgr = BranchManager(git_repo, n_branches=1)
        branches = mgr.initialize()
        branch = branches[0]

        # Modify train.py
        train_py = os.path.join(branch.worktree_path, "train.py")
        with open(train_py, "w") as f:
            f.write("# modified\nprint('modified')\n")

        old_hash = branch.commit_hash
        new_hash = mgr.commit_change(
            branch.branch_id,
            files=["train.py"],
            message="test: modify train.py",
        )

        assert new_hash != old_hash
        assert branch.commit_hash == new_hash

    def test_revert_to(self, git_repo):
        """Should hard-reset a branch to a specific commit."""
        mgr = BranchManager(git_repo, n_branches=1)
        branches = mgr.initialize()
        branch = branches[0]
        original_hash = branch.commit_hash

        # Make a change
        train_py = os.path.join(branch.worktree_path, "train.py")
        with open(train_py, "w") as f:
            f.write("# changed\n")

        mgr.commit_change(branch.branch_id, files=["train.py"], message="change")
        assert branch.commit_hash != original_hash

        # Revert
        mgr.revert_to(branch.branch_id, original_hash)
        assert branch.commit_hash == original_hash

        # Verify file contents reverted
        with open(train_py) as f:
            content = f.read()
        assert "baseline" in content

    def test_reset_branch_to_source(self, git_repo):
        """Should reset one branch to match another."""
        mgr = BranchManager(git_repo, n_branches=2)
        branches = mgr.initialize()

        # Modify branch 0
        train_py_0 = os.path.join(branches[0].worktree_path, "train.py")
        with open(train_py_0, "w") as f:
            f.write("# branch 0 best\n")
        mgr.commit_change(0, files=["train.py"], message="best change")
        branches[0].current_score = 1.0

        # Branch 1 has worse score
        branches[1].current_score = 2.0

        # Reset branch 1 to branch 0
        mgr.reset_branch_to_source(target_branch_id=1, source_branch_id=0)

        # Verify branch 1 now has branch 0's state
        train_py_1 = os.path.join(branches[1].worktree_path, "train.py")
        with open(train_py_1) as f:
            content = f.read()
        assert "branch 0 best" in content
        assert branches[1].current_score == 1.0

    def test_cleanup_removes_worktrees(self, git_repo):
        """Cleanup should remove all worktrees and branches."""
        mgr = BranchManager(git_repo, n_branches=2)
        branches = mgr.initialize()

        worktree_paths = [b.worktree_path for b in branches]
        for path in worktree_paths:
            assert os.path.exists(path)

        mgr.cleanup()

        for path in worktree_paths:
            assert not os.path.exists(path)
        assert len(mgr.branches) == 0

    def test_list_branches_sorted_by_score(self, git_repo):
        """list_branches should return branches sorted by score (best first)."""
        mgr = BranchManager(git_repo, n_branches=3)
        branches = mgr.initialize()

        branches[0].current_score = 1.5
        branches[1].current_score = 1.2
        branches[2].current_score = 1.8

        sorted_branches = mgr.list_branches()
        scores = [b.current_score for b in sorted_branches]
        assert scores == [1.2, 1.5, 1.8]

    def test_get_branch(self, git_repo):
        """Should retrieve a specific branch by ID."""
        mgr = BranchManager(git_repo, n_branches=2)
        mgr.initialize()

        branch = mgr.get_branch(1)
        assert branch.branch_id == 1

    def test_get_branch_invalid_id(self, git_repo):
        """Should raise KeyError for invalid branch ID."""
        mgr = BranchManager(git_repo, n_branches=2)
        mgr.initialize()

        with pytest.raises(KeyError):
            mgr.get_branch(99)

    def test_reinitialize_cleans_up(self, git_repo):
        """Reinitializing should clean up existing worktrees first."""
        mgr = BranchManager(git_repo, n_branches=2)
        mgr.initialize()
        old_paths = [b.worktree_path for b in mgr.list_branches()]

        # Reinitialize
        mgr.initialize()
        new_paths = [b.worktree_path for b in mgr.list_branches()]

        # Paths should be the same (same naming convention)
        assert old_paths == new_paths
        # All should exist
        for path in new_paths:
            assert os.path.exists(path)


class TestBranchState:
    """Tests for the BranchState dataclass."""

    def test_is_initialized_false_by_default(self):
        """New branch state should not be initialized."""
        state = BranchState(
            branch_id=0,
            branch_name="test",
            worktree_path="/tmp/test",
        )
        assert not state.is_initialized

    def test_is_initialized_true_with_data(self):
        """Branch state should be initialized with commit hash and score."""
        state = BranchState(
            branch_id=0,
            branch_name="test",
            worktree_path="/tmp/test",
            current_score=1.5,
            commit_hash="abc1234",
        )
        assert state.is_initialized


class TestGitError:
    """Tests for git error handling."""

    def test_invalid_git_command(self):
        """Invalid git commands should raise GitError."""
        with pytest.raises(GitError):
            _run_git(["not-a-real-command"])
