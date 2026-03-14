"""Git worktree-based branch management for PBAR.

Manages parallel experiment branches as git worktrees, providing
isolated working directories for each population member.

Agent: Architect
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Serialize git operations to prevent corruption with concurrent worktree access
_git_lock = threading.Lock()


@dataclass
class BranchState:
    """State of a single population branch.

    Attributes:
        branch_id: Unique identifier (e.g., 0, 1, 2, 3).
        branch_name: Git branch name (e.g., 'pbar/branch-0').
        worktree_path: Absolute path to the worktree directory.
        current_score: Best score achieved on this branch (lower is better).
        generation: Current generation number for this branch.
        commit_hash: Current HEAD commit hash.
        experiment_count: Total experiments run on this branch.
    """

    branch_id: int
    branch_name: str
    worktree_path: str
    current_score: float = float("inf")
    generation: int = 0
    commit_hash: str = ""
    experiment_count: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_initialized(self) -> bool:
        return self.commit_hash != "" and self.current_score < float("inf")


def _run_git(
    args: List[str],
    cwd: Optional[str] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a git command with the global lock held.

    All git operations are serialized to prevent corruption when
    multiple branches share the same .git directory.
    """
    cmd = ["git"] + args
    with _git_lock:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result


class GitError(Exception):
    """Raised when a git operation fails."""


class BranchManager:
    """Manages git worktrees for parallel PBAR branches.

    Each branch gets its own worktree directory so experiments can run
    in complete isolation. The main repo's .git directory is shared.

    Args:
        repo_path: Path to the main git repository.
        n_branches: Number of parallel branches to manage.
        branch_prefix: Prefix for branch names (default: 'pbar/branch').
        worktree_base: Base directory for worktrees. Defaults to
                       sibling directories of repo_path.
    """

    def __init__(
        self,
        repo_path: str,
        n_branches: int = 4,
        branch_prefix: str = "pbar/branch",
        worktree_base: Optional[str] = None,
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.n_branches = n_branches
        self.branch_prefix = branch_prefix

        if worktree_base is None:
            parent = os.path.dirname(self.repo_path)
            repo_name = os.path.basename(self.repo_path)
            self.worktree_base = os.path.join(parent, f"{repo_name}-branches")
        else:
            self.worktree_base = os.path.abspath(worktree_base)

        self.branches: Dict[int, BranchState] = {}

    def _branch_name(self, branch_id: int) -> str:
        return f"{self.branch_prefix}-{branch_id}"

    def _worktree_path(self, branch_id: int) -> str:
        return os.path.join(self.worktree_base, f"branch-{branch_id}")

    def _get_head(self, cwd: str) -> str:
        """Get the HEAD commit hash for a given working directory."""
        result = _run_git(["rev-parse", "HEAD"], cwd=cwd)
        return result.stdout.strip()

    def initialize(self, source_branch: str = "HEAD") -> List[BranchState]:
        """Create worktrees for all branches from a source point.

        Args:
            source_branch: Git ref to branch from (default: HEAD).

        Returns:
            List of initialized BranchState objects.
        """
        os.makedirs(self.worktree_base, exist_ok=True)
        source_hash = _run_git(
            ["rev-parse", source_branch], cwd=self.repo_path
        ).stdout.strip()

        for branch_id in range(self.n_branches):
            branch_name = self._branch_name(branch_id)
            worktree_path = self._worktree_path(branch_id)

            # Clean up existing worktree if present
            if os.path.exists(worktree_path):
                self._remove_worktree(branch_id)

            # Remove existing branch if present (force)
            _run_git(
                ["branch", "-D", branch_name],
                cwd=self.repo_path,
                check=False,
            )

            # Create worktree with new branch from source
            _run_git(
                ["worktree", "add", worktree_path, "-b", branch_name, source_hash],
                cwd=self.repo_path,
            )

            commit_hash = self._get_head(worktree_path)
            state = BranchState(
                branch_id=branch_id,
                branch_name=branch_name,
                worktree_path=worktree_path,
                commit_hash=commit_hash,
            )
            self.branches[branch_id] = state
            logger.info(
                "Created branch %s at %s (commit %s)",
                branch_name,
                worktree_path,
                commit_hash[:7],
            )

        return list(self.branches.values())

    def _remove_worktree(self, branch_id: int) -> None:
        """Remove a worktree and clean up."""
        worktree_path = self._worktree_path(branch_id)
        if os.path.exists(worktree_path):
            _run_git(
                ["worktree", "remove", worktree_path, "--force"],
                cwd=self.repo_path,
                check=False,
            )
        # If git worktree remove didn't clean up, force remove
        if os.path.exists(worktree_path):
            shutil.rmtree(worktree_path, ignore_errors=True)

    def commit_change(
        self,
        branch_id: int,
        files: List[str],
        message: str,
    ) -> str:
        """Stage files and commit on a branch.

        Args:
            branch_id: Branch to commit on.
            files: List of file paths (relative to worktree) to stage.
            message: Commit message.

        Returns:
            New commit hash.
        """
        state = self.branches[branch_id]
        cwd = state.worktree_path

        for f in files:
            _run_git(["add", f], cwd=cwd)

        _run_git(["commit", "-m", message], cwd=cwd)
        new_hash = self._get_head(cwd)
        state.commit_hash = new_hash
        return new_hash

    def revert_to(self, branch_id: int, commit_hash: str) -> None:
        """Hard-reset a branch to a specific commit.

        Args:
            branch_id: Branch to reset.
            commit_hash: Target commit to reset to.
        """
        state = self.branches[branch_id]
        _run_git(["reset", "--hard", commit_hash], cwd=state.worktree_path)
        state.commit_hash = commit_hash

    def reset_branch_to_source(
        self,
        target_branch_id: int,
        source_branch_id: int,
    ) -> None:
        """Reset a target branch to match a source branch's state.

        Used for branch pruning: kill worst branch, clone best branch.

        Args:
            target_branch_id: Branch to reset.
            source_branch_id: Branch to copy from.
        """
        source = self.branches[source_branch_id]
        target = self.branches[target_branch_id]

        _run_git(
            ["reset", "--hard", source.commit_hash],
            cwd=target.worktree_path,
        )

        target.commit_hash = source.commit_hash
        target.current_score = source.current_score
        logger.info(
            "Reset branch %d to branch %d (commit %s, score %.6f)",
            target_branch_id,
            source_branch_id,
            source.commit_hash[:7],
            source.current_score,
        )

    def merge_branch(
        self,
        source_branch_id: int,
        target_branch_id: int,
        strategy: str = "theirs",
    ) -> bool:
        """Attempt to merge source branch into target branch.

        Args:
            source_branch_id: Branch to merge from.
            target_branch_id: Branch to merge into.
            strategy: Merge strategy ('theirs' or 'ours').

        Returns:
            True if merge succeeded, False if conflicts arose and was aborted.
        """
        source = self.branches[source_branch_id]
        target = self.branches[target_branch_id]

        strategy_option = (
            f"-X{strategy}" if strategy in ("theirs", "ours") else ""
        )

        args = ["merge", source.branch_name, "--no-edit"]
        if strategy_option:
            args.append(strategy_option)

        result = _run_git(args, cwd=target.worktree_path, check=False)

        if result.returncode != 0:
            # Abort the failed merge
            _run_git(["merge", "--abort"], cwd=target.worktree_path, check=False)
            logger.warning(
                "Merge of branch %d into %d failed, aborted",
                source_branch_id,
                target_branch_id,
            )
            return False

        target.commit_hash = self._get_head(target.worktree_path)
        logger.info(
            "Merged branch %d into %d (new commit %s)",
            source_branch_id,
            target_branch_id,
            target.commit_hash[:7],
        )
        return True

    def cleanup(self) -> None:
        """Remove all worktrees and branches created by this manager."""
        for branch_id in list(self.branches.keys()):
            branch_name = self._branch_name(branch_id)
            self._remove_worktree(branch_id)
            _run_git(
                ["branch", "-D", branch_name],
                cwd=self.repo_path,
                check=False,
            )
        self.branches.clear()

        # Remove the worktree base directory if empty
        if os.path.exists(self.worktree_base):
            try:
                os.rmdir(self.worktree_base)
            except OSError:
                pass  # Not empty, leave it

    def list_branches(self) -> List[BranchState]:
        """Return all branch states sorted by score (best first)."""
        return sorted(self.branches.values(), key=lambda b: b.current_score)

    def get_branch(self, branch_id: int) -> BranchState:
        """Get state for a specific branch."""
        if branch_id not in self.branches:
            raise KeyError(f"Branch {branch_id} not found")
        return self.branches[branch_id]
