"""SQLite results database for PBAR.

Tracks experiments, generations, and branch performance over time.
Thread-safe via SQLite's built-in locking.

Agent: Architect
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ExperimentRecord:
    """A single experiment result.

    Attributes:
        id: Auto-incremented row ID (None before insertion).
        generation: Generation number.
        branch_id: Branch this experiment ran on.
        score: Evaluation metric (val_bpb, lower is better).
        commit_hash: Git commit hash for this experiment.
        parent_score: Score before this experiment.
        status: 'keep', 'discard', or 'crash'.
        description: What was tried.
        duration_seconds: Wall-clock time for the experiment.
        memory_gb: Peak memory usage in GB.
        timestamp: Unix timestamp of completion.
        metadata: JSON-serializable extra data.
    """

    generation: int
    branch_id: int
    score: float
    commit_hash: str
    parent_score: float
    status: str
    description: str
    duration_seconds: float = 0.0
    memory_gb: float = 0.0
    timestamp: float = 0.0
    metadata: Optional[Dict] = None
    id: Optional[int] = None


class ResultsDB:
    """SQLite database for PBAR experiment tracking.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL,
                    branch_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    commit_hash TEXT NOT NULL,
                    parent_score REAL NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('keep', 'discard', 'crash')),
                    description TEXT NOT NULL,
                    duration_seconds REAL DEFAULT 0.0,
                    memory_gb REAL DEFAULT 0.0,
                    timestamp REAL DEFAULT 0.0,
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER NOT NULL UNIQUE,
                    temperature REAL NOT NULL,
                    best_score REAL NOT NULL,
                    mean_score REAL NOT NULL,
                    branch_scores TEXT NOT NULL,
                    timestamp REAL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_experiments_gen
                    ON experiments(generation);
                CREATE INDEX IF NOT EXISTS idx_experiments_branch
                    ON experiments(branch_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_status
                    ON experiments(status);
                """
            )

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_experiment(self, record: ExperimentRecord) -> int:
        """Insert an experiment record.

        Returns:
            Row ID of the inserted record.
        """
        if record.timestamp == 0.0:
            record.timestamp = time.time()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiments
                    (generation, branch_id, score, commit_hash, parent_score,
                     status, description, duration_seconds, memory_gb,
                     timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.generation,
                    record.branch_id,
                    record.score,
                    record.commit_hash,
                    record.parent_score,
                    record.status,
                    record.description,
                    record.duration_seconds,
                    record.memory_gb,
                    record.timestamp,
                    json.dumps(record.metadata or {}),
                ),
            )
            record.id = cursor.lastrowid
            return cursor.lastrowid

    def record_generation(
        self,
        generation: int,
        temperature: float,
        best_score: float,
        mean_score: float,
        branch_scores: Dict[int, float],
    ) -> None:
        """Record generation-level summary."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO generations
                    (generation, temperature, best_score, mean_score,
                     branch_scores, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    generation,
                    temperature,
                    best_score,
                    mean_score,
                    json.dumps(branch_scores),
                    time.time(),
                ),
            )

    def get_best_score(self) -> Optional[float]:
        """Get the best (lowest) score across all experiments."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(score) FROM experiments WHERE status = 'keep'"
            ).fetchone()
            return row[0] if row and row[0] is not None else None

    def get_branch_history(
        self,
        branch_id: int,
        limit: int = 50,
    ) -> List[ExperimentRecord]:
        """Get experiment history for a specific branch."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, generation, branch_id, score, commit_hash,
                       parent_score, status, description, duration_seconds,
                       memory_gb, timestamp, metadata
                FROM experiments
                WHERE branch_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (branch_id, limit),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_generation_summary(
        self,
        generation: int,
    ) -> Optional[Dict]:
        """Get summary for a specific generation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM generations WHERE generation = ?",
                (generation,),
            ).fetchone()
            if row is None:
                return None
            return {
                "generation": row[1],
                "temperature": row[2],
                "best_score": row[3],
                "mean_score": row[4],
                "branch_scores": json.loads(row[5]),
                "timestamp": row[6],
            }

    def get_experiment_count(self) -> int:
        """Total number of experiments recorded."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
            return row[0]

    def get_all_generations(self) -> List[Dict]:
        """Get all generation summaries, ordered by generation number."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM generations ORDER BY generation"
            ).fetchall()
            return [
                {
                    "generation": row[1],
                    "temperature": row[2],
                    "best_score": row[3],
                    "mean_score": row[4],
                    "branch_scores": json.loads(row[5]),
                    "timestamp": row[6],
                }
                for row in rows
            ]

    @staticmethod
    def _row_to_record(row: tuple) -> ExperimentRecord:
        return ExperimentRecord(
            id=row[0],
            generation=row[1],
            branch_id=row[2],
            score=row[3],
            commit_hash=row[4],
            parent_score=row[5],
            status=row[6],
            description=row[7],
            duration_seconds=row[8],
            memory_gb=row[9],
            timestamp=row[10],
            metadata=json.loads(row[11]) if row[11] else None,
        )
