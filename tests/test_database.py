"""Tests for PBAR results database.

Agent: Architect
"""

import os
import tempfile

import pytest

from pbar.database import ExperimentRecord, ResultsDB


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_results.db")
        yield ResultsDB(db_path)


class TestResultsDB:
    """Tests for the SQLite results database."""

    def test_create_database(self, db):
        """Database should be created successfully."""
        assert os.path.exists(db.db_path)

    def test_record_experiment(self, db):
        """Should insert and retrieve experiment records."""
        record = ExperimentRecord(
            generation=1,
            branch_id=0,
            score=1.234567,
            commit_hash="abc1234",
            parent_score=1.500000,
            status="keep",
            description="test experiment",
            duration_seconds=312.4,
            memory_gb=26.9,
        )
        row_id = db.record_experiment(record)
        assert row_id > 0
        assert record.id == row_id

    def test_get_best_score(self, db):
        """Should return the best (lowest) score."""
        for score, status in [(1.5, "keep"), (1.2, "keep"), (1.8, "discard")]:
            db.record_experiment(
                ExperimentRecord(
                    generation=1,
                    branch_id=0,
                    score=score,
                    commit_hash="abc",
                    parent_score=2.0,
                    status=status,
                    description="test",
                )
            )

        best = db.get_best_score()
        assert best == 1.2  # Only 'keep' records count

    def test_get_best_score_empty(self, db):
        """Should return None when no experiments exist."""
        assert db.get_best_score() is None

    def test_get_branch_history(self, db):
        """Should return experiments for a specific branch."""
        for branch_id in [0, 1]:
            for i in range(3):
                db.record_experiment(
                    ExperimentRecord(
                        generation=i,
                        branch_id=branch_id,
                        score=1.5 - i * 0.1,
                        commit_hash=f"hash_{branch_id}_{i}",
                        parent_score=2.0,
                        status="keep",
                        description=f"exp {i}",
                        timestamp=float(i),
                    )
                )

        history = db.get_branch_history(0, limit=10)
        assert len(history) == 3
        assert all(r.branch_id == 0 for r in history)

    def test_record_generation(self, db):
        """Should record and retrieve generation summaries."""
        db.record_generation(
            generation=5,
            temperature=1.5,
            best_score=1.2,
            mean_score=1.4,
            branch_scores={0: 1.2, 1: 1.4, 2: 1.5, 3: 1.6},
        )

        summary = db.get_generation_summary(5)
        assert summary is not None
        assert summary["generation"] == 5
        assert summary["temperature"] == 1.5
        assert summary["best_score"] == 1.2
        assert summary["mean_score"] == 1.4
        assert summary["branch_scores"] == {"0": 1.2, "1": 1.4, "2": 1.5, "3": 1.6}

    def test_get_generation_summary_missing(self, db):
        """Should return None for non-existent generation."""
        assert db.get_generation_summary(999) is None

    def test_get_experiment_count(self, db):
        """Should count all experiments."""
        assert db.get_experiment_count() == 0

        for i in range(5):
            db.record_experiment(
                ExperimentRecord(
                    generation=0,
                    branch_id=0,
                    score=1.0,
                    commit_hash="abc",
                    parent_score=2.0,
                    status="keep",
                    description="test",
                )
            )

        assert db.get_experiment_count() == 5

    def test_get_all_generations(self, db):
        """Should return all generations in order."""
        for gen in [3, 1, 5, 2]:
            db.record_generation(
                generation=gen,
                temperature=2.0 - gen * 0.1,
                best_score=1.5 - gen * 0.01,
                mean_score=1.6 - gen * 0.01,
                branch_scores={0: 1.5},
            )

        gens = db.get_all_generations()
        assert len(gens) == 4
        assert [g["generation"] for g in gens] == [1, 2, 3, 5]

    def test_invalid_status_raises(self, db):
        """Invalid status should be rejected by the database."""
        with pytest.raises(Exception):
            db.record_experiment(
                ExperimentRecord(
                    generation=0,
                    branch_id=0,
                    score=1.0,
                    commit_hash="abc",
                    parent_score=2.0,
                    status="invalid_status",
                    description="test",
                )
            )

    def test_metadata_roundtrip(self, db):
        """Metadata should survive JSON serialization roundtrip."""
        metadata = {"hyperparams": {"lr": 0.001}, "notes": "test run"}
        db.record_experiment(
            ExperimentRecord(
                generation=0,
                branch_id=0,
                score=1.0,
                commit_hash="abc",
                parent_score=2.0,
                status="keep",
                description="test",
                metadata=metadata,
            )
        )

        history = db.get_branch_history(0)
        assert len(history) == 1
        assert history[0].metadata == metadata

    def test_concurrent_writes(self, db):
        """Multiple rapid writes should not corrupt the database."""
        import threading

        errors = []

        def write_records(branch_id):
            try:
                for i in range(20):
                    db.record_experiment(
                        ExperimentRecord(
                            generation=i,
                            branch_id=branch_id,
                            score=1.0 + i * 0.01,
                            commit_hash=f"hash_{branch_id}_{i}",
                            parent_score=2.0,
                            status="keep",
                            description=f"concurrent test {branch_id}/{i}",
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_records, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"
        assert db.get_experiment_count() == 80
