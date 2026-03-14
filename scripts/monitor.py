#!/usr/bin/env python3
"""Live PBAR monitoring — watches the results database.

Agent: Jarvis

Usage:
    python scripts/monitor.py [--db results.db] [--watch]
"""

import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path


def get_status(db_path: str) -> dict:
    """Pull current status from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Latest generation
    gen = conn.execute("""
        SELECT * FROM generations ORDER BY generation DESC LIMIT 1
    """).fetchone()
    
    # Branch scores
    branches = conn.execute("""
        SELECT branch_id, 
               MIN(score) as best_score,
               COUNT(*) as experiments,
               MAX(created_at) as last_activity
        FROM experiments 
        GROUP BY branch_id
        ORDER BY best_score
    """).fetchall()
    
    # Recent experiments
    recent = conn.execute("""
        SELECT branch_id, score, description, status, created_at
        FROM experiments 
        ORDER BY created_at DESC 
        LIMIT 5
    """).fetchall()
    
    # Overall stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total_experiments,
            MIN(score) as global_best,
            AVG(score) as mean_score
        FROM experiments WHERE status = 'success'
    """).fetchone()
    
    conn.close()
    return {
        'generation': dict(gen) if gen else None,
        'branches': [dict(b) for b in branches],
        'recent': [dict(r) for r in recent],
        'stats': dict(stats) if stats else None
    }


def print_status(status: dict, clear: bool = True):
    """Pretty-print the status."""
    if clear:
        print("\033[2J\033[H", end="")  # Clear screen
    
    print("=" * 60)
    print("  PBAR MONITOR  ".center(60))
    print("=" * 60)
    print()
    
    # Generation info
    gen = status['generation']
    if gen:
        print(f"Generation: {gen['generation']}")
        print(f"Temperature: {gen['temperature']:.3f}")
        print(f"Best Score: {gen['best_score']:.6f}" if gen['best_score'] else "Best Score: --")
        print()
    else:
        print("No generations recorded yet.\n")
    
    # Branch leaderboard
    print("BRANCHES (by best score):")
    print("-" * 40)
    for b in status['branches']:
        score_str = f"{b['best_score']:.6f}" if b['best_score'] != float('inf') else "∞"
        print(f"  Branch {b['branch_id']}: {score_str}  ({b['experiments']} experiments)")
    print()
    
    # Global stats
    stats = status['stats']
    if stats and stats['total_experiments']:
        print(f"Total Experiments: {stats['total_experiments']}")
        print(f"Global Best: {stats['global_best']:.6f}")
        print(f"Mean Score: {stats['mean_score']:.6f}")
        print()
    
    # Recent activity
    print("RECENT EXPERIMENTS:")
    print("-" * 40)
    for r in status['recent']:
        ts = datetime.fromtimestamp(r['created_at']).strftime("%H:%M:%S")
        desc = (r['description'][:30] + "...") if r['description'] and len(r['description']) > 30 else (r['description'] or "")
        print(f"  [{ts}] B{r['branch_id']}: {r['score']:.4f} — {desc}")
    
    print()
    print(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Monitor PBAR runs")
    parser.add_argument("--db", default="results.db", help="Path to results database")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode (refresh every 2s)")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Start a PBAR run first, or specify --db path")
        return 1
    
    if args.watch:
        try:
            while True:
                status = get_status(str(db_path))
                print_status(status)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        status = get_status(str(db_path))
        print_status(status, clear=False)
    
    return 0


if __name__ == "__main__":
    exit(main())
