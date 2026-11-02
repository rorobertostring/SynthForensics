#!/usr/bin/env python3
"""
02_screen_subjects.py — Two-stage participant screening.

Stage 1 — Completion filter:
    Drop sessions with completed_pairs / 35 < 0.80.

Stage 2 — Disengagement check:
    Drop sessions whose top-voted Q1+Q2 category accounts for ≥ 95% of
    responses (no behavioural variance ⇒ disengaged / non-discriminating responder).

Run:
    python 02_screen_subjects.py

Input:
    prepared.parquet   (output of 01_prepare_data.py)

Outputs:
    kept_sessions.csv      (sessions that pass both stages)
    screening_log.csv      (sessions rejected, with reason and metric)
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
LOG_CSV = HERE / "screening_log.csv"

PAIRS_ATTENDED = 35
COMPLETION_THRESHOLD = 0.80
MIN_PAIRS_KEPT = int(np.ceil(COMPLETION_THRESHOLD * PAIRS_ATTENDED))   # = 28
DISENGAGEMENT_THRESHOLD = 0.95         # drop if top-category share ≥ 95%
PREFERENCE_QUESTIONS = ["better_made", "more_realistic"]   # Q3 excluded


def stage1_completion(df: pd.DataFrame) -> tuple[set, pd.DataFrame]:
    """Return (kept_set, dropped_log_df) for the completion filter."""
    pairs_per_session = df.groupby("session_id")["pair_number"].nunique()
    keep = set(pairs_per_session[pairs_per_session >= MIN_PAIRS_KEPT].index)
    drop = set(pairs_per_session.index) - keep
    log = pd.DataFrame([
        {
            "session_id": sess,
            "reason": "completion",
            "n_pairs_completed": int(pairs_per_session[sess]),
            "top_category": None,
            "top_pct": None,
        }
        for sess in sorted(drop)
    ])
    print(f"Stage 1 (completion ≥ {MIN_PAIRS_KEPT}/{PAIRS_ATTENDED}): "
          f"{len(pairs_per_session):>3} → {len(keep):>3} kept, "
          f"{len(drop):>3} dropped.")
    return keep, log


def disengagement_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Per session, compute the top-voted Q1+Q2 category and its share."""
    sub = df[df["question_key"].isin(PREFERENCE_QUESTIONS)]
    rows = []
    for sess, grp in sub.groupby("session_id"):
        counts = grp["chosen_video"].value_counts(dropna=False)
        if len(counts) == 0:
            rows.append({"session_id": sess, "top_category": None, "top_pct": np.nan, "n_q12": 0})
            continue
        rows.append({
            "session_id": sess,
            "top_category": counts.idxmax(),
            "top_pct": float(counts.iloc[0] / counts.sum()),
            "n_q12": int(counts.sum()),
        })
    return pd.DataFrame(rows)


def stage2_disengagement(df: pd.DataFrame, alive: set) -> tuple[set, pd.DataFrame]:
    """Return (kept_set, dropped_log_df) for the disengagement filter."""
    metrics = disengagement_metric(df[df["session_id"].isin(alive)])
    drop_mask = metrics["top_pct"] >= DISENGAGEMENT_THRESHOLD
    drop = set(metrics.loc[drop_mask, "session_id"])
    keep = alive - drop

    pairs_per_session = df.groupby("session_id")["pair_number"].nunique()
    log = pd.DataFrame([
        {
            "session_id": row["session_id"],
            "reason": "variance",
            "n_pairs_completed": int(pairs_per_session[row["session_id"]]),
            "top_category": row["top_category"],
            "top_pct": float(row["top_pct"]),
        }
        for _, row in metrics.loc[drop_mask].iterrows()
    ])
    print(f"Stage 2 (top-category share < {DISENGAGEMENT_THRESHOLD:.0%}): "
          f"{len(alive):>3} → {len(keep):>3} kept, "
          f"{len(drop):>3} dropped.")
    return keep, log


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {INPUT_PARQUET.name}")
    print(f"Initial unique sessions: {df['session_id'].nunique()}")
    print()

    # Stage 1
    kept_after_stage1, log1 = stage1_completion(df)

    # Stage 2
    kept_after_stage2, log2 = stage2_disengagement(df, kept_after_stage1)

    # Build kept output
    metrics = disengagement_metric(df[df["session_id"].isin(kept_after_stage2)])
    pairs_per_session = df.groupby("session_id")["pair_number"].nunique()
    kept_df = pd.DataFrame([
        {
            "session_id": s,
            "n_pairs_completed": int(pairs_per_session[s]),
            "top_category": metrics.loc[metrics["session_id"] == s, "top_category"].iloc[0],
            "top_pct": float(metrics.loc[metrics["session_id"] == s, "top_pct"].iloc[0]),
        }
        for s in sorted(kept_after_stage2)
    ])
    kept_df.to_csv(KEPT_CSV, index=False)

    # Combine and save the rejection log (skip empty frames to avoid
    # pandas FutureWarning on concat of all-NA dataframes)
    logs = [lg for lg in (log1, log2) if not lg.empty]
    log_df = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    log_df.to_csv(LOG_CSV, index=False)

    print()
    print("=== SUMMARY ===")
    n0 = df["session_id"].nunique()
    n1 = len(kept_after_stage1)
    n2 = len(kept_after_stage2)
    print(f"Initial sessions:           {n0:>4}")
    print(f"After Stage 1 (completion): {n1:>4} (rejected {n0 - n1})")
    print(f"After Stage 2 (variance):   {n2:>4} (rejected {n1 - n2})")
    print(f"Final retention rate:       {100*n2/n0:.1f}%")
    print()
    print(f"Saved: {KEPT_CSV.name} ({len(kept_df)} rows)")
    print(f"Saved: {LOG_CSV.name} ({len(log_df)} rows)")


if __name__ == "__main__":
    main()
