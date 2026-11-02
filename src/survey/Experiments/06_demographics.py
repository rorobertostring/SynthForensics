#!/usr/bin/env python3
"""
06_demographics.py — Demographic breakdown by screening segment.

Produces a long-format CSV with counts and percentages per demographic
dimension, broken down by:
    - initial               (all 159 participants)
    - kept                  (post both screening stages, 118)
    - rejected_completion   (failed Stage 1, 39)
    - rejected_variance     (failed Stage 2, 2)

Dimensions covered: age_range, ai_background, knows_deepfakes, language.

Run:
    python 06_demographics.py

Outputs:
    demographics.csv
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
LOG_CSV = HERE / "screening_log.csv"
OUTPUT_CSV = HERE / "demographics.csv"

DIMENSIONS = [
    ("age_range", ["18-25", "26-35", "36-50", "50+"]),
    ("ai_background", [True, False]),
    ("knows_deepfakes", ["yes", "heard_term", "no"]),
    ("language", ["it", "en"]),
]


def session_demographics(df: pd.DataFrame, session_ids: set) -> pd.DataFrame:
    """Return a one-row-per-session dataframe with demographic columns."""
    return (
        df[df["session_id"].isin(session_ids)]
        .drop_duplicates("session_id")
        [["session_id", "age_range", "ai_background", "knows_deepfakes", "language"]]
    )


def counts_for_segment(seg_name: str, demo: pd.DataFrame) -> list[dict]:
    """Build long-format rows (segment, dimension, value, count, pct) for one segment."""
    n_total = len(demo)
    rows = []
    for dim, ordered_values in DIMENSIONS:
        vc = demo[dim].value_counts(dropna=False)
        # Iterate in canonical order, then any unexpected values that appeared
        seen = set()
        for v in ordered_values:
            seen.add(v)
            count = int(vc.get(v, 0))
            rows.append({
                "segment": seg_name,
                "n_total": n_total,
                "dimension": dim,
                "value": str(v),
                "count": count,
                "pct": (count / n_total) if n_total else 0.0,
            })
        # Fallback for any unexpected values (won't normally happen but log them)
        for v, count in vc.items():
            if v in seen or pd.isna(v):
                continue
            rows.append({
                "segment": seg_name,
                "n_total": n_total,
                "dimension": dim,
                "value": f"UNEXPECTED:{v}",
                "count": int(count),
                "pct": int(count) / n_total if n_total else 0.0,
            })
    return rows


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    log = pd.read_csv(LOG_CSV)

    initial_ids = set(df["session_id"].unique())
    kept_ids = set(kept["session_id"])
    rej_completion = set(log.loc[log["reason"] == "completion", "session_id"])
    rej_variance = set(log.loc[log["reason"] == "variance", "session_id"])

    # Sanity arithmetic
    assert kept_ids.isdisjoint(rej_completion), "kept ∩ rej_completion not empty"
    assert kept_ids.isdisjoint(rej_variance), "kept ∩ rej_variance not empty"
    assert rej_completion.isdisjoint(rej_variance), "completion ∩ variance not empty"
    assert len(initial_ids) == len(kept_ids) + len(rej_completion) + len(rej_variance), (
        f"counts mismatch: {len(initial_ids)} vs "
        f"{len(kept_ids)} + {len(rej_completion)} + {len(rej_variance)}"
    )

    rows: list[dict] = []
    for seg_name, ids in [
        ("initial", initial_ids),
        ("kept", kept_ids),
        ("rejected_completion", rej_completion),
        ("rejected_variance", rej_variance),
    ]:
        demo = session_demographics(df, ids)
        rows.extend(counts_for_segment(seg_name, demo))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Initial            : {len(initial_ids):>3}")
    print(f"Kept               : {len(kept_ids):>3}  ({100*len(kept_ids)/len(initial_ids):.1f}%)")
    print(f"Rejected completion: {len(rej_completion):>3}  ({100*len(rej_completion)/len(initial_ids):.1f}%)")
    print(f"Rejected variance  : {len(rej_variance):>3}  ({100*len(rej_variance)/len(initial_ids):.1f}%)")
    print()

    # Print a wide-format summary for human reading
    pivot = out.pivot_table(
        index=["dimension", "value"],
        columns="segment",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    # Reorder columns
    cols = ["initial", "kept", "rejected_completion", "rejected_variance"]
    pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns])
    print(pivot.to_string())
    print()
    print(f"Saved {OUTPUT_CSV.name}: {len(out)} rows")


if __name__ == "__main__":
    main()
