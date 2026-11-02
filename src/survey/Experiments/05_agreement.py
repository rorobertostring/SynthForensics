#!/usr/bin/env python3
"""
05_agreement.py — Inter-rater agreement (Krippendorff's α).

For each question (Q1, Q2, Q3), the reliability data matrix has
    rows    = participants (kept sessions, 118)
    columns = unique pairs (110)
    values  = chosen_video for that participant on that pair (or NaN if not seen)

α is computed at the nominal level. Missing values are handled natively by
the krippendorff library.

Subgroups also reported:
    - vs_type ∈ {competitor, real}
    - ai_background ∈ {True, False}
    - knows_deepfakes ∈ {yes, heard_term, no}

The script also reports the chi-square test of the uniform-choice null on the
Q2 responses (A, B, Equal) of the SF-vs-real pairs, used in Appendix C.4.4 to
show that the vs-real α collapse is not uniformly random responding.

Run:
    python 05_agreement.py

Output:
    agreement.json
"""

import json
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import chisquare

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
OUTPUT_JSON = HERE / "agreement.json"

QUESTIONS = ["better_made", "more_realistic", "could_be_fake"]
QUESTION_LABEL = {"better_made": "Q1", "more_realistic": "Q2", "could_be_fake": "Q3"}


def encode_categorical(value: object) -> int | float:
    """Map chosen_video string to an integer for the reliability matrix.
    NaN (= 'Neither' for Q3, or unanswered) → np.nan.
    """
    mapping = {"A": 0, "B": 1, "Equal": 2, "Both": 3}
    if pd.isna(value):
        return np.nan
    if value not in mapping:
        return np.nan
    return mapping[value]


def build_reliability_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build a (n_raters × n_items) reliability matrix from a long-format df.

    Items are identified by (video_a_path, video_b_path) — a unique pair key.
    """
    df = df.copy()
    df["pair_key"] = df["video_a_path"] + "||" + df["video_b_path"]
    df["code"] = df["chosen_video"].apply(encode_categorical)
    pivot = df.pivot_table(
        index="session_id",
        columns="pair_key",
        values="code",
        aggfunc="first",
    )
    return pivot.to_numpy(dtype=float)


def alpha_for(df: pd.DataFrame, question_key: str) -> float | None:
    """Compute α for one question; return None if data are too sparse."""
    sub = df[df["question_key"] == question_key]
    if len(sub) == 0:
        return None
    matrix = build_reliability_matrix(sub)
    # Need at least 2 raters and 2 items with at least 2 ratings each.
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    try:
        a = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
        return float(a)
    except Exception as e:
        print(f"  α computation failed: {e}")
        return None


def report_block(df: pd.DataFrame, label: str, n_sessions: int) -> dict:
    block = {"_n_sessions": int(n_sessions)}
    for q in QUESTIONS:
        a = alpha_for(df, q)
        block[QUESTION_LABEL[q]] = a
    print(f"  {label:30s}  Q1={fmt(block['Q1'])}  Q2={fmt(block['Q2'])}  Q3={fmt(block['Q3'])}  (n_sessions={n_sessions})")
    return block


def fmt(a: float | None) -> str:
    return f"{a:.3f}" if a is not None else "  N/A"


def indecision_chisquare(df: pd.DataFrame) -> dict:
    """Chi-square test of the uniform-choice null on Q2 for SF-vs-real pairs.

    Counts A (SF), B (real) and Equal responses on the real-video stratum,
    excluding unanswered items (NaN), and tests them against a uniform split.
    """
    sub = df[(df["vs_type"] == "real") & (df["question_key"] == "more_realistic")]
    counts = sub["chosen_video"].value_counts().reindex(["A", "B", "Equal"], fill_value=0)
    res = chisquare(counts.to_numpy())
    block = {
        "n": int(counts.sum()),
        "counts": {k: int(v) for k, v in counts.items()},
        "chi2": float(res.statistic),
        "df": len(counts) - 1,
        "p": float(res.pvalue),
    }
    print(f"  Q2 vs_real  A/B/Equal={list(block['counts'].values())}  n={block['n']}  "
          f"chi2={block['chi2']:.3f}  df={block['df']}  p={block['p']:.2e}")
    return block


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    df = df[df["session_id"].isin(set(kept["session_id"]))].copy()

    real = {"FF++", "DFD"}
    df["vs_type"] = df["benchmark_name"].apply(
        lambda b: "real" if b in real else "competitor"
    )

    n_total = df["session_id"].nunique()
    print(f"Computing Krippendorff α (nominal) on {n_total} kept sessions, "
          f"{df['video_a_path'].apply(lambda p: p).nunique()} SF videos seen.")
    print()

    out: dict = {}

    # Overall
    print("=== OVERALL ===")
    out["overall"] = report_block(df, "all", n_total)

    # By vs_type
    print()
    print("=== BY vs_type ===")
    out["by_vs_type"] = {}
    for vt in ["competitor", "real"]:
        sub = df[df["vs_type"] == vt]
        out["by_vs_type"][vt] = report_block(sub, f"vs_{vt}", sub["session_id"].nunique())

    # By ai_background
    print()
    print("=== BY ai_background ===")
    out["by_ai_background"] = {}
    for bg in [True, False]:
        sub = df[df["ai_background"] == bg]
        key = "ai_yes" if bg else "ai_no"
        out["by_ai_background"][key] = report_block(sub, key, sub["session_id"].nunique())

    # By knows_deepfakes
    print()
    print("=== BY knows_deepfakes ===")
    out["by_knows_deepfakes"] = {}
    for k in ["yes", "heard_term", "no"]:
        sub = df[df["knows_deepfakes"] == k]
        out["by_knows_deepfakes"][k] = report_block(sub, k, sub["session_id"].nunique())

    # By age_range
    print()
    print("=== BY age_range ===")
    out["by_age_range"] = {}
    for age in ["18-25", "26-35", "36-50", "50+"]:
        sub = df[df["age_range"] == age]
        out["by_age_range"][age] = report_block(sub, age, sub["session_id"].nunique())

    # Q2 vs real: indecision vs uniform random responding (Appendix C.4.4)
    print()
    print("=== Q2 vs_real: chi-square vs uniform choice ===")
    out["vs_real_q2_uniform_chisquare"] = indecision_chisquare(df)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved {OUTPUT_JSON.name}")


if __name__ == "__main__":
    main()
