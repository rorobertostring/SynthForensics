#!/usr/bin/env python3
"""
08_sensitivity_subgroup.py — Demographic-subgroup sensitivity for the headline
human-study numbers (Q1/Q2 SF-win-rate and Q3 SF-flagged-rate vs benchmarks).

For each demographic dimension (ai_background, knows_deepfakes, age_range),
we re-aggregate the SF-vs-benchmarks pool (excluding FF++ and DFD pristine
pairs) restricted to participants in each subgroup. The bootstrap clusters by
session_id within the subgroup (so CIs widen on small subgroups, e.g.
knows_deepfakes='no' or age='50+').

Output: sensitivity_subgroup.csv with columns
  subgroup_dim, subgroup_value, n_sessions, question, n_responses,
  rate, rate_lo, rate_hi
where 'question' ∈ {Q1, Q2, Q3} and 'rate' is the SF win rate (Q1, Q2) or the
SF flagged rate (Q3), restricted to vs_type='competitor'.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
OUT_CSV = HERE / "sensitivity_subgroup.csv"

PREFERENCE_QUESTIONS = ["better_made", "more_realistic"]   # Q1, Q2
Q3_KEY = "could_be_fake"
QUESTION_LABEL = {"better_made": "Q1", "more_realistic": "Q2", "could_be_fake": "Q3"}
N_BOOTSTRAP = 1000
SEED = 42

# Subgroup definitions: (column_name, [allowed_values_in_order])
SUBGROUPS = {
    "ai_background": [True, False],
    "knows_deepfakes": ["yes", "heard_term", "no"],
    "age_range": ["18-25", "26-35", "36-50", "50+"],
}


def encode_q3(value):
    """Decompose chosen_video into is_A_fake (SF flagged) for Q3."""
    if value == "A":
        return 1
    if value == "Both":
        return 1
    # 'B', NaN/'Neither' → not flagging SF
    return 0


def winrate_with_ci(
    df_q: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float, int, int]:
    """Compute SF-win rate (chosen_video == 'A') with cluster-bootstrap 95% CI.

    Returns: (point, lo, hi, n_responses, n_sessions).
    """
    n_resp = len(df_q)
    if n_resp == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0

    # Per-session counts: aggregate n_resp and n_sf per session_id for the
    # cluster bootstrap.
    sess_counts = (
        df_q.assign(is_sf=(df_q["chosen_video"] == "A").astype(int))
            .groupby("session_id")
            .agg(n=("chosen_video", "size"), a=("is_sf", "sum"))
            .reset_index()
    )
    point = sess_counts["a"].sum() / sess_counts["n"].sum()

    rng = np.random.default_rng(seed)
    sids = sess_counts["session_id"].to_numpy()
    arr_n = sess_counts["n"].to_numpy()
    arr_a = sess_counts["a"].to_numpy()
    n_sess = len(sids)

    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_sess, size=n_sess)
        tot_n = arr_n[idx].sum()
        tot_a = arr_a[idx].sum()
        boot[b] = tot_a / tot_n if tot_n > 0 else np.nan

    lo = float(np.nanpercentile(boot, 2.5))
    hi = float(np.nanpercentile(boot, 97.5))
    return float(point), lo, hi, n_resp, n_sess


def q3_sf_rate_with_ci(
    df_q3: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float, int, int]:
    """Compute SF-flagged-as-fake rate with cluster-bootstrap 95% CI.

    The SF-flagged binary is is_A_fake (decomposed from chosen_video; 'A' or
    'Both' → 1, else 0), since SF is always video A in our export.
    """
    n_resp = len(df_q3)
    if n_resp == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0

    df_q3 = df_q3.assign(is_sf_fake=df_q3["chosen_video"].apply(encode_q3))
    sess_counts = (
        df_q3.groupby("session_id")
             .agg(n=("chosen_video", "size"), a=("is_sf_fake", "sum"))
             .reset_index()
    )
    point = sess_counts["a"].sum() / sess_counts["n"].sum()

    rng = np.random.default_rng(seed)
    arr_n = sess_counts["n"].to_numpy()
    arr_a = sess_counts["a"].to_numpy()
    n_sess = len(arr_n)

    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_sess, size=n_sess)
        tot_n = arr_n[idx].sum()
        tot_a = arr_a[idx].sum()
        boot[b] = tot_a / tot_n if tot_n > 0 else np.nan

    lo = float(np.nanpercentile(boot, 2.5))
    hi = float(np.nanpercentile(boot, 97.5))
    return float(point), lo, hi, n_resp, n_sess


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    df = df[df["session_id"].isin(set(kept["session_id"]))].copy()

    # Restrict to vs_type = competitor only (the headline benchmark numbers).
    df = df[~df["benchmark_name"].isin({"FF++", "DFD"})].copy()
    print(f"Loaded {len(df):,} responses on {df['session_id'].nunique()} kept "
          f"sessions, vs benchmarks only")

    # Sanity: include the FULL kept-set baseline as an extra reference row.
    rows = []
    for q_key in PREFERENCE_QUESTIONS:
        sub = df[df["question_key"] == q_key]
        rate, lo, hi, n_resp, n_sess = winrate_with_ci(sub)
        rows.append({
            "subgroup_dim": "_baseline",
            "subgroup_value": "all_kept",
            "n_sessions": n_sess,
            "question": QUESTION_LABEL[q_key],
            "n_responses": n_resp,
            "rate": rate, "rate_lo": lo, "rate_hi": hi,
        })
    sub3 = df[df["question_key"] == Q3_KEY]
    rate, lo, hi, n_resp, n_sess = q3_sf_rate_with_ci(sub3)
    rows.append({
        "subgroup_dim": "_baseline",
        "subgroup_value": "all_kept",
        "n_sessions": n_sess,
        "question": "Q3",
        "n_responses": n_resp,
        "rate": rate, "rate_lo": lo, "rate_hi": hi,
    })

    # Per-subgroup sensitivity.
    for dim, values in SUBGROUPS.items():
        for v in values:
            df_sg = df[df[dim] == v]
            if df_sg.empty:
                continue
            for q_key in PREFERENCE_QUESTIONS:
                sub = df_sg[df_sg["question_key"] == q_key]
                rate, lo, hi, n_resp, n_sess = winrate_with_ci(sub)
                rows.append({
                    "subgroup_dim": dim,
                    "subgroup_value": str(v),
                    "n_sessions": n_sess,
                    "question": QUESTION_LABEL[q_key],
                    "n_responses": n_resp,
                    "rate": rate, "rate_lo": lo, "rate_hi": hi,
                })
            sub3 = df_sg[df_sg["question_key"] == Q3_KEY]
            rate, lo, hi, n_resp, n_sess = q3_sf_rate_with_ci(sub3)
            rows.append({
                "subgroup_dim": dim,
                "subgroup_value": str(v),
                "n_sessions": n_sess,
                "question": "Q3",
                "n_responses": n_resp,
                "rate": rate, "rate_lo": lo, "rate_hi": hi,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV.name}: {len(out)} rows")
    print()
    print("=== SUBGROUP SENSITIVITY (vs benchmarks pool) ===")
    for (dim, val), g in out.groupby(["subgroup_dim", "subgroup_value"]):
        n_sess_str = int(g["n_sessions"].iloc[0])
        cells = []
        for _, r in g.iterrows():
            cells.append(f"{r['question']}={r['rate']*100:.1f} "
                         f"[{r['rate_lo']*100:.1f},{r['rate_hi']*100:.1f}]")
        print(f"  {dim:18s} {val:12s} n_sess={n_sess_str:3d}   " + "  ".join(cells))


if __name__ == "__main__":
    main()
