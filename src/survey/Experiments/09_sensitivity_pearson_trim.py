#!/usr/bin/env python3
"""
09_sensitivity_pearson_trim.py — Robustness check against the consensus-based
post-screening we deliberately did not apply.

Question addressed: would the headline numbers change if the Stage-2
disengagement check were replaced with a fixed-fraction consensus-correlation
trim?

Baseline = post-Stage-1 set of 120 sessions (those that passed the completion
filter), i.e. all retained 118 + the 2 sessions rejected by Stage-2 only.
Computing the Pearson r_1 on the full 159 raw set is statistically unstable
because many incomplete sessions have too few rated pairs.

Procedure:
  1. Encode Q1+Q2 responses as A=+1, B=-1, Equal=0.
  2. For each (stimulus pair, question), compute the consensus mean across the
     120 post-Stage-1 sessions.
  3. For each session, compute the Pearson r_1 between the session's response
     vector and the corresponding consensus vector.
  4. Sort sessions by r_1 ascending and drop the bottom 25%.
  5. Recompute the headline numbers (Q1/Q2 SF win rate, Q3 SF flagged rate vs
     benchmarks) on the trimmed subset; compare to the full kept (118) and
     post-Stage-1 (120) baselines.

Output: sensitivity_pearson_trim.csv with one row per (variant, question).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
SCREENING_LOG_CSV = HERE / "screening_log.csv"
OUT_CSV = HERE / "sensitivity_pearson_trim.csv"
OUT_R1 = HERE / "sensitivity_pearson_trim_r1.csv"

PREFERENCE_QUESTIONS = ["better_made", "more_realistic"]
Q3_KEY = "could_be_fake"
QUESTION_LABEL = {"better_made": "Q1", "more_realistic": "Q2", "could_be_fake": "Q3"}
ENCODE_PC = {"A": 1.0, "B": -1.0, "Equal": 0.0}
N_BOOTSTRAP = 1000
SEED = 42
TRIM_FRACTION = 0.25


def encode_q3(value):
    if value == "A":
        return 1
    if value == "Both":
        return 1
    return 0


def winrate_with_ci(df_q, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    n_resp = len(df_q)
    if n_resp == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    sess_counts = (
        df_q.assign(is_sf=(df_q["chosen_video"] == "A").astype(int))
            .groupby("session_id")
            .agg(n=("chosen_video", "size"), a=("is_sf", "sum"))
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
    return float(point), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)), n_resp, n_sess


def q3_sf_rate_with_ci(df_q3, n_bootstrap=N_BOOTSTRAP, seed=SEED):
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
    return float(point), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)), n_resp, n_sess


def compute_session_r1(df_pc: pd.DataFrame) -> pd.DataFrame:
    """Pearson r_1 per session against the consensus mean on Q1+Q2 trials.

    Each (pair_number, question_key) point is encoded ∈ {-1, 0, +1} via
    ENCODE_PC. Stimulus identity is captured by (video_a_path, video_b_path)
    since pair_number is per-session shuffled order, NOT a global pair id.
    """
    df_pc = df_pc.copy()
    df_pc["pc_score"] = df_pc["chosen_video"].map(ENCODE_PC)
    df_pc = df_pc.dropna(subset=["pc_score"])

    # Define a global pair identifier from the two video paths, since
    # pair_number is the per-session presentation index (different participants
    # see the same (video_a, video_b) under different pair_numbers).
    df_pc["pair_uid"] = df_pc["video_a_path"] + "||" + df_pc["video_b_path"]

    # Consensus per (pair_uid, question_key): mean across sessions that rated
    # this point.
    consensus = (
        df_pc.groupby(["pair_uid", "question_key"])["pc_score"]
             .mean()
             .rename("consensus")
             .reset_index()
    )
    df_pc = df_pc.merge(consensus, on=["pair_uid", "question_key"], how="left")

    rows = []
    for sid, sg in df_pc.groupby("session_id"):
        x = sg["pc_score"].to_numpy()
        y = sg["consensus"].to_numpy()
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            r = float("nan")
        else:
            r = float(pearsonr(x, y).statistic)
        rows.append({"session_id": sid, "n_points": len(x), "r1": r})
    return pd.DataFrame(rows)


def main() -> None:
    df_full = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    screening = pd.read_csv(SCREENING_LOG_CSV)
    # Post-Stage-1 set = kept (118) ∪ Stage-2-rejected (variance) = 120.
    stage2_rejected = set(screening[screening["reason"] == "variance"]["session_id"])
    post_stage1 = set(kept["session_id"]) | stage2_rejected
    print(f"Post-Stage-1 set: {len(post_stage1)} sessions "
          f"(kept {len(kept)} + Stage-2 rejected {len(stage2_rejected)})")

    df_120 = df_full[df_full["session_id"].isin(post_stage1)].copy()

    # Step 1-3: compute r_1 per session on Q1+Q2 from the 120-session pool.
    df_pc = df_120[df_120["question_key"].isin(PREFERENCE_QUESTIONS)].copy()
    r1 = compute_session_r1(df_pc).sort_values("r1")
    r1.to_csv(OUT_R1, index=False)
    print(f"Saved {OUT_R1.name}: {len(r1)} sessions with r_1")
    print(f"  r_1 distribution: median={r1['r1'].median():.3f}, "
          f"min={r1['r1'].min():.3f}, max={r1['r1'].max():.3f}")

    # Step 4: drop bottom 25% by r_1.
    n_drop = int(round(len(r1) * TRIM_FRACTION))
    keep_ids = set(r1.iloc[n_drop:]["session_id"])
    drop_ids = set(r1.iloc[:n_drop]["session_id"])
    print(f"  trimming bottom {TRIM_FRACTION*100:.0f}%: drop {len(drop_ids)}, keep {len(keep_ids)}")

    # Restrict to vs_type = competitor only (the headline benchmark numbers).
    df_bench = df_full[~df_full["benchmark_name"].isin({"FF++", "DFD"})].copy()

    rows = []
    for variant_name, sids in [
        ("kept (n=118)", set(kept["session_id"])),
        (f"post-Stage-1 (n={len(post_stage1)})", post_stage1),
        (f"Pearson-trimmed (n={len(keep_ids)})", keep_ids),
    ]:
        df_v = df_bench[df_bench["session_id"].isin(sids)]
        for q_key in PREFERENCE_QUESTIONS:
            sub = df_v[df_v["question_key"] == q_key]
            rate, lo, hi, n_resp, n_sess = winrate_with_ci(sub)
            rows.append({
                "variant": variant_name,
                "question": QUESTION_LABEL[q_key],
                "n_sessions": n_sess,
                "n_responses": n_resp,
                "rate": rate, "rate_lo": lo, "rate_hi": hi,
            })
        sub3 = df_v[df_v["question_key"] == Q3_KEY]
        rate, lo, hi, n_resp, n_sess = q3_sf_rate_with_ci(sub3)
        rows.append({
            "variant": variant_name,
            "question": "Q3",
            "n_sessions": n_sess,
            "n_responses": n_resp,
            "rate": rate, "rate_lo": lo, "rate_hi": hi,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV.name}: {len(out)} rows")
    print()
    print("=== KEPT vs TRIMMED-BY-PEARSON (vs benchmarks pool) ===")
    for variant, g in out.groupby("variant", sort=False):
        cells = []
        for _, r in g.iterrows():
            cells.append(f"{r['question']}={r['rate']*100:.1f} "
                         f"[{r['rate_lo']*100:.1f},{r['rate_hi']*100:.1f}]")
        n_sess = int(g["n_sessions"].iloc[0])
        print(f"  {variant:25s} n_sess={n_sess:3d}   " + "  ".join(cells))


if __name__ == "__main__":
    main()
