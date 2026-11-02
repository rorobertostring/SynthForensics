#!/usr/bin/env python3
"""
07_per_sf_breakdown.py - SF-side breakdown, aggregated cross-benchmark.

Aggregates win rates (Q1, Q2) and Q3 detection rates by SynthForensics
generator and by SynthForensics modality, pooling responses across all
nine competitor benchmarks (vs_type=competitor) or both real baselines
(vs_type=real). This complements the per-benchmark breakdowns of
winrate_by_generator.csv / q3_by_generator.csv (and ..._by_modality.csv),
which keep the benchmark axis as a separate dimension.

Bootstrap procedure: cluster-by-participant, B=1000 resamples, seed=42,
percentile 2.5/97.5.

Outputs (new files; existing CSVs are NOT modified):
  - winrate_per_generator.csv       per (sf_generator, question_key, vs_type)
  - winrate_per_modality.csv        per (sf_modality,  question_key, vs_type)
  - q3_per_generator.csv            per (sf_generator, vs_type)
  - q3_per_modality.csv             per (sf_modality,  vs_type)
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"

OUT_WR_GEN = HERE / "winrate_per_generator.csv"
OUT_WR_MOD = HERE / "winrate_per_modality.csv"
OUT_Q3_GEN = HERE / "q3_per_generator.csv"
OUT_Q3_MOD = HERE / "q3_per_modality.csv"

PREFERENCE_QUESTIONS = ["better_made", "more_realistic"]
Q3_KEY = "could_be_fake"
N_BOOTSTRAP = 1000
SEED = 42
CATEGORIES = ["A", "B", "Equal"]
REAL = {"FF++", "DFD"}


def _proportions_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0)


def compute_winrate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    sess_group_cat = (
        df.groupby([*group_cols, "session_id", "chosen_video"])
          .size()
          .rename("n")
          .reset_index()
    )
    pivot = (
        sess_group_cat.pivot_table(
            index=[*group_cols, "session_id"],
            columns="chosen_video",
            values="n",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=CATEGORIES, fill_value=0)
    )

    point_counts = pivot.groupby(level=group_cols).sum()
    point_props = _proportions_from_counts(point_counts)

    pivot_reset = pivot.reset_index()
    group_keys = pivot_reset[group_cols].apply(tuple, axis=1)
    unique_groups = list(point_counts.index)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    pivot_reset["_g_idx"] = group_keys.map(group_to_idx)

    session_ids = np.array(sorted(df["session_id"].unique()))
    sess_to_rows: dict[str, list[tuple[int, int, int, int]]] = {
        sid: [] for sid in session_ids
    }
    for _, r in pivot_reset.iterrows():
        sess_to_rows[r["session_id"]].append(
            (int(r["_g_idx"]), int(r["A"]), int(r["B"]), int(r["Equal"]))
        )

    n_groups = len(unique_groups)
    boot_props = {cat: np.zeros((N_BOOTSTRAP, n_groups)) for cat in CATEGORIES}

    rng = np.random.default_rng(SEED)
    n_sessions = len(session_ids)
    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(session_ids, size=n_sessions, replace=True)
        accum = np.zeros((n_groups, 3), dtype=np.int64)
        for sid in sampled:
            for g_idx, a, b_, e in sess_to_rows[sid]:
                accum[g_idx, 0] += a
                accum[g_idx, 1] += b_
                accum[g_idx, 2] += e
        totals = accum.sum(axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            props = accum / totals[:, None]
        boot_props["A"][b] = props[:, 0]
        boot_props["B"][b] = props[:, 1]
        boot_props["Equal"][b] = props[:, 2]

    rows = []
    for g_idx, group_key in enumerate(unique_groups):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
        else:
            group_dict = {group_cols[0]: group_key}
        n_resp = int(point_counts.loc[group_key].sum())
        rec = dict(group_dict)
        rec["n_responses"] = n_resp
        for cat, label in [("A", "w_sf"), ("B", "w_other"), ("Equal", "w_equal")]:
            point = float(point_props.loc[group_key, cat])
            ci_lo = float(np.nanpercentile(boot_props[cat][:, g_idx], 2.5))
            ci_hi = float(np.nanpercentile(boot_props[cat][:, g_idx], 97.5))
            rec[label] = point
            rec[f"{label}_lo"] = ci_lo
            rec[f"{label}_hi"] = ci_hi
        rows.append(rec)

    return pd.DataFrame(rows)


def encode_q3(value: str) -> tuple[int, int]:
    if value == "A":
        return 1, 0
    if value == "B":
        return 0, 1
    if value == "Both":
        return 1, 1
    return 0, 0


def compute_q3_rates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    pairs = df["chosen_video"].apply(encode_q3)
    df["is_A_fake"] = [p[0] for p in pairs]
    df["is_B_fake"] = [p[1] for p in pairs]

    sess_group = (
        df.groupby([*group_cols, "session_id"])
          .agg(n=("chosen_video", "size"),
               a=("is_A_fake", "sum"),
               b=("is_B_fake", "sum"))
          .reset_index()
    )

    point = sess_group.groupby(group_cols).agg(n=("n", "sum"),
                                               a=("a", "sum"),
                                               b=("b", "sum"))
    point["sf_det_rate"] = point["a"] / point["n"]
    point["other_det_rate"] = point["b"] / point["n"]

    unique_groups = list(point.index)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    if isinstance(unique_groups[0], tuple):
        sess_group["_g_idx"] = sess_group[group_cols].apply(tuple, axis=1).map(group_to_idx)
    else:
        sess_group["_g_idx"] = sess_group[group_cols[0]].map(group_to_idx)

    session_ids = np.array(sorted(df["session_id"].unique()))
    sess_to_rows: dict[str, list[tuple[int, int, int, int]]] = {
        sid: [] for sid in session_ids
    }
    for _, r in sess_group.iterrows():
        sess_to_rows[r["session_id"]].append(
            (int(r["_g_idx"]), int(r["n"]), int(r["a"]), int(r["b"]))
        )

    n_groups = len(unique_groups)
    boot_sf = np.zeros((N_BOOTSTRAP, n_groups))
    boot_oth = np.zeros((N_BOOTSTRAP, n_groups))

    rng = np.random.default_rng(SEED)
    n_sessions = len(session_ids)
    for i in range(N_BOOTSTRAP):
        sampled = rng.choice(session_ids, size=n_sessions, replace=True)
        accum = np.zeros((n_groups, 3), dtype=np.int64)
        for sid in sampled:
            for g_idx, n_, a_, b_ in sess_to_rows[sid]:
                accum[g_idx, 0] += n_
                accum[g_idx, 1] += a_
                accum[g_idx, 2] += b_
        totals = accum[:, 0].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_sf[i] = accum[:, 1] / totals
            boot_oth[i] = accum[:, 2] / totals

    rows = []
    for g_idx, group_key in enumerate(unique_groups):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
        else:
            group_dict = {group_cols[0]: group_key}
        rec = dict(group_dict)
        rec["n_responses"] = int(point.loc[group_key, "n"])
        rec["sf_det_rate"] = float(point.loc[group_key, "sf_det_rate"])
        rec["sf_det_rate_lo"] = float(np.nanpercentile(boot_sf[:, g_idx], 2.5))
        rec["sf_det_rate_hi"] = float(np.nanpercentile(boot_sf[:, g_idx], 97.5))
        rec["other_det_rate"] = float(point.loc[group_key, "other_det_rate"])
        rec["other_det_rate_lo"] = float(np.nanpercentile(boot_oth[:, g_idx], 2.5))
        rec["other_det_rate_hi"] = float(np.nanpercentile(boot_oth[:, g_idx], 97.5))
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    df = df[df["session_id"].isin(set(kept["session_id"]))].copy()
    df["vs_type"] = df["benchmark_name"].apply(
        lambda b: "real" if b in REAL else "competitor"
    )

    df_q12 = df[df["question_key"].isin(PREFERENCE_QUESTIONS)].copy()
    df_q3 = df[df["question_key"] == Q3_KEY].copy()

    print(f"Q1+Q2: {len(df_q12):,} responses on {df_q12['session_id'].nunique()} kept sessions")
    print(f"Q3:    {len(df_q3):,} responses on {df_q3['session_id'].nunique()} kept sessions")
    print()

    # Win rates
    wr_gen = compute_winrate(df_q12, ["sf_generator", "question_key", "vs_type"])
    wr_gen = wr_gen.sort_values(["vs_type", "sf_generator", "question_key"])
    wr_gen.to_csv(OUT_WR_GEN, index=False)
    print(f"Saved {OUT_WR_GEN.name}: {len(wr_gen)} rows")

    wr_mod = compute_winrate(df_q12, ["sf_modality", "question_key", "vs_type"])
    wr_mod = wr_mod.sort_values(["vs_type", "sf_modality", "question_key"])
    wr_mod.to_csv(OUT_WR_MOD, index=False)
    print(f"Saved {OUT_WR_MOD.name}: {len(wr_mod)} rows")

    # Q3 detection rates
    q3_gen = compute_q3_rates(df_q3, ["sf_generator", "vs_type"])
    q3_gen = q3_gen.sort_values(["vs_type", "sf_generator"])
    q3_gen.to_csv(OUT_Q3_GEN, index=False)
    print(f"Saved {OUT_Q3_GEN.name}: {len(q3_gen)} rows")

    q3_mod = compute_q3_rates(df_q3, ["sf_modality", "vs_type"])
    q3_mod = q3_mod.sort_values(["vs_type", "sf_modality"])
    q3_mod.to_csv(OUT_Q3_MOD, index=False)
    print(f"Saved {OUT_Q3_MOD.name}: {len(q3_mod)} rows")

    # Sanity print
    print()
    print("=== WIN RATE per generator (vs_type=competitor) ===")
    sub = wr_gen[wr_gen["vs_type"] == "competitor"]
    sub = sub.sort_values(["question_key", "w_sf"], ascending=[True, False])
    for _, r in sub.iterrows():
        print(f"  {r['sf_generator']:25s} {r['question_key']:15s} "
              f"w_sf={r['w_sf']:.3f} [{r['w_sf_lo']:.3f},{r['w_sf_hi']:.3f}]  "
              f"n={int(r['n_responses'])}")

    print()
    print("=== Q3 SF flag rate per generator (vs_type=competitor) ===")
    sub = q3_gen[q3_gen["vs_type"] == "competitor"].sort_values("sf_det_rate")
    for _, r in sub.iterrows():
        print(f"  {r['sf_generator']:25s}  SF flag: {r['sf_det_rate']:.3f} "
              f"[{r['sf_det_rate_lo']:.3f},{r['sf_det_rate_hi']:.3f}]  "
              f"n={int(r['n_responses'])}")

    print()
    print("=== WIN RATE per modality (vs_type=competitor) ===")
    sub = wr_mod[wr_mod["vs_type"] == "competitor"].sort_values(["question_key", "sf_modality"])
    for _, r in sub.iterrows():
        print(f"  {r['sf_modality']:5s} {r['question_key']:15s} "
              f"w_sf={r['w_sf']:.3f} [{r['w_sf_lo']:.3f},{r['w_sf_hi']:.3f}]  "
              f"n={int(r['n_responses'])}")

    print()
    print("=== Q3 SF flag rate per modality (vs_type=competitor) ===")
    sub = q3_mod[q3_mod["vs_type"] == "competitor"].sort_values("sf_det_rate")
    for _, r in sub.iterrows():
        print(f"  {r['sf_modality']:5s}  SF flag: {r['sf_det_rate']:.3f} "
              f"[{r['sf_det_rate_lo']:.3f},{r['sf_det_rate_hi']:.3f}]  "
              f"n={int(r['n_responses'])}")


if __name__ == "__main__":
    main()
