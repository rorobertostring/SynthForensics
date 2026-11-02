#!/usr/bin/env python3
"""
03_winrate.py — Per-benchmark win-rate analysis with cluster-bootstrap 95% CI.

For Q1 (better_made) and Q2 (more_realistic), aggregates the
chosen_video category {A, B, Equal} per benchmark and computes the three
proportions (w_sf, w_other, w_equal) plus 95% CI via cluster bootstrap by
session_id (1000 resamples, percentile 2.5/97.5).

Four aggregation levels are produced:
  1. winrate_aggregate.csv       — per (vs_type, question_key) — used in MAIN
  2. winrate_primary.csv         — per (benchmark, question_key) — used in APPENDIX
  3. winrate_by_generator.csv    — per (benchmark, sf_generator, question_key)
  4. winrate_by_modality.csv     — per (benchmark, sf_modality, question_key)

`vs_type ∈ {competitor, real}` is derived from `is_pristine_pair` so callers
can split SF-vs-Competitor from SF-vs-Real reporting.

Q3 (could_be_fake) is NOT analysed here; it has 4 categories and a different
analysis (binary decomposition + per-benchmark detection rate) handled by
04_q3_detection.py.

Run:
    python 03_winrate.py

Inputs:
    prepared.parquet
    kept_sessions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
OUT_AGGREGATE = HERE / "winrate_aggregate.csv"
OUT_PRIMARY = HERE / "winrate_primary.csv"
OUT_BY_GEN = HERE / "winrate_by_generator.csv"
OUT_BY_MOD = HERE / "winrate_by_modality.csv"

PREFERENCE_QUESTIONS = ["better_made", "more_realistic"]
N_BOOTSTRAP = 1000
SEED = 42
CATEGORIES = ["A", "B", "Equal"]   # for Q1 and Q2 only


def _proportions_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """counts: rows = groups, cols = {A, B, Equal}. Returns proportions."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0)


def compute_winrate(
    df: pd.DataFrame,
    group_cols: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """Aggregate w_sf / w_other / w_equal per group with cluster-bootstrap CI.

    The bootstrap samples session_ids with replacement (clustering by
    participant). For each resample, all responses from sampled sessions are
    re-aggregated per group, and the proportions recomputed.
    """
    # Pre-build a per-session table of counts per (group, category) for fast
    # bootstrap accumulation.
    sess_group_cat = (
        df.groupby([*group_cols, "session_id", "chosen_video"])
          .size()
          .rename("n")
          .reset_index()
    )
    # Pivot so each (group_cols, session_id) row has columns A, B, Equal.
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

    # Point estimate: sum across all sessions, then proportions per group.
    point_counts = pivot.groupby(level=group_cols).sum()
    point_props = _proportions_from_counts(point_counts)

    # Bootstrap.
    rng = np.random.default_rng(seed)
    session_ids = np.array(sorted(df["session_id"].unique()))
    n_sessions = len(session_ids)

    # For fast bootstrap accumulation, build a mapping
    # session_id → (group_index, A_count, B_count, Equal_count).
    pivot_reset = pivot.reset_index()
    # Encode group as a categorical index for vectorised aggregation.
    group_keys = pivot_reset[group_cols].apply(tuple, axis=1)
    unique_groups = list(point_counts.index)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    pivot_reset["_g_idx"] = group_keys.map(group_to_idx)

    # session → list of (g_idx, A, B, Equal) rows (sessions can appear in
    # multiple groups, e.g. one session rates pairs from many benchmarks).
    sess_to_rows: dict[str, list[tuple[int, int, int, int]]] = {
        sid: [] for sid in session_ids
    }
    for _, r in pivot_reset.iterrows():
        sess_to_rows[r["session_id"]].append(
            (int(r["_g_idx"]), int(r["A"]), int(r["B"]), int(r["Equal"]))
        )

    n_groups = len(unique_groups)
    boot_props = {cat: np.zeros((n_bootstrap, n_groups)) for cat in CATEGORIES}

    for b in range(n_bootstrap):
        sampled = rng.choice(session_ids, size=n_sessions, replace=True)
        accum = np.zeros((n_groups, 3), dtype=np.int64)   # cols: A, B, Equal
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

    # Build output dataframe.
    rows = []
    for g_idx, group_key in enumerate(unique_groups):
        # group_key may be a scalar or a tuple
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


def add_vs_type(df_out: pd.DataFrame) -> pd.DataFrame:
    """Derive vs_type ∈ {competitor, real} from benchmark_name."""
    real = {"FF++", "DFD"}
    df_out = df_out.copy()
    df_out["vs_type"] = df_out["benchmark_name"].apply(
        lambda b: "real" if b in real else "competitor"
    )
    return df_out


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    df = df[df["session_id"].isin(set(kept["session_id"]))]
    df = df[df["question_key"].isin(PREFERENCE_QUESTIONS)].copy()
    # Pre-derive vs_type so we can group on it directly.
    real = {"FF++", "DFD"}
    df["vs_type"] = df["benchmark_name"].apply(
        lambda b: "real" if b in real else "competitor"
    )
    print(f"Loaded {len(df):,} responses (Q1+Q2) on {df['session_id'].nunique()} kept sessions")
    print()

    # 0. Aggregate (MAIN paper): one row per (vs_type, question_key)
    aggregate = compute_winrate(df, group_cols=["vs_type", "question_key"])
    aggregate = aggregate.sort_values(["vs_type", "question_key"])
    aggregate.to_csv(OUT_AGGREGATE, index=False)
    print(f"Saved {OUT_AGGREGATE.name}: {len(aggregate)} rows")

    # 1. Primary aggregation (APPENDIX): per-benchmark breakdown
    primary = compute_winrate(df, group_cols=["benchmark_name", "question_key"])
    primary = add_vs_type(primary).sort_values(
        ["vs_type", "benchmark_name", "question_key"]
    )
    primary.to_csv(OUT_PRIMARY, index=False)
    print(f"Saved {OUT_PRIMARY.name}: {len(primary)} rows")

    # 2. By generator
    by_gen = compute_winrate(
        df, group_cols=["benchmark_name", "sf_generator", "question_key"]
    )
    by_gen = add_vs_type(by_gen).sort_values(
        ["vs_type", "benchmark_name", "sf_generator", "question_key"]
    )
    by_gen.to_csv(OUT_BY_GEN, index=False)
    print(f"Saved {OUT_BY_GEN.name}: {len(by_gen)} rows")

    # 3. By modality
    by_mod = compute_winrate(
        df, group_cols=["benchmark_name", "sf_modality", "question_key"]
    )
    by_mod = add_vs_type(by_mod).sort_values(
        ["vs_type", "benchmark_name", "sf_modality", "question_key"]
    )
    by_mod.to_csv(OUT_BY_MOD, index=False)
    print(f"Saved {OUT_BY_MOD.name}: {len(by_mod)} rows")

    # Quick sanity print on aggregate (the MAIN-paper numbers)
    print()
    print("=== AGGREGATE win rates (MAIN PAPER NUMBERS) ===")
    for _, r in aggregate.iterrows():
        print(f"  vs_{r['vs_type']:10s} {r['question_key']:15s} "
              f"w_sf={r['w_sf']:.3f} [{r['w_sf_lo']:.3f},{r['w_sf_hi']:.3f}]  "
              f"w_other={r['w_other']:.3f} [{r['w_other_lo']:.3f},{r['w_other_hi']:.3f}]  "
              f"w_eq={r['w_equal']:.3f} [{r['w_equal_lo']:.3f},{r['w_equal_hi']:.3f}]  "
              f"n={int(r['n_responses'])}")
    print()
    print("=== PRIMARY win rates (APPENDIX, sorted by w_sf desc within vs_type) ===")
    for vt in ["competitor", "real"]:
        sub = primary[primary["vs_type"] == vt].copy()
        sub = sub.sort_values("w_sf", ascending=False)
        print(f"\n  {vt}:")
        for _, r in sub.iterrows():
            print(f"    {r['benchmark_name']:12s} {r['question_key']:15s} "
                  f"w_sf={r['w_sf']:.3f} [{r['w_sf_lo']:.3f},{r['w_sf_hi']:.3f}]  "
                  f"w_other={r['w_other']:.3f}  w_eq={r['w_equal']:.3f}  "
                  f"n={int(r['n_responses'])}")


if __name__ == "__main__":
    main()
