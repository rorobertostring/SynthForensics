#!/usr/bin/env python3
"""
04_q3_detection.py — Q3 (could_be_fake) per-benchmark human detection rate.

Q3 has four categories: A, B, Both, Neither. We decompose each response
into two independent binary judgments:

    chosen_video = 'A'    → is_A_fake=1, is_B_fake=0
    chosen_video = 'B'    → is_A_fake=0, is_B_fake=1
    chosen_video = 'Both' → is_A_fake=1, is_B_fake=1
    chosen_video = NaN    → is_A_fake=0, is_B_fake=0   (= 'Neither')

Then per benchmark we report:
    sf_det_rate    = mean(is_A_fake)  — fraction flagging SF as a forgery
    other_det_rate = mean(is_B_fake)  — fraction flagging competitor/real as fake

Note: sf_det_rate + other_det_rate does NOT sum to 1 (they are independent
binary judgments derived from a 4-category response; the 'Both' answers
contribute to both rates, the 'Neither' answers contribute to neither).

Cluster bootstrap 95% CI by session_id × 1000 resamples.

Four aggregation levels are produced:
  1. q3_aggregate.csv       — per vs_type — used in MAIN
  2. q3_primary.csv         — per benchmark — used in APPENDIX
  3. q3_by_generator.csv    — per (benchmark, sf_generator)
  4. q3_by_modality.csv     — per (benchmark, sf_modality)

Run:
    python 04_q3_detection.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
OUT_AGGREGATE = HERE / "q3_aggregate.csv"
OUT_PRIMARY = HERE / "q3_primary.csv"
OUT_BY_GEN = HERE / "q3_by_generator.csv"
OUT_BY_MOD = HERE / "q3_by_modality.csv"

Q3_KEY = "could_be_fake"
N_BOOTSTRAP = 1000
SEED = 42


def encode_q3(value: str) -> tuple[int, int]:
    """Decompose chosen_video into (is_A_fake, is_B_fake)."""
    if value == "A":
        return 1, 0
    if value == "B":
        return 0, 1
    if value == "Both":
        return 1, 1
    # NaN, 'Neither', anything else → no flag
    return 0, 0


def compute_q3_rates(
    df: pd.DataFrame,
    group_cols: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """Aggregate sf_det_rate / other_det_rate per group with cluster-bootstrap CI.

    The bootstrap samples session_ids with replacement (clustering by
    participant). For each resample, all responses from sampled sessions are
    re-aggregated per group, and the two means recomputed.
    """
    # Encode is_A_fake / is_B_fake.
    df = df.copy()
    pairs = df["chosen_video"].apply(encode_q3)
    df["is_A_fake"] = [p[0] for p in pairs]
    df["is_B_fake"] = [p[1] for p in pairs]

    # Per (group, session) sums. We track totals separately because
    # sf_det_rate = sum(is_A_fake) / n_responses (and same for B).
    sess_group = (
        df.groupby([*group_cols, "session_id"])
          .agg(n=("chosen_video", "size"),
               a=("is_A_fake", "sum"),
               b=("is_B_fake", "sum"))
          .reset_index()
    )

    # Point estimate.
    point = sess_group.groupby(group_cols).agg(n=("n", "sum"),
                                               a=("a", "sum"),
                                               b=("b", "sum"))
    point["sf_det_rate"] = point["a"] / point["n"]
    point["other_det_rate"] = point["b"] / point["n"]

    # Build session → list of (g_idx, n, a, b) for bootstrap accumulation.
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
    boot_sf = np.zeros((n_bootstrap, n_groups))
    boot_oth = np.zeros((n_bootstrap, n_groups))

    rng = np.random.default_rng(seed)
    n_sessions = len(session_ids)

    for i in range(n_bootstrap):
        sampled = rng.choice(session_ids, size=n_sessions, replace=True)
        accum = np.zeros((n_groups, 3), dtype=np.int64)   # cols: n, a, b
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


def add_vs_type(df_out: pd.DataFrame) -> pd.DataFrame:
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
    df = df[df["question_key"] == Q3_KEY].copy()
    real = {"FF++", "DFD"}
    df["vs_type"] = df["benchmark_name"].apply(
        lambda b: "real" if b in real else "competitor"
    )
    print(f"Loaded {len(df):,} Q3 responses on {df['session_id'].nunique()} kept sessions")
    print()

    # 0. Aggregate (MAIN paper)
    aggregate = compute_q3_rates(df, group_cols=["vs_type"])
    aggregate = aggregate.sort_values("vs_type")
    aggregate.to_csv(OUT_AGGREGATE, index=False)
    print(f"Saved {OUT_AGGREGATE.name}: {len(aggregate)} rows")

    # 1. Primary (APPENDIX)
    primary = compute_q3_rates(df, group_cols=["benchmark_name"])
    primary = add_vs_type(primary).sort_values(["vs_type", "benchmark_name"])
    primary.to_csv(OUT_PRIMARY, index=False)
    print(f"Saved {OUT_PRIMARY.name}: {len(primary)} rows")

    # 2. By generator
    by_gen = compute_q3_rates(df, group_cols=["benchmark_name", "sf_generator"])
    by_gen = add_vs_type(by_gen).sort_values(
        ["vs_type", "benchmark_name", "sf_generator"]
    )
    by_gen.to_csv(OUT_BY_GEN, index=False)
    print(f"Saved {OUT_BY_GEN.name}: {len(by_gen)} rows")

    # 3. By modality
    by_mod = compute_q3_rates(df, group_cols=["benchmark_name", "sf_modality"])
    by_mod = add_vs_type(by_mod).sort_values(
        ["vs_type", "benchmark_name", "sf_modality"]
    )
    by_mod.to_csv(OUT_BY_MOD, index=False)
    print(f"Saved {OUT_BY_MOD.name}: {len(by_mod)} rows")

    print()
    print("=== AGGREGATE Q3 detection rates (MAIN PAPER NUMBERS) ===")
    for _, r in aggregate.iterrows():
        print(f"  vs_{r['vs_type']:10s}  "
              f"SF flagged: {r['sf_det_rate']:.3f} [{r['sf_det_rate_lo']:.3f},{r['sf_det_rate_hi']:.3f}]   "
              f"other flagged: {r['other_det_rate']:.3f} [{r['other_det_rate_lo']:.3f},{r['other_det_rate_hi']:.3f}]   "
              f"n={int(r['n_responses'])}")
    print()
    print("=== PRIMARY Q3 detection rates (APPENDIX) ===")
    for vt in ["competitor", "real"]:
        sub = primary[primary["vs_type"] == vt].copy().sort_values("sf_det_rate")
        print(f"\n  {vt}:")
        for _, r in sub.iterrows():
            print(f"    {r['benchmark_name']:12s}  "
                  f"SF flagged: {r['sf_det_rate']:.3f} [{r['sf_det_rate_lo']:.3f},{r['sf_det_rate_hi']:.3f}]   "
                  f"other flagged: {r['other_det_rate']:.3f} [{r['other_det_rate_lo']:.3f},{r['other_det_rate_hi']:.3f}]   "
                  f"n={int(r['n_responses'])}")


if __name__ == "__main__":
    main()
