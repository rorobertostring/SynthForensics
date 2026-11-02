#!/usr/bin/env python3
"""
10_split_half_reliability.py — Split-half reliability of per-pair aggregate estimates.

Krippendorff's alpha measures agreement between INDIVIDUAL raters; the
paper's headline claims rest on PER-PAIR AGGREGATES (win rates, SF-flag
rates) computed over ~37 ratings per pair. This script quantifies the
reliability of those aggregates directly: it repeatedly partitions the
retained sessions into two random halves, recomputes the per-pair statistics
independently in each half, and correlates the two resulting vectors across
pairs.

Per-pair statistics (mirroring the published pipeline):
    Q1/Q2 (better_made, more_realistic):
        value per response = 1 if chosen_video == 'A' (SF preferred) else 0.
        NaN chosen_video = missing response from partial session → excluded
        (same convention as 03_winrate.py, where groupby drops NaN).
    Q3 (could_be_fake):
        value per response = is_A_fake from the binary decomposition of
        04_q3_detection.py (A or Both → 1; B or NaN='Neither' → 0).

For each of N_SPLITS random partitions (59/59 sessions, split by session so
the rater stays the clustering unit, consistent with the cluster bootstrap):
    - per-pair mean value in each half (pairs with >= 1 rating in BOTH halves)
    - Pearson r and Spearman rho between the two per-pair vectors
    - majority-direction agreement: fraction of pairs where both halves
      agree on the binary verdict (Q1/Q2: w_sf > w_other in that half;
      Q3: SF-flag rate > 0.5 in that half)

Strata: computed separately on the 100 SF-vs-benchmark pairs (primary; the
headline claims concern these) and on all 110 pairs (secondary). Reported:
mean and percentile 2.5/97.5 across splits, plus the Spearman-Brown
correction to full-sample reliability 2r/(1+r) and, for reference, the
theoretical Spearman-Brown prediction from the overall alpha values
(Appendix C.4.4).

Run:
    python 10_split_half_reliability.py

Inputs:
    prepared.parquet
    kept_sessions.csv

Output:
    split_half_reliability.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INPUT_PARQUET = HERE / "prepared.parquet"
KEPT_CSV = HERE / "kept_sessions.csv"
OUT_CSV = HERE / "split_half_reliability.csv"

N_SPLITS = 1000
SEED = 42
QUESTIONS = ["better_made", "more_realistic", "could_be_fake"]
REAL_BENCHMARKS = {"FF++", "DFD"}

# Overall Krippendorff's alpha (Appendix C.4.4), used only for the theoretical
# Spearman-Brown reference printed alongside the empirical results.
ALPHA_OVERALL = {"better_made": 0.260, "more_realistic": 0.247, "could_be_fake": 0.152}


def build_pair_session_matrix(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """Return a (pair x session) matrix of response values for one question.

    Cells are NaN where the session did not rate the pair. Pair identity is
    video_a_path (110 unique stimuli; A is always the SynthForensics video).
    """
    q = df[df["question_key"] == question].copy()
    if question == "could_be_fake":
        # Binary decomposition of 04_q3_detection.py: A/Both flag SF;
        # B and NaN ('Neither') do not. Rows themselves are valid responses.
        q["value"] = q["chosen_video"].isin(["A", "Both"]).astype(float)
    else:
        # Q1/Q2: NaN chosen_video is a missing response → drop.
        q = q.dropna(subset=["chosen_video"])
        q["value"] = (q["chosen_video"] == "A").astype(float)

    return q.pivot_table(
        index="video_a_path", columns="session_id", values="value", aggfunc="mean"
    )


def majority_direction(means: np.ndarray, question: str, counts_a: np.ndarray,
                       counts_b: np.ndarray) -> np.ndarray:
    """Binary verdict per pair for one half.

    Q1/Q2: SF wins the pair iff w_sf > w_other (share of 'A' responses
    strictly greater than share of 'B' responses; Equal responses dilute
    both). Q3: the majority of raters flags SF iff flag rate > 0.5.
    """
    if question == "could_be_fake":
        return means > 0.5
    return counts_a > counts_b


def main() -> None:
    df = pd.read_parquet(INPUT_PARQUET)
    kept = pd.read_csv(KEPT_CSV)
    df = df[df["session_id"].isin(set(kept["session_id"]))].copy()

    # Stratum membership per pair (video_a_path).
    pair_bench = (
        df.groupby("video_a_path")["benchmark_name"].first()
    )
    is_vs_benchmark = ~pair_bench.isin(REAL_BENCHMARKS)

    sessions = np.array(sorted(df["session_id"].unique()))
    n_half = len(sessions) // 2
    print(f"{len(sessions)} kept sessions → halves of {n_half}/{len(sessions) - n_half}")

    # Pre-build per-question matrices. For Q1/Q2 the majority direction needs
    # the A-vs-B comparison, so also build A-count and B-count matrices.
    matrices: dict[str, pd.DataFrame] = {}
    ab_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for question in QUESTIONS:
        matrices[question] = build_pair_session_matrix(df, question)
        if question != "could_be_fake":
            q = df[(df["question_key"] == question)].dropna(subset=["chosen_video"])
            a = (
                q.assign(v=(q["chosen_video"] == "A").astype(float))
                 .pivot_table(index="video_a_path", columns="session_id",
                              values="v", aggfunc="mean")
            )
            b = (
                q.assign(v=(q["chosen_video"] == "B").astype(float))
                 .pivot_table(index="video_a_path", columns="session_id",
                              values="v", aggfunc="mean")
            )
            ab_matrices[question] = (a, b)

    strata = {
        "vs_benchmark": lambda idx: idx[is_vs_benchmark.reindex(idx).values],
        "all_pairs": lambda idx: idx,
    }

    rng = np.random.default_rng(SEED)
    results: dict[tuple[str, str], dict[str, list[float]]] = {
        (q, s): {"pearson": [], "spearman": [], "majority": [], "n_ratings": []}
        for q in QUESTIONS for s in strata
    }

    for _ in range(N_SPLITS):
        perm = rng.permutation(sessions)
        half1, half2 = perm[:n_half], perm[n_half:]

        for question in QUESTIONS:
            mat = matrices[question]
            m1 = mat.reindex(columns=half1)
            m2 = mat.reindex(columns=half2)
            mean1_all = m1.mean(axis=1)
            mean2_all = m2.mean(axis=1)
            n1_all = m1.notna().sum(axis=1)
            n2_all = m2.notna().sum(axis=1)

            if question != "could_be_fake":
                a_mat, b_mat = ab_matrices[question]
                a1 = a_mat.reindex(columns=half1).mean(axis=1)
                b1 = b_mat.reindex(columns=half1).mean(axis=1)
                a2 = a_mat.reindex(columns=half2).mean(axis=1)
                b2 = b_mat.reindex(columns=half2).mean(axis=1)

            for stratum, select in strata.items():
                pairs = select(mat.index)
                valid = (n1_all.reindex(pairs) >= 1) & (n2_all.reindex(pairs) >= 1)
                pairs = pairs[valid.values]

                v1 = mean1_all.reindex(pairs).values
                v2 = mean2_all.reindex(pairs).values

                pearson = float(np.corrcoef(v1, v2)[0, 1])
                r1 = pd.Series(v1).rank().values
                r2 = pd.Series(v2).rank().values
                spearman = float(np.corrcoef(r1, r2)[0, 1])

                if question == "could_be_fake":
                    d1 = majority_direction(v1, question, None, None)
                    d2 = majority_direction(v2, question, None, None)
                else:
                    d1 = majority_direction(
                        v1, question,
                        a1.reindex(pairs).values, b1.reindex(pairs).values)
                    d2 = majority_direction(
                        v2, question,
                        a2.reindex(pairs).values, b2.reindex(pairs).values)
                majority = float(np.mean(d1 == d2))

                res = results[(question, stratum)]
                res["pearson"].append(pearson)
                res["spearman"].append(spearman)
                res["majority"].append(majority)
                res["n_ratings"].append(
                    float((n1_all.reindex(pairs).mean() + n2_all.reindex(pairs).mean()) / 2)
                )

    rows = []
    for (question, stratum), res in results.items():
        pearson = np.array(res["pearson"])
        spearman = np.array(res["spearman"])
        majority = np.array(res["majority"])
        mean_r = float(pearson.mean())
        k_half = float(np.mean(res["n_ratings"]))
        alpha = ALPHA_OVERALL[question]
        rows.append({
            "question_key": question,
            "stratum": stratum,
            "n_pairs": int(110 if stratum == "all_pairs" else 100),
            "mean_ratings_per_half": round(k_half, 1),
            "pearson_mean": mean_r,
            "pearson_lo": float(np.percentile(pearson, 2.5)),
            "pearson_hi": float(np.percentile(pearson, 97.5)),
            "spearman_mean": float(spearman.mean()),
            "spearman_lo": float(np.percentile(spearman, 2.5)),
            "spearman_hi": float(np.percentile(spearman, 97.5)),
            "majority_agree_mean": float(majority.mean()),
            "majority_agree_lo": float(np.percentile(majority, 2.5)),
            "majority_agree_hi": float(np.percentile(majority, 97.5)),
            "spearman_brown_full": 2 * mean_r / (1 + mean_r),
            "sb_theoretical_half": k_half * alpha / (1 + (k_half - 1) * alpha),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV.name}: {len(out)} rows over {N_SPLITS} splits")
    print()

    print("=== SPLIT-HALF RELIABILITY (mean over splits, [2.5, 97.5] pct) ===")
    for _, r in out.iterrows():
        print(f"  {r['question_key']:15s} {r['stratum']:13s} "
              f"pearson={r['pearson_mean']:.3f} [{r['pearson_lo']:.3f},{r['pearson_hi']:.3f}]  "
              f"spearman={r['spearman_mean']:.3f}  "
              f"majority-agree={r['majority_agree_mean']:.3f}  "
              f"SB-full={r['spearman_brown_full']:.3f}  "
              f"(theory half-sample r≈{r['sb_theoretical_half']:.3f}, "
              f"k/half={r['mean_ratings_per_half']:.1f})")


if __name__ == "__main__":
    main()
