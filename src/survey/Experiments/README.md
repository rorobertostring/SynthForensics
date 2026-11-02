# Survey Analysis Pipeline

Reproduction code for the **Human Study** comparative analysis (paper **Section 4.3** and Appendix C.4).

This folder is the analytical counterpart of the survey described in [`../README.md`](../README.md): given the raw export of the survey responses, it produces all the human-study numbers reported in the paper, from the headline win rates and Q3 fake-flag rates to the demographic breakdowns and the two sensitivity analyses.

## Inputs

A single raw CSV export from the survey backend:

```
survey_export_2026-05-01.csv     # 13,575 responses, 159 participants, 110 video pairs
```

The export is in long format, one row per session, pair and question, with 13 columns: `session_id`, four demographic fields (`language`, `age_range`, `ai_background`, `knows_deepfakes`), `pair_number`, category and path of the two videos (`video_a_category`, `video_a_path`, `video_b_category`, `video_b_path`), `question_key`, `chosen_video` and `chosen_category`.

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `pyarrow` (parquet I/O)
- `scipy` (bootstrap and statistical tests)
- `krippendorff` (Step 5 only)

```bash
pip install pandas numpy pyarrow scipy krippendorff
```

## Pipeline

The 10 scripts must be run **in order**. `01_prepare_data.py` produces the `prepared.parquet` file that all subsequent scripts consume; `02_screen_subjects.py` produces the `kept_sessions.csv` filter consumed by Steps 3–10.

```bash
python 01_prepare_data.py            # raw CSV -> prepared.parquet
python 02_screen_subjects.py         # two-stage participant screening
python 03_winrate.py                 # Q1/Q2 win rates with cluster-bootstrap CIs
python 04_q3_detection.py            # Q3 fake-flag rates (binary decomposition)
python 05_agreement.py               # Krippendorff's α per question and subgroup
python 06_demographics.py            # demographic breakdown of initial vs kept sets
python 07_per_sf_breakdown.py        # SF-side breakdown (per-generator, per-modality)
python 08_sensitivity_subgroup.py    # sensitivity across demographic subgroups
python 09_sensitivity_pearson_trim.py  # ITU-T P.910 Annex A.1 consensus-trim sensitivity
python 10_split_half_reliability.py  # split-half reliability of the per-pair aggregates
```

| Script | Purpose | Reproduces |
|---|---|---|
| `01_prepare_data.py` | Parses the raw CSV, derives `vs_type` / `is_pristine_pair` / `sf_generator` / `sf_modality` columns from the video paths, writes `prepared.parquet` | — |
| `02_screen_subjects.py` | Two-stage participant screening: Stage 1 drops sessions with completion rate &lt; 80%; Stage 2 drops sessions whose top-voted Q1+Q2 category accounts for ≥ 95% of responses (disengaged, non-discriminating responders) | Section 4.3 retention figures (159 → 118 kept) |
| `03_winrate.py` | Per-benchmark Q1 (better quality) and Q2 (more realistic) win rates with cluster-bootstrap 95% CIs over participants | Headline win rates of Section 4.3; per-benchmark table in Appendix C.4.3 |
| `04_q3_detection.py` | Q3 (could be a fake) per-benchmark fake-flag rate via binary decomposition `{A, B, Both, Neither}` → `(is_A_fake, is_B_fake)` | Headline Q3 rate of Section 4.3; per-benchmark Q3 table in Appendix C.4.3 |
| `05_agreement.py` | Inter-rater agreement (Krippendorff's α) per question (Q1, Q2, Q3), overall and stratified by `vs_type`, `ai_background`, `knows_deepfakes`, plus the chi-square test of the uniform-choice null on the Q2 responses of the SF-vs-real pairs | Appendix C.4 agreement table and chi-square test of Appendix C.4.4 |
| `06_demographics.py` | Demographic breakdown (age, AI background, deepfake awareness, language) for the four screening segments (initial / kept / rejected by Stage 1 / rejected by Stage 2) | Demographics table in Appendix C.4 |
| `07_per_sf_breakdown.py` | Re-aggregates Q1/Q2 win rates and Q3 fake-flag rates by SynthForensics generator and modality, pooled across the nine competitor benchmarks | Per-generator and per-modality tables of Appendix C.4.3 |
| `08_sensitivity_subgroup.py` | Robustness check: re-aggregates the headline SF-vs-competitor numbers within each demographic subgroup, with cluster-bootstrap CIs widening on small subgroups | Sensitivity table in Appendix C.4 |
| `09_sensitivity_pearson_trim.py` | Robustness check inspired by ITU-T P.910 Annex A.1: replaces our Stage-2 disengagement filter with a fixed-fraction Pearson-correlation consensus trim and recomputes the headline numbers | Pearson-trim sensitivity numbers in Appendix C.4 |
| `10_split_half_reliability.py` | Reliability of the per-pair aggregates over 1,000 random 59/59 partitions of the retained sessions, with per-pair rates recomputed in each half and correlated across pairs; reports Spearman-Brown full-sample reliability and pair-level verdict agreement | Split-half reliability table in Appendix C.4 |

## Outputs

Each script writes its outputs in this folder, next to the scripts:

- `prepared.parquet`, `kept_sessions.csv`, `screening_log.csv`
- `winrate_aggregate.csv`, `winrate_primary.csv`, `winrate_by_generator.csv`, `winrate_by_modality.csv`
- `q3_aggregate.csv`, `q3_primary.csv`, `q3_by_generator.csv`, `q3_by_modality.csv`
- `agreement.json`
- `demographics.csv`
- `winrate_per_generator.csv`, `winrate_per_modality.csv`, `q3_per_generator.csv`, `q3_per_modality.csv`
- `sensitivity_subgroup.csv`
- `sensitivity_pearson_trim.csv`, `sensitivity_pearson_trim_r1.csv`
- `split_half_reliability.csv`

## Notes

- **Bootstrap.** All confidence intervals reported in the paper come from cluster-bootstrap by `session_id`, $B = 1{,}000$ resamples, fixed seed `42`, percentile method (2.5 / 97.5). The bootstrap output is bit-for-bit reproducible on the same input parquet up to numpy/BLAS nondeterminism.
- **Q3 four-category response.** The survey UI exposes `{A, B, Both, Neither}`. `Neither` is encoded as `NaN` in the raw export and decoded as `(is_A_fake = 0, is_B_fake = 0)` by `04_q3_detection.py`; it is a valid 4th response option, **not** missing data.
- **vs_type derivation.** `competitor` covers the nine existing synthetic-video benchmarks; `real` covers the FF++ and DFD pristine baselines. `vs_type` is derived from `benchmark_name` by the scripts that split on it.
