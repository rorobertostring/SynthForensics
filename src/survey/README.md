# Survey

This folder contains the data and the analysis code of the perceptual survey conducted as part of the SynthForensics evaluation. The survey was designed to assess human ability to distinguish AI-generated (synthetic) videos from real or competitor-produced ones, across multiple deepfake detection benchmarks. The web platform used to collect the responses is not included; the full protocol is described in Appendix C.4 of the paper.

## Structure

```
survey/
└── Experiments/      # Raw anonymous survey export and statistical analysis of the results
```

## Survey Design

Participants were shown pairs of videos and asked three questions per pair:

- **Q1 (better_made)** — which video looks better produced?
- **Q2 (more_realistic)** — which video looks more realistic?
- **Q3 (could_be_fake)** — which video(s), if any, could be AI-generated?

Pairs contrasted a synthetic video produced by a SynthForensics generator against either a real video or a competitor-generated one (`vs_type ∈ {competitor, real}`).

## Experiments

The `Experiments/` folder contains the raw survey export and a numbered series of Python analysis scripts that reproduce all statistical results reported in the paper.

| Script | Description |
|---|---|
| `01_prepare_data.py` | Parses the raw CSV export, derives benchmark / modality / generator columns, outputs `prepared.parquet` |
| `02_screen_subjects.py` | Two-stage participant screening (completion check + response-variance filter) |
| `03_winrate.py` | Per-benchmark win-rate analysis with cluster-bootstrap 95 % CI for Q1 and Q2 |
| `04_q3_detection.py` | Human detection rate for Q3 via binary decomposition; bootstrap CIs |
| `05_agreement.py` | Inter-rater agreement (Krippendorff's α) per question and subgroup |
| `06_demographics.py` | Demographic breakdown of initial vs. kept participants |
| `07_per_sf_breakdown.py` | Q1/Q2 win rates and Q3 fake-flag rates per SynthForensics generator and modality |
| `08_sensitivity_subgroup.py` | Sensitivity analysis across subgroups (AI background, deepfake awareness) |
| `09_sensitivity_pearson_trim.py` | Sensitivity analysis with Pearson trimming |
| `10_split_half_reliability.py` | Split-half reliability of the per-pair aggregates over random participant partitions |

The raw data file is `Experiments/survey_export_2026-05-01.csv` (13 575 responses, 159 participants across 110 video pairs).

Scripts must be run in order starting from `01_prepare_data.py`, which produces the `prepared.parquet` file consumed by all subsequent scripts.
