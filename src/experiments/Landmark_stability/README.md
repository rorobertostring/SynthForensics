# Landmark Stability

Reproduction code for the **Landmark Stability** comparative analysis (paper **Section 4.1**).

The analysis quantifies how reliably a 68-point face-alignment network (FAN, 2DFAN4) localizes facial landmarks on each video. From the per-frame heatmaps it derives two video-level metrics:

- **$\bar{a}$** — mean per-landmark heatmap-peak activation across the 68 landmarks and the 32 sampled frames; high values indicate sharp, unambiguous landmark localization.
- **$\mathrm{Comp}_\tau$** — landmark completeness, i.e., the fraction of frames with a detected face on which all 68 heatmap peaks reach threshold $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$.

## Pipeline

The analysis is a two-step pipeline:

1. **Per-video metrics** ([`landmark_quality.py`](landmark_quality.py)) — extracts 32 equidistant frames from each video, runs FAN, and writes one CSV row per video with `mean_confidence` (= $\bar{a}$), `completeness_th_0.{1..9}`, `faces_detected`, and a few diagnostic columns.
2. **Bootstrap aggregation** ([`bootstrap_landmark_stability.py`](bootstrap_landmark_stability.py)) — restricts each per-benchmark CSV to the face-positive subset (`faces_detected > 0`), then computes per-benchmark mean and 95% percentile bootstrap CI ($B=1{,}000$ resamples, chunked numpy) for $\bar{a}$ and $\mathrm{Comp}_\tau$. Reports the per-benchmark values shown in the landmark plot of Section 4.1 and in the per-benchmark table of Appendix C.2.

## Requirements

- Python 3.10+
- `face-alignment` (FAN 2DFAN4 weights are downloaded on first run)
- `decord` for fast video frame extraction
- `numpy`, `torch` (a CUDA GPU is required)

```bash
pip install face-alignment decord numpy torch
```

## Step 1 — Per-Video Metrics

Run on each benchmark in turn, pointing `--video-dir` at the benchmark's video corpus and `--dataset-name` at a unique slug used for the output CSV file name.

```bash
python landmark_quality.py \
    --video-dir /path/to/<benchmark>/videos \
    --dataset-name <slug> \
    --gpu-id 0 \
    --recursive
```

By default, results are written to `results_32/<slug>.csv` (one row per video, with the per-threshold completeness columns plus `mean_confidence`).

For very large corpora, the script supports sharding via `--start-index` / `--end-index`, producing `results_32/<slug>_<start>_<end>.csv` files that can be concatenated.

The CSVs in [`bootstrap_landmark_stability.py`](bootstrap_landmark_stability.py) (`BENCHMARKS`, `REALS`, `SF_FILES`) name one CSV per benchmark / per real-video baseline / per SynthForensics generator–modality pair; give the Step 1 outputs these file names so that the bootstrap finds its inputs.

Files named `<slug>_wo_duplicates.csv` are the Step 1 CSV of the same benchmark with byte-identical (MD5) duplicate videos removed, keeping one copy of each.

## Step 2 — Bootstrap Aggregation

Update the `ROOT` variable at the top of [`bootstrap_landmark_stability.py`](bootstrap_landmark_stability.py) to point at the directory holding the per-video CSVs from Step 1, then run:

```bash
python bootstrap_landmark_stability.py
```

The script restricts each entity to its face-positive subset, computes mean and 95% percentile bootstrap CI for $\bar{a}$ and $\mathrm{Comp}_\tau$ (with $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$), and prints a per-entity table that matches the values reported in **Section 4.1** of the paper and the per-benchmark table of **Appendix C.2**.

## Notes

- **Frame sampling.** 32 equidistant frames per video, capped at `total_frames` for shorter clips.
- **Face-positive subset.** Per-video metrics are undefined when FAN fails to detect a face on every sampled frame; such videos are excluded from the per-benchmark aggregate (`faces_detected = 0` rows in the CSVs).
- **Bootstrap.** $B = 1{,}000$ resamples, fixed seed `42`, chunked numpy implementation that keeps memory bounded for large benchmarks (GenVideo, AIGVDBench).
- **Reproducibility.** All randomness is seeded via the script-level `SEED` constant; on the same per-video CSVs the bootstrap output is bit-for-bit reproducible up to numpy/BLAS nondeterminism.
