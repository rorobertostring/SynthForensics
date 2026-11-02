# Face Quality

Reproduction code for the **Face Quality** comparative analysis (paper **Section 4.2**).

The analysis quantifies the perceptual quality of detected faces in each video using **TOPIQ-NR-Face** (CGFIQA-trained no-reference face image quality assessment), and additionally compares the per-video score distribution of every benchmark against the FF++/DFD real-video baselines via 1D Wasserstein distance.

- **$Q$** — mean TOPIQ-NR-Face score across the 32 sampled frames per video; high values indicate sharp, undistorted faces independent of authenticity.
- **$W(\cdot, \mathrm{FF\text{++}})$, $W(\cdot, \mathrm{DFD})$** — 1D Wasserstein distance between the per-video $Q$ distribution of a benchmark and the corresponding real-video baseline (lower is closer in distribution).

## Pipeline

The analysis is a three-step pipeline:

1. **Per-video metrics** ([`face_quality.py`](face_quality.py)) — extracts 32 equidistant frames from each video, runs TOPIQ-NR-Face on each frame, and writes one CSV row per video with `mean_score` (= $Q$), `faces_detected`, and a few diagnostic columns.
2. **Bootstrap aggregation** ([`bootstrap_face_quality.py`](bootstrap_face_quality.py)) — restricts each per-benchmark CSV to the face-positive subset (`faces_detected > 0`), then computes per-benchmark mean $Q$ and 95% percentile bootstrap CI ($B=1{,}000$ resamples, chunked numpy).
3. **Wasserstein distance** ([`wasserstein_face_quality.py`](wasserstein_face_quality.py)) — computes 1D Wasserstein distance between every benchmark's per-video $Q$ distribution and the FF++/DFD baselines.

Together, these scripts reproduce the values reported in **Section 4.2** of the paper and the per-benchmark table of **Appendix C.3**.

## Requirements

- Python 3.10+
- `pyiqa` 0.1.15.post2 (provides the TOPIQ-NR-Face implementation; weights downloaded on first run)
- `scipy` (for `wasserstein_distance`)
- `decord` for fast video frame extraction
- `numpy`, `torch` (CUDA strongly recommended for Step 1)

```bash
pip install pyiqa==0.1.15.post2 scipy decord numpy torch
```

## Step 1 — Per-Video Metrics

Run on each benchmark in turn, pointing `--video-dir` at the benchmark's video corpus and `--dataset-name` at a unique slug used for the output CSV file name.

```bash
# GPU (recommended): single worker, model loaded once
python face_quality.py \
    --video-dir /path/to/<benchmark>/videos \
    --dataset-name <slug> \
    --device cuda \
    --gpu-id 0 \
    --recursive

# CPU multi-worker (for very large corpora when a GPU is not available)
python face_quality.py \
    --video-dir /path/to/<benchmark>/videos \
    --dataset-name <slug> \
    --device cpu \
    --workers 16 \
    --recursive
```

Each row in the output CSV contains the per-video `mean_score` (= $Q$) plus diagnostic columns. Videos for which the upstream face detector returns no face on every sampled frame are written with `faces_detected = 0` and excluded from the per-benchmark aggregate by Steps 2 and 3.

The CSVs in [`bootstrap_face_quality.py`](bootstrap_face_quality.py) and [`wasserstein_face_quality.py`](wasserstein_face_quality.py) (`BENCHMARKS`, `REALS`, `SF_FILES`) name one CSV per benchmark / per real-video baseline / per SynthForensics generator–modality pair; give the Step 1 outputs these file names so that the aggregation scripts find their inputs.

Files named `<slug>_wo_duplicates.csv` are the Step 1 CSV of the same benchmark with byte-identical (MD5) duplicate videos removed, keeping one copy of each.

## Step 2 — Bootstrap Aggregation (Mean $Q$)

Update the `ROOT` variable at the top of [`bootstrap_face_quality.py`](bootstrap_face_quality.py) to point at the directory holding the per-video CSVs from Step 1, then run:

```bash
python bootstrap_face_quality.py
```

The script restricts each entity to its face-positive subset, computes mean $Q$ and 95% percentile bootstrap CI, and prints a per-entity table that matches the values reported in **Section 4.2** and **Appendix C.3**.

## Step 3 — Wasserstein Distance

Update the `ROOT` variable at the top of [`wasserstein_face_quality.py`](wasserstein_face_quality.py) to point at the same directory, then run:

```bash
python wasserstein_face_quality.py
```

The script reports $W(\cdot, \mathrm{FF\text{++}})$ and $W(\cdot, \mathrm{DFD})$ for every benchmark, plus the pairwise SF $\leftrightarrow$ FF++ $\leftrightarrow$ DFD distance, matching the values in **Appendix C.3**.

## Notes

- **Frame sampling.** 32 equidistant frames per video, capped at `total_frames` for shorter clips.
- **Face-positive subset.** TOPIQ-NR-Face is undefined on frames with no detected face; videos without a single detected face across all sampled frames are excluded from the per-benchmark aggregate (`faces_detected = 0` rows in the CSVs).
- **Score domain.** The metric is bounded in $[0, 1]$; on rare inputs, numerical noise from the network can push the predicted score slightly negative, in which case the row is filtered out by both Step 2 and Step 3.
- **Bootstrap.** $B = 1{,}000$ resamples, fixed seed `42`, chunked numpy implementation that keeps memory bounded for large benchmarks (GenVideo, AIGVDBench).
- **Reproducibility.** All randomness is seeded via the script-level `SEED` constant; on the same per-video CSVs the bootstrap output is bit-for-bit reproducible up to numpy/BLAS nondeterminism.
