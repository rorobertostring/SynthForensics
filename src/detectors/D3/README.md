# D3 — SynthForensics Evaluation

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

Training-Free AI-Generated Video Detection Using Second-Order Features.


## Environment

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml      # add -n <name> to use a name other than d3
conda activate d3
```

`ffmpeg` is pinned to the version used for the reported results (8.0.1): it decodes the clips used by D3, so other versions may produce different frames.


## Overview

Two scripts in `SF_scripts/` run the full SynthForensics evaluation pipeline:

| Script | Purpose |
|---|---|
| `sf_extract_frames.py` | Extract frames from raw videos into `preprocessing/frames/` |
| `sf_run_all.py` | Run D3 inference over the selected generator/compression/dataset combinations |

Run them in order: extraction first, then inference.


## Prerequisites

**Split ID files** (already present in the D3 root):

```
SF-DFD_fake_ids_363.txt
SF-FF_fake_ids_140.txt
```

**Dataset layout** — pass the roots on the command line. `REAL_ROOT` must contain only the **test set** real videos of each dataset:

```
<REAL_ROOT>/
├── DFD/real_{compression}_test_set/*.mp4
└── FF++/real_{compression}_test_set/*.mp4

<FAKE_ROOT>/                                   # the SynthForensics folder
├── T2V/videos/{compression}/{generator}/*_t2v.mp4
└── I2V/videos/{compression}/{generator}/*_i2v.mp4
```

Compressions: `raw`, `canonical`, `crf23`, `crf40`. Generators are discovered from the folder names.


## Step 1 — Frame Extraction

Run from the **D3 root directory**:

```bash
python SF_scripts/sf_extract_frames.py \
    --real-root /path/to/data \
    --fake-root /path/to/data/SynthForensics
```

Subset (e.g. FF++ reals and T2V CogVideoX, raw only):

```bash
python SF_scripts/sf_extract_frames.py \
    --real-root /path/to/data \
    --fake-root /path/to/data/SynthForensics \
    --real-datasets FF++ --compressions raw --modalities t2v --generators cogvideox
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--real-root` | required | Folder containing `DFD/`, `FF++/` with `real_{compression}_test_set/` |
| `--fake-root` | required | SynthForensics folder containing `T2V/` and `I2V/` |
| `--split-dir` | D3 root | Folder with the `SF-*_fake_ids_*.txt` split files |
| `--output-dir` | `preprocessing` | Output root (`frames/` and `metadata/` are created inside) |
| `--real-datasets` | all | One or more of `DFD`, `FF++` |
| `--compressions` | all | One or more of `raw`, `canonical`, `crf23`, `crf40` |
| `--modalities` | all | One or more of `t2v`, `i2v` |
| `--generators` | all found | Generator folders to process |
| `--skip-real` / `--skip-fake` | off | Do not extract real / fake videos |
| `--all-videos` | off | Extract every fake video, not only the test split |
| `--ffmpeg` | `ffmpeg` | ffmpeg executable |
| `--workers` | `32` | Parallel workers |
| `--face-detection` | off | Optional: run a Haar-cascade face check on the extracted frames |

### Output

```
preprocessing/
├── frames/
│   ├── real/{DFD,FF++}/{compression}/{video_stem}/
│   └── fake/{t2v,i2v}/{compression}/{generator}/{video_stem}/
└── metadata/
    ├── extraction_log.csv          (one row per video: status, duration, fps, …)
    └── no_face_videos.csv          (only if --face-detection is set)
```

Extraction follows the original `utils/video2frame.py` settings: 8 fps, 3-second clip, same ffmpeg command line. Already-extracted videos are skipped.


## Step 2 — Inference

Run from the **D3 root directory**:

```bash
python SF_scripts/sf_run_all.py \
    --gpu-id 0 \
    --encoder XCLIP-16 \
    --loss l2
```

Subset (same filters as the extraction):

```bash
python SF_scripts/sf_run_all.py --gpu-id 0 --encoder XCLIP-16 --loss l2 \
    --real-datasets FF++ --compressions raw --modalities t2v --generators cogvideox
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--encoder` | `XCLIP-16` | Feature encoder (see choices below) |
| `--loss` | `l2` | Distance metric: `l2` or `cos` |
| `--gpu-id` | `0` | CUDA device index |
| `--seed` | `42` | Random seed |
| `--base-dir` | D3 root | D3 root: frames are read from `preprocessing/frames`, results are written to `results/` |
| `--split-dir` | `--base-dir` | Folder with the `SF-*_fake_ids_*.txt` split files |
| `--real-datasets` | all | One or more of `DFD`, `FF++` |
| `--compressions` | all | One or more of `raw`, `canonical`, `crf23`, `crf40` |
| `--modalities` | all | One or more of `t2v`, `i2v` |
| `--generators` | all found | Generator folders to evaluate (found in `preprocessing/frames/fake/`) |
| `--overwrite` | off | Recompute combinations whose JSON result already exists |

**Encoder choices:** `CLIP-16`, `CLIP-32`, `XCLIP-16`, `XCLIP-32`, `DINO-base`, `DINO-large`, `ResNet-18`, `VGG-16`, `EfficientNet-b4`, `MobileNet-v3`

### Output

```
results/{encoder}_{loss}/
├── summary.csv           (rebuilt from all runs/*.json)
└── runs/
    └── {real_ds}_{compression}_{modality}_{generator}.json
```

Combinations already present in `runs/` are reused unless `--overwrite` is given (safe to resume). Combinations with missing frames are reported and skipped without writing a JSON, so they are evaluated once their frames are extracted.

### Metrics

Each run reports: `auc` (paper metric), `ap_real`, `ap_fake`, `eer`, `accuracy`, `f1`, `precision`, `recall`, `tp/tn/fp/fn`, `n_real`, `n_fake`. `accuracy`, `f1`, `precision`, `recall` and `tp/tn/fp/fn` use the threshold that maximises accuracy on the evaluated set itself (oracle threshold); only `auc` is reported in the paper.
