# GenD — SynthForensics Evaluation

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

Deepfake Detection that Generalizes Across Benchmarks (WACV 2026).

## Overview

Three scripts in `SF_scripts/` cover the full SynthForensics pipeline:

| Script | Purpose |
|---|---|
| `extract_frames_gend.py` | Extract aligned face crops from evaluation videos into `preprocessed_gend/` |
| `extract_frames_training.py` | Extract aligned face crops for training/fine-tuning (multi-worker capable) |
| `test_gend.py` | Run GenD inference on pre-built real/fake frame-list txt pairs |

For **evaluation**: run Steps 1–2 (extraction), then Step 3 (inference).
For **training/fine-tuning**: run Step 4 (extraction), then Step 5 (`run_exp.py`).


## Prerequisites

**Split ID files** (already present in the GenD root):

```
SF-DFD_fake_ids_363.txt
SF-FF_fake_ids_140.txt
```

**Training split files**: `splits/train.json`, `splits/val.json` and `splits/test.json` at the root of this repository.

All scripts must be run from the **GenD root directory** so that `detector.py` and the RetinaFace ONNX model are found correctly.


## Step 1 — Extract Real Frames (Evaluation)

```bash
python SF_scripts/extract_frames_gend.py \
    --mode real \
    --real_dir /path/to/FF++/real_canonical_test_set \
    --output_dir preprocessed_gend/real_ffpp_canonical/real \
    --txt_out config/datasets/gend/real_ffpp_canonical.txt
```

Repeat for each `(dataset, compression)` combination — 2 datasets (`dfd`, `ffpp`) × 4 compressions (`canonical`, `crf23`, `crf40`, `raw`) = 8 runs total. The naming pattern for `--output_dir` is `preprocessed_gend/real_{ds}_{compression}/real`.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | — | `real` for real videos |
| `--real_dir` | — | Folder containing real `.mp4` files |
| `--output_dir` | — | Output directory (`<output_dir>/<video_id>/frame_XXXXXX.png`) |
| `--txt_out` | `<output_dir>.txt` | Path for the frame-list `.txt` consumed by `test_gend.py` |
| `--num_frames` | `32` | Uniformly-spaced frames to extract per video |
| `--target_size` | `256,256` | Face crop size as `W,H` |
| `--scale` | `1.3` | Landmark alignment scale factor |
| `--det_thres` | `0.1` | RetinaFace detection threshold |


## Step 2 — Extract Fake Frames (Evaluation)

```bash
python SF_scripts/extract_frames_gend.py \
    --mode fake \
    --fake_dir /path/to/SynthForensics/videos/canonical/cogvideox \
    --split_file SF-FF_fake_ids_140.txt \
    --output_dir preprocessed_gend/fake_SF-FF_canonical/cogvideox \
    --txt_out config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt
```

Repeat for each `(fake_dataset, compression, generator)` combination. The naming pattern is `preprocessed_gend/fake_{FAKE_DS}_{compression}/{generator}`.

Split file to use per fake dataset:

| Fake dataset | Split file |
|---|---|
| `SF-DFD` | `SF-DFD_fake_ids_363.txt` |
| `SF-FF` | `SF-FF_fake_ids_140.txt` |

### Additional arguments for `--mode fake`

| Argument | Default | Description |
|---|---|---|
| `--fake_dir` | — | Folder containing fake `.mp4` files |
| `--split_file` | — | TXT/JSON with video IDs to include |
| `--video_suffix` | auto | Explicit generator suffix to strip from filenames (e.g. `ltx_2-3_t2v` for the ltx2-3 generator). Auto-derived if not set. |

### path_list mode

For custom collections not following the standard `{id}_{generator}` naming:

```bash
python SF_scripts/extract_frames_gend.py \
    --mode path_list \
    --path_list_file /path/to/videos.json \
    --output_dir preprocessed_gend/custom/my_collection \
    --txt_out config/datasets/gend/custom/my_collection.txt
```

`videos.json` must be a JSON array of absolute video paths. Video IDs are derived from each path relative to the common prefix, with `/` replaced by `_` (guarantees uniqueness across subdirectories).

### Output

```
preprocessed_gend/
├── real_{ds}_{compression}/real/<video_id>/frame_XXXXXX.png
└── fake_{FAKE_DS}_{compression}/{generator}/<video_id>/frame_XXXXXX.png

config/datasets/gend/
├── real_{ds}_{compression}.txt
└── {FAKE_DS}/{compression}/fake_{generator}.txt
```

A stats JSON (`extraction_stats_real.json` or `extraction_stats_fake.json`) is written into each output directory. Already-processed videos are skipped on re-run (resume-safe).


## Step 3 — Inference

```bash
python SF_scripts/test_gend.py \
    --real_txt config/datasets/gend/real_ffpp_canonical.txt \
    --fake_txt config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt \
    --run_name SF-FF_canonical_cogvideox
```

Multiple fake sources can be passed at once:

```bash
python SF_scripts/test_gend.py \
    --real_txt config/datasets/gend/real_ffpp_canonical.txt \
    --fake_txt config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt \
               config/datasets/gend/SF-FF/canonical/fake_wan2-1.txt \
    --run_name SF-FF_canonical_multi
```

Alternative checkpoint:

```bash
python SF_scripts/test_gend.py \
    --real_txt config/datasets/gend/real_ffpp_canonical.txt \
    --fake_txt config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt \
    --run_name SF-FF_canonical_cogvideox_PE \
    --checkpoint yermandy/GenD_PE_L
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--real_txt` | — | One or more `.txt` files listing real frame paths |
| `--fake_txt` | — | One or more `.txt` files listing fake frame paths |
| `--run_name` | — | Run name (used as subfolder under `--run_dir`) |
| `--checkpoint` | `yermandy/GenD_CLIP_L_14` | HuggingFace repo ID or local `.ckpt` path |
| `--run_dir` | `runs/synthforensics` | Root directory for run outputs |
| `--batch_size` | `512` | Inference batch size |
| `--devices` | `[0]` | GPU device ids |
| `--num_workers` | `12` | DataLoader workers |
| `--precision` | `bf16-mixed` | PyTorch Lightning precision |
| `--backbone` | — | Backbone override for local `.ckpt` files: `CLIP_L_14` or `CLIP_B_32`. Not needed for HuggingFace checkpoints. |

**Available HuggingFace checkpoints:** `yermandy/GenD_CLIP_L_14`, `yermandy/GenD_PE_L`

### Output

```
runs/synthforensics/{run_name}/
└── metrics.csv
```

Labels are inferred from the path structure: a directory whose third-from-last component contains `real` receives label 0; all others receive label 1. If a run directory with the same name already exists under `runs/synthforensics/`, it is removed and the run is repeated.


## Step 4 — Extract Frames for Training

`extract_frames_training.py` has the same interface as `extract_frames_gend.py` but adds `--workers` for parallel GPU-accelerated extraction. Each worker process loads its own RetinaFace ONNX session.

Real videos:

```bash
python SF_scripts/extract_frames_training.py \
    --mode real \
    --real_dir /path/to/FF++_Full \
    --output_dir preprocessed_gend/training/real_ffpp/real \
    --workers 4
```

Fake videos (filtered by a split JSON):

```bash
python SF_scripts/extract_frames_training.py \
    --mode fake \
    --fake_dir /path/to/SynthForensics/videos/raw/cogvideox \
    --split_file /path/to/SynthForensics/splits/train.json \
    --output_dir preprocessed_gend/training/t2v/cogvideox \
    --workers 4
```

### Additional argument

| Argument | Default | Description |
|---|---|---|
| `--workers` | `1` | Parallel worker processes. Recommended: 2–4 for GPU preprocessing. |

### Training output structure

```
preprocessed_gend/training/
├── real_ffpp/real/<id>/frame_XXXXXX.png
├── real_dfd/real/<id>/frame_XXXXXX.png
├── t2v/{generator}/<id>/frame_XXXXXX.png
└── i2v/{generator}/<id>/frame_XXXXXX.png
```


## Step 5 — Training / Fine-tuning

`run_exp.py` reads the frame lists in `config/datasets/gend/training/{train,val,test}/*.txt`: for each split, one `.txt` per source, listing the PNG paths of the videos of that split (`splits/*.json`). The file name must end with the generator name as written in `_IN_DOMAIN` / `_OUT_DOMAIN` of `src/exp/synthforensics.py` (e.g. `t2v_cogvideox.txt`, `i2v_daVinci.txt`): `sf-scratch` uses it to separate the In-Domain and Out-of-Domain generators. `extract_frames_training.py` writes one list per source (`<output_dir>.txt`); split it by video ID into these files.

Then launch an experiment via `run_exp.py` from the GenD root:

```bash
python run_exp.py sf-finetune
```

```bash
python run_exp.py sf-scratch
```

| Experiment | Description |
|---|---|
| `sf-finetune` | Fine-tune LayerNorm layers starting from the `GenD_CLIP_L_14` checkpoint |
| `sf-scratch` | Full training from CLIP-L/14 pre-trained initialisation (no GenD weights) |

Experiments are defined in `src/exp/synthforensics.py`.
