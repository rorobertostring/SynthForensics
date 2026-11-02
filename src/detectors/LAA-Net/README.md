# LAA-Net — SynthForensics Evaluation

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection (CVPR 2024).

## Environment

Create the conda environment from `environment.yml`, then install RetinaFace without dependencies (its metadata pins `torch==1.9.0`, which would replace torch 2.8.0):

```bash
conda env create -f environment.yml      # add -n <name> to use a name other than laa_net
conda activate laa_net
pip install --no-deps retinaface-pytorch==0.0.8
```

Version 0.0.8 is required to reproduce the face crops: 0.0.7 computes slightly different bounding boxes.

## Overview

Three scripts in `scripts/` run the full SynthForensics evaluation pipeline:

| Script | Purpose |
|---|---|
| `extract_frames_custom.py` | Extract and crop faces from raw videos into `preprocessed/` |
| `test.py` | Run LAA-Net inference over preprocessed frame directories |
| `train.py` | Train a LAA-Net model from a config file |

For evaluation, run them in order: extraction first, then inference.


## Prerequisites

**Split ID files** (already present in the LAA-Net root):

```
SF-DFD_fake_ids_363.txt
SF-FF_fake_ids_140.txt
```

**Weights** — `configs/test_custom.yaml` loads the SBI checkpoint (used for the reported results) from the [official pre-trained weights](https://www.dropbox.com/scl/fo/dzmldaytujdeuky69d5x1/AIJrH2mit1hxnl1qzavM3vk?rlkey=nzzliincrfwejw2yr0ovldru1&st=z8ds7il7&dl=0):

```
pretrained_weights/PoseEfficientNet_EFN_hm10_EFPN_NoBasedCLS_Focal_C3_256Cst100_8SBI_SAM(Adam)_ADV_Era1_OutSigmoid_1e7_boost500_UnFZ_model_best.pth
```

**Dataset layout expected by the extraction script:**

```
<DATA_ROOT>/
├── DFD/real_{compression}_test_set/     # real DFD videos
├── FF++/real_{compression}_test_set/    # real FF++ videos
└── SynthForensics/
    ├── T2V/videos/{compression}/{generator}/*_t2v.mp4   # t2v fake videos
    └── I2V/videos/{compression}/{generator}/*_i2v.mp4   # i2v fake videos
```

Compressions: `canonical`, `crf23`, `crf40`, `raw`


## Step 1 — Frame Extraction

Run from the **LAA-Net root directory**. The script uses RetinaFace to detect and crop faces, saving 256×256 PNG crops.

> **GPU note:** the face crops used for the reported results were extracted on an NVIDIA RTX PRO 6000 (Blackwell). On other GPU architectures RetinaFace can shift some crops by ±1 pixel, which slightly changes the scores.

### Extract real videos

```bash
python scripts/extract_frames_custom.py \
    --mode real \
    --real_dir /path/to/FF++/real_canonical_test_set \
    --output_dir preprocessed/real/ff/canonical
```

### Extract fake videos

```bash
python scripts/extract_frames_custom.py \
    --mode fake \
    --fake_dir /path/to/SynthForensics/T2V/videos/canonical/cogvideox \
    --split_file SF-FF_fake_ids_140.txt \
    --output_dir preprocessed/fake/SF-FF/canonical/cogvideox
```

### Extract both at once

```bash
python scripts/extract_frames_custom.py \
    --mode both \
    --real_dir /path/to/real_videos \
    --fake_dir /path/to/fake_videos \
    --split_file SF-FF_fake_ids_140.txt \
    --output_dir preprocessed/both_output
```

### Extract from an explicit video list

```bash
python scripts/extract_frames_custom.py \
    --mode path_list \
    --path_list_file /path/to/video_list.json \
    --output_dir preprocessed/custom_output
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `both` | Processing mode: `real`, `fake`, `both`, or `path_list` |
| `--real_dir` | — | Folder of real videos (required for `real` / `both`) |
| `--fake_dir` | — | Folder of fake videos (required for `fake` / `both`) |
| `--output_dir` | — | Root directory for PNG face crops (required) |
| `--split_file` | — | ID file (TXT or JSON) to filter fake videos (optional for `fake` / `both`) |
| `--path_list_file` | — | JSON file with absolute video paths (required for `path_list`) |
| `--num_frames` | `32` | Number of uniformly sampled frames per video |
| `--padding` | `0.25` | Face bounding-box padding ratio |

### Output

```
preprocessed/
├── real/<dataset>/<compression>/<video_id>/*.png
├── fake/<fake_dataset>/<compression>/<generator>/<video_id>/*.png   ← t2v
└── fake-i2v/<fake_dataset>/<compression>/<generator>/<video_id>/*.png  ← i2v

extraction_stats_real.json   (per-video face detection stats)
extraction_stats_fake.json
```

Already-processed videos are skipped automatically (safe to resume).


## Step 2 — Inference

Run from the **LAA-Net root directory**. The standard config for SynthForensics evaluation is `configs/test_custom.yaml`.

### Dataset evaluation

```bash
python scripts/test.py \
    --cfg configs/test_custom.yaml \
    --real_root preprocessed/real/ff/canonical \
    --fake_root preprocessed/fake/SF-FF/canonical/cogvideox \
    --output_path results/SF-FF/canonical/cogvideox.json
```

### Single-image test

Set `TEST.subtask: test_img` in the config, then:

```bash
python scripts/test.py \
    --cfg configs/resnet_fpn_hm.yaml \
    --image /path/to/face_crop.png
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--cfg` / `-c` | — | Config file path (required) |
| `--real_root` | — | Override `DATASET.DATA.TEST.REAL_ROOT` from the config |
| `--fake_root` | — | Override `DATASET.DATA.TEST.FAKE_ROOT` from the config |
| `--output_path` | — | Path to save JSON results (e.g. `results/SF-FF/canonical/cogvideox.json`) |
| `--image` / `-i` | — | Single image path, used when the config sets `TEST.subtask: test_img` |

**Available configs:**

| Config | Backbone | Training type |
|---|---|---|
| `resnet_fpn_hm.yaml` | ResNet | Heatmap |
| `efn4_fpn_hm.yaml` | EfficientNet-B4 | Heatmap |
| `efn4_fpn_hm_adv.yaml` | EfficientNet-B4 | Heatmap + adversarial |
| `efn4_fpn_sbi_adv.yaml` | EfficientNet-B4 | SBI + adversarial |
| `resnet_fpn_sbi.yaml` | ResNet | SBI |
| `hrnet_sbi.yaml` | HRNet | SBI |
| `xception_sbi.yaml` | Xception | SBI |
| `test_custom.yaml` | — | SynthForensics evaluation (no training data) |

### Output

```
results/
└── SF-{DFD,FF}/{compression}/{generator}.json   (path given by --output_path)
```

Each JSON file contains:

```json
{
  "real_root": "...",
  "fake_root": "...",
  "pretrained": "...",
  "timestamp": "...",
  "n_videos": ...,
  "metrics": {
    "auc": ...,
    "ap": ...,
    "ar": ...,
    "precision": ...,
    "eer": ...,
    "mf1": ...,
    "acc": ...
  }
}
```


## Step 3 (optional) — Training

Run from the **LAA-Net root directory**:

```bash
python scripts/train.py --cfg configs/efn4_fpn_hm_adv.yaml
```

With GPU memory pre-allocation:

```bash
python scripts/train.py --cfg configs/efn4_fpn_hm_adv.yaml --alloc_mem
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--cfg` | — | Config file path (required) |
| `--alloc_mem` / `-a` | off | Pre-allocate GPU memory before training |

The best checkpoint is saved to `logs/<model>_<task>_model_best.pth` and training curves are written to `.tensorboard/`.
