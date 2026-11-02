# FakeSTormer — SynthForensics Evaluation

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

Spatiotemporal deepfake detection with FakeSTormer (FakeSFormer).

## Environment

Run from the **FakeSTormer root directory**:

```bash
bash SF_scripts/setup_env.sh
conda activate fakestormer
```

The script creates the `fakestormer` environment from `environment.yml`, then installs the packages that cannot go in the YAML file:

- `torch==1.8.0+cu111` / `torchvision==0.9.0+cu111` wheels (conda resolves `pytorch==1.8.0` to a CPU build);
- `retinaface-pytorch==0.0.8` with `--no-deps`, since its metadata pins `torch==1.9.0`. Version 0.0.8 is required to reproduce the face crops: 0.0.7 computes slightly different bounding boxes;
- `mmcv-full==1.6.1` from the openmmlab index.

The CUDA 11.1 build runs on GPUs up to the Ampere generation (e.g. A100); it does not support Blackwell GPUs.

## Overview

Four scripts in `SF_scripts/` cover the full SynthForensics evaluation pipeline:

| Script | Purpose |
|---|---|
| `preprocess_synthforensics.py` | Extract face crops (RetinaFace) from real and fake videos |
| `eval_synthforensics.py` | Run FakeSTormer inference over all generator/compression/dataset combinations |
| `run_pipeline.py` | Orchestrator: runs preprocessing → eval sequentially for every combination |
| `infer_video.py` | Single-video inference: returns a REAL/FAKE verdict |

Standard order: preprocessing first, then evaluation. `run_pipeline.py` does both in one call.


## Prerequisites

**Split ID files** (already present in `SF_scripts/`):

```
SF-FF_fake_ids_140.txt
SF-DFD_fake_ids_363.txt
```

**Base eval config** (already present in the FakeSTormer root):

```
configs/temporal/FakeSFormer_synthforensics_eval.yaml
```

**Weights** — the eval config loads the C23 checkpoint from the [official pre-trained weights](https://www.dropbox.com/scl/fo/elk2szqf0du4l6zm5job9/AAdVmNH--6ywHBZGNQJlR5o?rlkey=j8xesf2fu4ahxdw99w5ndrkb2&st=fe6drzpx&dl=0):

```
pretrained/TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth
```

**Data layout** — pass `--data_root` at runtime. The script expects:

```
{data_root}/
├── FF++/real_{compression}_test_set/*.mp4
├── DFD/real_{compression}_test_set/*.mp4
└── SynthForensics/
    ├── T2V/videos/{compression}/{generator}/*_t2v.mp4  (t2v fakes)
    └── I2V/videos/{compression}/{generator}/*_i2v.mp4  (i2v fakes)
```


## Step 1 — Face Extraction

Run from the **FakeSTormer root directory**.

### t2v (text-to-video)

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/preprocess_synthforensics.py \
    --data_root /path/to/Data \
    --output_root preprocessed
```

### i2v (image-to-video)

Real frames are reused from the t2v run via a symlink — no re-extraction needed:

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/preprocess_synthforensics.py \
    --mode i2v \
    --fake_videos_subdir SynthForensics/I2V/videos \
    --real_preprocessed_root preprocessed \
    --output_root preprocessed_i2v \
    --data_root /path/to/Data
```

### Single combination

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/preprocess_synthforensics.py \
    --dataset SF-FF --compression crf23 --generator cogvideox \
    --output_root preprocessed \
    --data_root /path/to/Data
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `t2v` | Generation mode: `t2v` or `i2v` |
| `--dataset` | `all` | `SF-FF`, `SF-DFD`, or `all` |
| `--compression` | `crf23` | `crf23`, `crf40`, `raw`, `canonical`, or `all` |
| `--generator` | auto | Process only this generator (default: all, auto-discovered) |
| `--output_root` | required | Root folder for extracted face crops |
| `--data_root` | required | Root of the SynthForensics Data directory |
| `--fake_videos_subdir` | `SynthForensics/T2V/videos` | Subdirectory for fake videos under `data_root` (`SynthForensics/I2V/videos` for i2v) |
| `--real_preprocessed_root` | — | (i2v) Path to t2v real frames; creates a symlink instead of re-extracting |
| `--fake_only` | off | Skip real video processing |
| `--real_only` | off | Skip fake video processing |
| `--num_frames` | `32` | Face frames to extract per video |
| `--gpu` | `0` | GPU index for RetinaFace |

### path_list mode (explicit video list)

To preprocess a custom set of videos, pass a JSON file containing a flat list of absolute video paths:

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/preprocess_synthforensics.py \
    --path_list_file /path/to/dataset_split.json \
    --output_dir preprocessed/fake/MyDataset \
    --num_frames 32 \
    --gpu 0
```

Frames are saved to `{output_dir}/{video_id}/*.png`.

### Output

```
preprocessed/
└── {SF-FF,SF-DFD}/
    └── {compression}/
        └── test/videos/
            ├── real/{video_id}/*.png
            └── {generator}/{video_id}/*.png
```

Each completed subset writes a sentinel `extraction_stats_{type}.json`. Already-processed subsets are skipped automatically.


## Step 2 — Evaluation

Run from the **FakeSTormer root directory**:

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/eval_synthforensics.py \
    --preprocessed_root preprocessed \
    --results_root results
```

### i2v

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/eval_synthforensics.py \
    --mode i2v \
    --preprocessed_root preprocessed_i2v \
    --results_root results_i2v
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `t2v` | `t2v` or `i2v`; sets path defaults automatically |
| `--preprocessed_root` | `preprocessed` | Root of extracted face crops |
| `--results_root` | `results` | Root folder for result JSONs |
| `--base_cfg` | `configs/temporal/FakeSFormer_synthforensics_eval.yaml` | Base YAML config |
| `--datasets` | both | One or both of `SF-FF`, `SF-DFD` |
| `--compressions` | all four | One or more of `crf23`, `crf40`, `raw`, `canonical` |
| `--generators` | auto | One or more generator names (default: auto-discover) |
| `--combined` | off | Also run real-vs-all-generators and save as `ALL.json` |
| `--skip_existing` | off | Skip a combination whose JSON result already exists |
| `--batch_size` | `8` | Batch size for `scripts/test.py` (the reported results use 8; other values change the scores slightly) |
| `--num_workers` | `8` | DataLoader workers |

### Output

```
results/
└── {SF-FF,SF-DFD}/
    └── {compression}/
        └── {generator}.json     (metrics per combination)
```

### Metrics

Each JSON contains: `acc`, `auc`, `ap`, `ar`, `mf1`, `best_thr` (APR mode) or `acc`, `bacc`, `auc`, `p`, `r`, `s`, `f1`, `eer`, `best_thr` (standard mode), depending on the config setting.


## Combined Pipeline

`run_pipeline.py` runs preprocessing and evaluation back-to-back for every combination. It is equivalent to running the two scripts separately.

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/run_pipeline.py \
    --data_root /path/to/Data \
    --output_root preprocessed \
    --results_root results
```

To run only one phase:

```bash
python SF_scripts/run_pipeline.py ... --preprocess_only
python SF_scripts/run_pipeline.py ... --eval_only
```

### Arguments

It takes the main options of the two child scripts, with `--dataset`, `--compression` and `--generator` selecting one value each (see `--help` for the full list), plus:

| Argument | Default | Description |
|---|---|---|
| `--preprocess_only` | off | Run only face extraction, skip eval |
| `--eval_only` | off | Run only eval, skip face extraction |
| `--gpu_preprocess` | `0` | GPU index passed to `preprocess_synthforensics.py` |

For the combined real-vs-all-generators result (`ALL.json`), run `eval_synthforensics.py --combined`.

A log is written to `{output_root}/pipeline_{mode}.log`. Combinations that fail preprocessing are skipped (with a warning) rather than aborting the whole run.


## Single-Video Inference

Run from the **FakeSTormer root directory**:

```bash
CUDA_VISIBLE_DEVICES=0 python SF_scripts/infer_video.py \
    --cfg configs/temporal/FakeSFormer_base_c23.yaml \
    --video path/to/video.mp4
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--cfg` | required | Config YAML (must specify `TEST.pretrained`) |
| `--video` | required | Path to the input video |
| `--num_frames` | `32` | Face frames to extract |
| `--padding` | `0.25` | RetinaFace bbox padding fraction |
| `--gpu` | `0` | GPU index |
| `--threshold` | from config | Override the decision threshold |

### Output

```
==================================================
  Video   : video.mp4
  Score   : 0.8123  (threshold: 0.5)
  Verdict : FAKE
  Clips   : [0.7812, 0.8434, ...]
==================================================
```

The video is split into non-overlapping 4-frame clips; the final score is the mean over all clips.
