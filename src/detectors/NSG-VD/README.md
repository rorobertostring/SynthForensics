# NSG-VD — SynthForensics Evaluation

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection (NeurIPS 2025).

## Environment

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml      # add -n <name> to use a name other than nsg-vd
conda activate nsg-vd
```

All packages are pinned to the versions used for the reported results, including `imageio-ffmpeg==0.6.0`, whose bundled ffmpeg is what `moviepy` uses to decode the videos.

## Overview

Two scripts in `SF_scripts/` run the SynthForensics evaluation:

| Script | Purpose |
|---|---|
| `SF_scripts/preprocess_synthforensics.py` | Extract frames and compute NSG features for every T2V/I2V generator × compression × real-dataset combination |
| `SF_scripts/test_synthforensics.py` | Run NSG-VD inference over all combinations and save per-combination CSVs |

Run preprocessing before inference. Caches are checked on every run, so re-runs skip already-processed videos.


## Prerequisites

**Checkpoints:**
```
ckpts/standard-Pika-mp.pth                    # NSG-VD checkpoint
checkpoints/256x256_diffusion_uncond.pt       # diffusion model used to compute the NSG features
```

`standard-Pika-mp.pth` is in the `ckpts/` folder of the [original repository](https://github.com/ZSHsh98/NSG-VD/tree/main/ckpts).

`256x256_diffusion_uncond.pt` can be downloaded from [OpenAI](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt).

**Dataset layout** — the pipeline expects the following structure, with `{compression}` in `raw`, `canonical`, `crf23`, `crf40`:

```
/path/to/SynthForensics/Data/
├── FF++/real_{compression}_test_set/*.mp4
├── DFD/real_{compression}_test_set/*.mp4
└── SynthForensics/
    ├── T2V/videos/{compression}/{generator}/*_t2v.mp4
    └── I2V/videos/{compression}/{generator}/*_i2v.mp4
```

**Hydra config** — edit `configs/synthforensics/test.yaml` to set dataset paths before running:

```yaml
synthforensics:
  t2v_videos_dir:  "/path/to/SynthForensics/Data/SynthForensics/T2V/videos"
  i2v_videos_dir:  "/path/to/SynthForensics/Data/SynthForensics/I2V/videos"
  real_dir:        "/path/to/SynthForensics/Data"
  splits_dir:      "."
  frames_cache_dir: "./data/SynthForensics/frames_cache"
  nsg_cache_dir:   "./data/SynthForensics/nsg_cache"
```

**SynthForensics split files** (already present in the NSG-VD root, read via `splits_dir: "."`):
```
SF-FF_fake_ids_140.txt
SF-DFD_fake_ids_363.txt
```

**Active generators** — edit the lists in `configs/synthforensics/test.yaml` to restrict which generators are evaluated:
```yaml
  t2v_generators:
    - cogvideox
    - ltx2-3
    - magi-1
    - self-forcing
    - skyreels-v2
    - wan2-1
    - daVinci-MagiHuman
    - helios

  i2v_generators:
    - cogvideox
    - ltx2-3
    - skyreels-v2
    - wan2-1
    - daVinci-MagiHuman
    - helios
    - magi-1
```


## Step 1 — Preprocessing

Run from the **NSG-VD root directory**:

```bash
python SF_scripts/preprocess_synthforensics.py
```

To run a subset (e.g. FF++ reals and T2V CogVideoX, raw only), override the lists on the command line:

```bash
python SF_scripts/preprocess_synthforensics.py 'synthforensics.compressions=[raw]' \
    'synthforensics.t2v_generators=[cogvideox]' 'synthforensics.i2v_generators=[]' \
    'synthforensics.real_datasets=[FF]'
```

### Arguments

The script is Hydra-based; all configuration comes from `configs/synthforensics/test.yaml`.
Any config key can be overridden on the command line with `key=value` (quote list values, as above).

| Argument | Default | Description |
|---|---|---|
| `--config-name` | `test.yaml` | Hydra config file under `configs/synthforensics/` |

### Output

```
data/SynthForensics/
├── frames_cache/
│   ├── real/{FF,DFD}/{compression}/
│   └── {T2V,I2V}/{generator}/{compression}/
└── nsg_cache/
    ├── real/{FF,DFD}/{compression}/
    └── {T2V,I2V}/{generator}/{compression}/{real_dataset}/
```

Real videos are extracted once per `(real_dataset, compression)` pair regardless of how many generators share that reference.


## Step 2 — Inference

Run from the **NSG-VD root directory**:

```bash
python SF_scripts/test_synthforensics.py
```

With a different checkpoint:

```bash
python SF_scripts/test_synthforensics.py ckpt_path=./ckpts/standard-SEINE-mp.pth
```

Subset (same overrides as preprocessing):

```bash
python SF_scripts/test_synthforensics.py 'synthforensics.compressions=[raw]' \
    'synthforensics.t2v_generators=[cogvideox]' 'synthforensics.i2v_generators=[]' \
    'synthforensics.real_datasets=[FF]'
```

### Arguments

Hydra-based; CLI overrides use `key=value` syntax:

| Argument | Default | Description |
|---|---|---|
| `--config-name` | `test.yaml` | Hydra config file under `configs/synthforensics/` |
| `ckpt_path=` | `./ckpts/standard-Pika-mp.pth` | Path to the pre-trained checkpoint |

### Output

```
results/test/synthforensics/
└── synthforensics_results_{T2V|I2V}_{FF|DFD}.csv
```

One CSV per `(video_type, real_dataset)` pair. Each row is one `video_type/generator/compression/real_dataset` combination. A final `Avg.` row is appended automatically.

### Metrics

Each row reports: `Recall`, `Accuracy`, `F1`, `AUROC`, `Precision` (values in %).
