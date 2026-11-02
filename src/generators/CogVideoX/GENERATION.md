# CogVideoX Generation

How we generated the CogVideoX videos of SynthForensics (1,363 T2V and 1,363 I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/CogVideoX/`.

## Environment

Tested with Python 3.12, PyTorch 2.7.1 (CUDA 12.8), diffusers 0.37.0, and transformers 4.57.6. The scripts use the CogVideoX pipelines of diffusers. The upstream code is vendored in this folder at commit [`7a1af71`](https://github.com/zai-org/CogVideo/tree/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2). `environment.yml` pins every package of the environment we used.

```bash
conda env create -f environment.yml
conda activate cogvideox
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| CogVideoX-5B (T2V) | zai-org/CogVideoX-5b | [`8fc5b28`](https://huggingface.co/zai-org/CogVideoX-5b/tree/8fc5b281006c82b82d34fd2543d2f0ebb4e7e321) |
| CogVideoX-5B-I2V (I2V) | zai-org/CogVideoX-5b-I2V | [`a6f0f48`](https://huggingface.co/zai-org/CogVideoX-5b-I2V/tree/a6f0f4858a8395e7429d82493864ce92bf73af11) |

```bash
hf download zai-org/CogVideoX-5b \
    --revision 8fc5b281006c82b82d34fd2543d2f0ebb4e7e321 \
    --local-dir models/CogVideoX-5b
hf download zai-org/CogVideoX-5b-I2V \
    --revision a6f0f4858a8395e7429d82493864ce92bf73af11 \
    --local-dir models/CogVideoX-5b-I2V
```

Each repo is downloaded in full (transformer, VAE, T5 text encoder, tokenizer, scheduler). The metadata records them under their former names `THUDM/CogVideoX-5b` and `THUDM/CogVideoX-5b-I2V`, which redirect to `zai-org`. Expected layout (override with `--model-path`):

```
models/
├── CogVideoX-5b/
└── CogVideoX-5b-I2V/
```

The SHA-256 checksum of every weight file is stored in the `environment` field of each video's metadata.

## Inputs

From the SynthForensics dataset on Hugging Face:

- `captions/{id}.json`: structured caption of each source video (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame (I2V only)

## Prompt construction

The prompt joins 4 of the 8 caption fields into a single paragraph, in this order: `dense_description`, `style`, `shot_type`, `background`; the other fields are not used.

## Generation

```bash
# T2V
python generate_t2v.py --dataset-root /path/to/SynthForensics --gpu 0

# I2V
python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0
```

`--dataset-root` is the SynthForensics dataset folder: the scripts read `captions/` and `I2V/i2v_frames/` from it (override with `--captions-dir` and `--images-dir`). `--start-index` and `--end-index` split the sorted caption list across GPUs (e.g. `0`–`681` and `681`–`1363`). Videos whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

The prompt and seed of each released video are recorded in its JSON sidecar.

## Settings

| Parameter | Value |
|-----------|-------|
| Pipeline | diffusers `CogVideoXPipeline` (T2V), `CogVideoXImageToVideoPipeline` (I2V) |
| Scheduler | `CogVideoXDPMScheduler`, trailing timestep spacing |
| Inference steps | 100 |
| Guidance | CFG 6.0, dynamic CFG |
| Resolution (W x H) | 720 x 480 |
| Frames | 49 at 8 fps (about 6 s) |
| I2V conditioning | reference frame scaled to cover 720 x 480 and center-cropped (Lanczos) |
| Precision | bfloat16 |
| VAE | slicing and tiling enabled |

## Output

Videos are written to `outputs/cogvideox_t2v_raw/` and `outputs/cogvideox_i2v_raw/` (override with `--output-dir`), with the same file names as in the released dataset (`_i2v` for I2V):

- `{id}_cogvideox_t2v.mp4`: H.264 video, no audio
- `{id}_cogvideox_t2v.json`: prompt, negative prompt, seed, all hyperparameters, model revision and checksums, software versions, and GPU

In the released dataset they are under `videos/raw/cogvideox/` and `metadata/cogvideox/` of the `T2V/` and `I2V/` folders.
