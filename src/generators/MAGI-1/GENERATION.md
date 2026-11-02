# MAGI-1 Generation

How we generated the MAGI-1 videos of SynthForensics (1,363 T2V and 1,363 I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/MAGI-1/`.

## Environment

Tested with Python 3.10.12, PyTorch 2.8.0 (CUDA 12.8), and flash-attn 2.8.3. The upstream code is vendored in this folder at commit [`6ff822e`](https://github.com/SandAI-org/MAGI-1/tree/6ff822e74ded50611e81e1d0e115146b5c4dd2a5). `requirements.txt` pins the other packages of the environment we used.

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install sageattention==1.0.6
pip install flash-attn==2.8.3 --no-build-isolation
pip install flashinfer-python==0.6.6 --extra-index-url https://flashinfer.ai/whl/cu128/torch2.8/
```

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| MAGI-1 24B distill, VAE, T5 v1.1 XXL text encoder | sand-ai/MAGI-1 | [`cd0a465`](https://huggingface.co/sand-ai/MAGI-1/tree/cd0a46589f3279a3c5c579f5f3c693a6ad36d0d4) |

```bash
hf download sand-ai/MAGI-1 --revision cd0a46589f3279a3c5c579f5f3c693a6ad36d0d4 \
    --include "ckpt/magi/24B_distill/*" --local-dir weights
hf download sand-ai/MAGI-1 --revision cd0a46589f3279a3c5c579f5f3c693a6ad36d0d4 \
    --include "ckpt/vae/*" --local-dir weights
hf download sand-ai/MAGI-1 --revision cd0a46589f3279a3c5c579f5f3c693a6ad36d0d4 \
    --include "ckpt/t5/t5-v1_1-xxl/*" --exclude "ckpt/t5/t5-v1_1-xxl/t5-v1_1-xxl/*" --local-dir weights
```

Expected layout:

```
weights/ckpt/
├── magi/24B_distill/inference_weight.distill/
├── t5/t5-v1_1-xxl/
└── vae/
```

The SHA-256 checksum of every weight file is stored in the `environment` field of each video's metadata.

## Inputs

From the SynthForensics dataset on Hugging Face:

- `captions/{id}.json`: structured caption of each source video (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame (I2V only)

## Prompt construction

The prompt joins the 8 caption fields with ". ", in this order: `short_description`, `dense_description`, `background`, `style`, `shot_type`, `lighting`, `atmosphere`, `camera_movement`. A field shorter than four words is written as "The {field name} is {value}", with the value in lowercase.

## Generation

```bash
# T2V
python generate.py --mode t2v --input_dir /path/to/SynthForensics/captions --gpu 0

# I2V
python generate.py --mode i2v --input_dir /path/to/SynthForensics/captions \
    --image_dir /path/to/SynthForensics/I2V/i2v_frames --gpu 0
```

`generate.py` relaunches itself with `torchrun` (one process on the GPU given by `--gpu`). `--start-index` and `--end-index` split the sorted caption list across GPUs (e.g. `0`–`681` and `681`–`1363`). Videos whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

Command-line options such as `--seed`, `--video_size_h`, `--video_size_w` and `--t5_device` override the values of the config file. Without `--seed` the seed of the config file is used; with `--seed -1` each video gets a new random seed. The seed of each video is stored in its metadata under `config.runtime_config.seed`.

## Settings

Values of `example/24B/24B_distill_config_1gpu.json`, the default `--config_file`. The values used for each video are in its metadata under `config`.

| Parameter | Value |
|-----------|-------|
| Model | MAGI-1 24B distill (`distill: true`, `shortcut_mode` `8,16,16`) |
| Resolution (W x H) | 1280 x 720; the T2V videos are 1280 x 720 or 720 x 1280. The size of each video is in its metadata under `config.runtime_config` |
| Frames | `num_frames` 96 at 24 fps (96 frames in T2V, 120 in I2V) |
| Steps | 16 |
| Chunks | `chunk_width` 6, `window_size` 4 |
| Guidance | `cfg_number` 1 (no classifier-free guidance) |
| Precision | bfloat16, no FP8 quantization |
| KV offload | enabled |
| I2V reference frame | cropped to the output aspect ratio when it differs by more than 1% |

## Output

Videos are written to `output/` (override with `--output_dir`), with the same file names as in the released dataset (`_i2v` for I2V):

- `{id}_magi-1_t2v.mp4`: H.264 video
- `{id}_magi-1_t2v.json`: prompt, seed and all config values, model revision and checksums, software versions, and GPU

In the released dataset they are under `videos/raw/magi-1/` and `metadata/magi-1/` of the `T2V/` and `I2V/` folders.
