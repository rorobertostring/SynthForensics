# Helios Generation

How we generated the Helios videos of SynthForensics (1,363 T2V and 1,363 I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/Helios/`.

## Environment

Tested with Python 3.11, PyTorch 2.11.0 (CUDA 13.0), diffusers 0.37.1, transformers 5.3.0, and flash-attn 2.8.3. The upstream code is vendored in this folder at commit [`5cb7ce9`](https://github.com/PKU-YuanGroup/Helios/tree/5cb7ce9f95ce7df336ef072b9c8105321d6db54a). `environment.yml` pins every package of the environment we used.

```bash
conda env create -f environment.yml
conda activate helios
pip install flash-attn==2.8.3 --no-build-isolation
```

At run time, the diffusers `flash_hub` attention backend downloads the FlashAttention-2 kernel from `kernels-community/flash-attn2` (branch `v1`).

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| Helios-Distilled 14B (transformer, text encoder, VAE, scheduler) | BestWishYsh/Helios-Distilled | [`1999182`](https://huggingface.co/BestWishYsh/Helios-Distilled/tree/1999182614cb08d3bdcc46b9827504af2914b87b) |

```bash
hf download BestWishYsh/Helios-Distilled \
    --revision 1999182614cb08d3bdcc46b9827504af2914b87b \
    --include "model_index.json" --include "scheduler/*" --include "text_encoder/*" \
    --include "tokenizer/*" --include "transformer/*" --include "vae/*" \
    --local-dir models/Helios-Distilled
```

Expected layout (override with `--model-path`):

```
models/
└── Helios-Distilled/
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── transformer/
    └── vae/
```

The SHA-256 checksums of the first transformer shard, the first text encoder shard, and the VAE are stored in the `environment` field of each video's metadata.

## Inputs

From the SynthForensics dataset on Hugging Face:

- `captions/{id}.json`: structured caption of each source video (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame (I2V only)

## Prompt construction

The prompt joins all 8 caption fields into a single paragraph, in this order: `style`, `shot_type`, `short_description`, `dense_description`, `background`, `lighting`, `atmosphere`, `camera_movement`. The values of `style`, `shot_type`, and `camera_movement` are lowercased and written as "The style is ...", "The shot type is ...", and "The camera movement is ...".

## Generation

```bash
# T2V
python generate_t2v.py --captions-dir /path/to/SynthForensics/captions \
    --variant distilled --enable-compile --gpu 0

# I2V
python generate_i2v.py --captions-dir /path/to/SynthForensics/captions \
    --images-dir /path/to/SynthForensics/I2V/i2v_frames \
    --variant distilled --enable-compile --gpu 0
```

All videos of the dataset were generated with `--variant distilled` and `--enable-compile`. `--start-index` and `--end-index` split the sorted caption list across GPUs (e.g. `0`–`681` and `681`–`1363`). Videos whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

Each video gets a new random seed.

## Settings

| Parameter | Value |
|-----------|-------|
| Model variant | Helios-Distilled (`--variant distilled`) |
| Resolution (W x H) | 640 x 384 |
| Frames | 99 requested at 24 fps; the output videos have 97 frames (about 4 s) |
| Sampling | stage 2 enabled, pyramid steps `[2, 2, 2]`, amplified first chunk, zero init off |
| Guidance scale | 1.0 |
| Chunking | 9 latent frames per chunk, history sizes `[16, 2, 1]`, first frame kept |
| I2V conditioning | reference frame center-cropped to the output aspect ratio and resized to 640 x 384 (Lanczos) |
| Precision | bfloat16 (VAE in float32) |
| torch.compile | text encoder, VAE, and transformer, mode `max-autotune-no-cudagraphs` |
| Attention backend | `flash_hub` (`_flash_3_hub` on GPUs with compute capability 9.0 or higher) |

## Output

Videos are written to `outputs/helios_distilled_t2v_raw/` and `outputs/helios_distilled_i2v_raw/` (override with `--output-dir`), with the same file names as in the released dataset (`_i2v` for I2V):

- `{id}_helios_t2v.mp4`: H.264 video
- `{id}_helios_t2v.json`: prompt, negative prompt, seed, all hyperparameters, model revision and checksums, software versions, and GPU

In the released dataset they are under `videos/raw/helios/` and `metadata/helios/` of the `T2V/` and `I2V/` folders.
