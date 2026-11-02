# SkyReels-V2 Generation

How we generated the SkyReels-V2 videos of SynthForensics (1,363 T2V and 1,363 I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/SkyReels-V2/`.

## Environment

Tested with Python 3.12, PyTorch 2.7.1 (CUDA 12.8), and flash-attn 2.8.0.post2. The upstream code is vendored in this folder at commit [`9351d13`](https://github.com/SkyworkAI/SkyReels-V2/tree/9351d13152207cc04de780e055346b08ade0b851). `environment.yml` pins the packages of the environment we used.

```bash
conda env create -f environment.yml
conda activate skyreels
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| SkyReels-V2 T2V 14B 540P, with Wan2.1 VAE and umT5-XXL text encoder | Skywork/SkyReels-V2-T2V-14B-540P | [`7031b32`](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P/tree/7031b32f2537572c1da0ce4de2b36ca69cb489f0) |
| SkyReels-V2 I2V 14B 540P, with Wan2.1 VAE, umT5-XXL text encoder and CLIP image encoder | Skywork/SkyReels-V2-I2V-14B-540P | [`0205be4`](https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P/tree/0205be4d0df6a2fbcf216bd8cdfb242a70c8a745) |

```bash
hf download Skywork/SkyReels-V2-T2V-14B-540P \
    --revision 7031b32f2537572c1da0ce4de2b36ca69cb489f0 \
    --local-dir models/SkyReels-V2-T2V-14B-540P

hf download Skywork/SkyReels-V2-I2V-14B-540P \
    --revision 0205be4d0df6a2fbcf216bd8cdfb242a70c8a745 \
    --local-dir models/SkyReels-V2-I2V-14B-540P
```

The scripts read the weights from these folders by default (override with `--ckpt-dir`). The SHA-256 checksum of every weight file is stored in the `environment` field of each video's metadata in the released dataset.

## Inputs

From the SynthForensics dataset on Hugging Face:

- `captions/{id}.json`: structured caption of each source video (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame (I2V only)

## Prompt construction

The prompt joins 7 of the 8 caption fields into a single paragraph, in this order: `dense_description`, `style`, `shot_type`, `camera_movement`, `background`, `atmosphere`, `lighting`. The `short_description` field is not used.

## Generation

```bash
# T2V
python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0

# I2V
python generate_i2v.py --captions-dir /path/to/SynthForensics/captions \
    --images-dir /path/to/SynthForensics/I2V/i2v_frames --gpu 0
```

`--start-index` and `--end-index` split the sorted caption list across GPUs (e.g. `0`–`681` and `681`–`1363`). Videos whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

In I2V the orientation follows the reference frame. The prompt, seed and orientation of each released video are recorded in its JSON sidecar.

## Settings

| Parameter | Value |
|-----------|-------|
| Pipeline | `Text2VideoPipeline` / `Image2VideoPipeline` of `skyreels_v2_infer`, single GPU |
| Sampler | flow-matching UniPC, 50 steps, shift 8.0 |
| Guidance | CFG 6.0 |
| Resolution (W x H) | 960 x 544 (landscape) or 544 x 960 (portrait) |
| Frames | 97 at 24 fps (about 4 s) |
| I2V conditioning | reference frame center-cropped to the output aspect ratio, used as the first frame |
| Precision | bfloat16 |
| Model offloading | off |

## Output

Videos are written to `outputs/skyreels_v2_t2v_raw/` and `outputs/skyreels_v2_i2v_raw/` (override with `--output-dir`), with the same file names as in the released dataset (`_i2v` for I2V):

- `{id}_skyreels-v2_t2v.mp4`: H.264 video
- `{id}_skyreels-v2_t2v.json`: prompt, negative prompt, seed, all hyperparameters, model revision, software versions, and GPU

In the released dataset they are under `videos/raw/skyreels-v2/` and `metadata/skyreels-v2/` of the `T2V/` and `I2V/` folders.
