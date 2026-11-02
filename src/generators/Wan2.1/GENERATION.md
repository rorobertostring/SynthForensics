# Wan2.1 Generation

How we generated the Wan2.1 videos of SynthForensics (1,363 T2V and 1,363 I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/Wan2.1/`.

## Environment

Tested with Python 3.12, PyTorch 2.7.1 (CUDA 12.8), and flash-attn 2.8.0.post2. The upstream code is vendored in this folder at commit [`9737cba`](https://github.com/Wan-Video/Wan2.1/tree/9737cba9c1c3c4d04b33fcad41c111989865d315). `environment.yml` pins every package of the environment we used.

```bash
conda env create -f environment.yml
conda activate wan21
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install --no-deps -e .
```

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| Wan2.1 T2V 14B | Wan-AI/Wan2.1-T2V-14B | [`a064a6c`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/a064a6c71f5be440641209c07bf2a5ce7a2ff5e4) |
| Wan2.1 I2V 14B 480P | Wan-AI/Wan2.1-I2V-14B-480P | [`6b73f84`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/tree/6b73f84e66371cdfe870c72acd6826e1d61cf279) |

```bash
hf download Wan-AI/Wan2.1-T2V-14B \
    --revision a064a6c71f5be440641209c07bf2a5ce7a2ff5e4 \
    --local-dir models/Wan2.1-T2V-14B
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    --revision 6b73f84e66371cdfe870c72acd6826e1d61cf279 \
    --local-dir models/Wan2.1-I2V-14B-480P
```

Expected layout:

```
models/
├── Wan2.1-T2V-14B/
└── Wan2.1-I2V-14B-480P/
```

The SHA-256 checksum of every weight file is stored in the `environment` field of each video's metadata.

## Inputs

From the SynthForensics dataset on Hugging Face, passed with `--dataset-root`:

- `captions/{id}.json`: structured caption of each source video (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame (I2V only)

## Prompt construction

The prompt joins the 8 caption fields in this order: `style`, `shot_type`, `short_description`, `dense_description`, `background`, `lighting`, `atmosphere`, `camera_movement`.

## Generation

```bash
# T2V
python generate_t2v.py --dataset-root /path/to/SynthForensics --gpu 0

# I2V
python generate_i2v.py --dataset-root /path/to/SynthForensics --resolution 480p --gpu 0
```

`--start-index` and `--end-index` split the sorted caption list across GPUs (e.g. `0`–`681` and `681`–`1363`). Videos whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

In I2V the orientation follows the reference frame, and the videos use `--resolution 480p` (Wan2.1-I2V-14B-480P). The prompt, seed and orientation of each released video are recorded in its JSON sidecar.

## Settings

| Parameter | T2V | I2V |
|-----------|-----|-----|
| Task | `t2v-14B` | `i2v-14B` |
| Resolution | 832 x 480 (landscape) or 480 x 832 (portrait) | 480p |
| Frames | 81 at 16 fps (about 5 s) | 81 at 16 fps (about 5 s) |
| Sampling steps | 50 | 40 |
| Sample shift | 5.0 | 3.0 |
| Guidance scale | 5.0 | 5.0 |
| Solver | UniPC | UniPC |
| Precision | bfloat16 | bfloat16 |

## Output

Videos are written to `outputs/wan21_t2v_raw/` and `outputs/wan21_i2v_480p_raw/` (override with `--output-dir`), with the same file names as in the released dataset (`_i2v` for I2V):

- `{id}_wan2-1_t2v.mp4`: H.264 video
- `{id}_wan2-1_t2v.json`: prompt, negative prompt, seed, all hyperparameters, model revisions and checksums, software versions, and GPU

In the released dataset they are under `videos/raw/wan2-1/` and `metadata/wan2-1/` of the `T2V/` and `I2V/` folders.
