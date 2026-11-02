# Self-Forcing Generation

How we generated the Self-Forcing videos of SynthForensics (1,363 T2V, no I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/Self-Forcing/`.

## Environment

Tested with Python 3.10.18, PyTorch 2.7.1 (CUDA 12.9), flash-attn 2.8.2, diffusers 0.31.0, transformers 4.54.1, and accelerate 1.9.0. The upstream code is vendored in this folder at commit [`33593df`](https://github.com/guandeh17/Self-Forcing/tree/33593df3e81fa3ec10239271dd2c100facac6de1). `environment.yml` pins every package of this environment.

```bash
conda env create -f environment.yml
conda activate self-forcing
```

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| Self-Forcing checkpoint `self_forcing_sid_v2.pt` | gdhe17/Self-Forcing | [`6014fb2`](https://huggingface.co/gdhe17/Self-Forcing/tree/6014fb261a5bf90e9328953b49cceb8f9654249e) |
| Wan2.1 T2V 1.3B base model, VAE, UMT5-XXL text encoder | Wan-AI/Wan2.1-T2V-1.3B | [`37ec512`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a) |

```bash
hf download gdhe17/Self-Forcing checkpoints/self_forcing_sid_v2.pt \
    --revision 6014fb261a5bf90e9328953b49cceb8f9654249e \
    --local-dir .
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --include config.json diffusion_pytorch_model.safetensors Wan2.1_VAE.pth \
    models_t5_umt5-xxl-enc-bf16.pth "google/umt5-xxl/*" \
    --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
    --local-dir wan_models/Wan2.1-T2V-1.3B
```

Expected layout:

```
checkpoints/
└── self_forcing_sid_v2.pt
wan_models/
└── Wan2.1-T2V-1.3B/
```

The SHA-256 checksum of every weight file is stored in the `environment` field of each video's metadata.

## Inputs

From the SynthForensics dataset on Hugging Face:

- `captions/{id}.json`: structured caption of each source video

## Prompt construction

The prompt joins the 8 caption fields with `. `, in this order: `style`, `shot_type`, `short_description`, `dense_description`, `background`, `lighting`, `atmosphere`, `camera_movement`. Empty fields are skipped, and a value shorter than 4 words is lowercased and written as `The <field> is <value>` (e.g. `The shot type is close-up shot`).

## Generation

```bash
python multipleVideo.py --config_path configs/self_forcing_sid.yaml \
    --checkpoint_path checkpoints/self_forcing_sid_v2.pt \
    --captions_dir /path/to/SynthForensics/captions
```

Captions whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

The prompt, seed and orientation of each released video are recorded in its JSON sidecar.

## Settings

| Parameter | Value |
|-----------|-------|
| Config | `configs/self_forcing_sid.yaml`, merged over `configs/default_config.yaml` |
| Sampling | 4 denoising steps (`denoising_step_list`), timestep shift 5.0, no classifier-free guidance or negative prompt |
| Generator weights | `generator` entry of the checkpoint, else `generator_ema`, else the whole state dict |
| Resolution (W x H) | 832 x 480 (landscape) or 480 x 832 (portrait) |
| Frames | 81 at 16 fps (about 5 s) |
| Precision | bfloat16 |

## Output

Videos are written to `./output/` (override with `--output_folder`), with the same file names as in the released dataset:

- `{id}_self-forcing_t2v.mp4`: H.264 video without audio
- `{id}_self-forcing_t2v.json`: prompt, seed, resolution, frames and fps, sampling settings, config and checkpoint, source caption, weight checksums, software versions, and GPU

In the released dataset they are under `T2V/videos/raw/self-forcing/` and `T2V/metadata/self-forcing/`.
