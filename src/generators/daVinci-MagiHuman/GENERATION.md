# daVinci-MagiHuman Generation

How we generated the daVinci-MagiHuman videos of SynthForensics (1,363 T2V and 1,363 I2V). The overall procedure and the note on reproducibility are in [`../README.md`](../README.md). All commands are run from `src/generators/daVinci-MagiHuman/`.

## Environment

Tested with Python 3.12, PyTorch 2.10.0 (CUDA 13.0), flash-attn 2.8.3, and MagiCompiler at commit [`719f40e`](https://github.com/SandAI-org/MagiCompiler/tree/719f40e1e7e7d7c56c4c3274cb1e4244fc8db281). The upstream code is vendored in this folder at commit [`f9764ce`](https://github.com/GAIR-NLP/daVinci-MagiHuman/tree/f9764ced3f33017e75795f80adb9dcfd3ca68d38), with the changes listed in [Source code modifications](#source-code-modifications). `environment.yml` pins the packages of the environment we used.

```bash
conda env create -f environment.yml
conda activate davinci
pip install flash-attn==2.8.3 --no-build-isolation
pip install --no-deps git+https://github.com/SandAI-org/MagiCompiler.git@719f40e1e7e7d7c56c4c3274cb1e4244fc8db281
```

## Model weights

| Component | Hugging Face repo | Revision |
|-----------|-------------------|----------|
| daVinci-MagiHuman distilled model, 1080p super-resolution model, TurboVAE decoder | GAIR/daVinci-MagiHuman | [`7fe95e5`](https://huggingface.co/GAIR/daVinci-MagiHuman/tree/7fe95e50c11bd92bdadf94cd51dbec18427f8e0c) |
| T5Gemma 9B-9B text encoder | google/t5gemma-9b-9b-ul2 | [`7071a2a`](https://huggingface.co/google/t5gemma-9b-9b-ul2/tree/7071a2ace9c8353e12e723e5382cc7256ee41d9f) |
| Stable Audio Open 1.0 audio VAE | stabilityai/stable-audio-open-1.0 | [`f21265c`](https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/f21265c1e2710b3bd2386596943f0007f55f802e) |
| Wan2.2 video VAE | Wan-AI/Wan2.2-TI2V-5B | [`921dbaf`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e) |

T5Gemma and Stable Audio Open are gated: request access on their Hugging Face pages before downloading.

```bash
hf download GAIR/daVinci-MagiHuman --include "distill/*" "1080p_sr/*" "turbo_vae/*" \
    --revision 7fe95e50c11bd92bdadf94cd51dbec18427f8e0c --local-dir checkpoints/davinci-magihuman
hf download google/t5gemma-9b-9b-ul2 \
    --revision 7071a2ace9c8353e12e723e5382cc7256ee41d9f --local-dir checkpoints/t5gemma-9b-9b-ul2
hf download stabilityai/stable-audio-open-1.0 model_config.json model.safetensors \
    --revision f21265c1e2710b3bd2386596943f0007f55f802e --local-dir checkpoints/stable-audio-open-1.0
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir checkpoints/Wan2.2-TI2V-5B
```

Expected layout (the paths used by `example/distill_sr_1080p/config.json`):

```
checkpoints/
├── davinci-magihuman/
│   ├── distill/
│   ├── 1080p_sr/
│   └── turbo_vae/
├── t5gemma-9b-9b-ul2/
├── stable-audio-open-1.0/
└── Wan2.2-TI2V-5B/
```

The SHA-256 checksums of the weight files (first shard for sharded models) are stored in the `environment` field of each video's metadata.

## Inputs

From the SynthForensics dataset on Hugging Face:

- `captions/{id}.json`: structured caption of each source video (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame (I2V only)

## Prompt construction

The prompt joins 6 of the 8 caption fields into sentences, in this order: `dense_description`, `background`, `shot_type` (as "Captured in a static ..."), `lighting` ("The lighting is ..."), `atmosphere` ("The atmosphere is ..."), and `style` ("The style is ..."); leading phrases such as "The shot type is" are first removed from the last four. A `Background Sound:` section with `<No prominent background sound>` is appended. The `short_description` and `camera_movement` fields are not used.

## Generation

```bash
# T2V
CUDA_VISIBLE_DEVICES=0 CPU_OFFLOAD=true SR2_1080=true KEEP_SR_ON_GPU=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 generate_t2v.py \
    --config-load-path example/distill_sr_1080p/config.json \
    --captions-dir /path/to/SynthForensics/captions

# I2V
CUDA_VISIBLE_DEVICES=0 CPU_OFFLOAD=true SR2_1080=true KEEP_SR_ON_GPU=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 generate_i2v.py \
    --config-load-path example/distill_sr_1080p/config.json \
    --captions-dir /path/to/SynthForensics/captions \
    --images-dir /path/to/SynthForensics/I2V/i2v_frames
```

`--start-index` and `--end-index` split the sorted caption list across GPUs (e.g. `0`–`681` and `681`–`1363`); give each `torchrun` process its own `--master_port`. Videos whose `.mp4` and `.json` already exist in the output folder are skipped, so an interrupted run can be resumed.

Each video gets a new random seed. In T2V the orientation (landscape or portrait) is drawn from the seed, in I2V it follows the reference frame.

## Settings

| Parameter | Value |
|-----------|-------|
| Config | `example/distill_sr_1080p/config.json` |
| Base stage | distilled model, 8 steps, `cfg_number` 1 |
| Super-resolution stage | 1080p SR model, 5 steps, `sr_cfg_number` 2, local attention layers (`SR2_1080=true`) |
| Resolution (W x H) | base 448 x 256, output 1920 x 1088 (landscape); base 256 x 448, output 1088 x 1920 (portrait) |
| Duration | 5 s at 25 fps |
| I2V conditioning | reference frame as the image input of the pipeline |
| Decoder | TurboVAE |
| Precision | bfloat16 |

## Output

Videos are written to `outputs/davinci_t2v_raw/` and `outputs/davinci_i2v_raw/` (override with `--output-dir`), with the same file names as in the released dataset (`_i2v` for I2V):

- `{id}_daVinci-MagiHuman_t2v.mp4`: H.264 video with the audio track generated by the model
- `{id}_daVinci-MagiHuman_t2v.json`: prompt, negative prompt, seed, hyperparameters, model revision and checksums, software versions, and GPU

In the released dataset they are under `videos/raw/daVinci-MagiHuman/` and `metadata/daVinci-MagiHuman/` of the `T2V/` and `I2V/` folders.

## Source code modifications

Changes to the upstream code at commit `f9764ce`:

- `inference/model/t5_gemma/t5_gemma_model.py`: the T5Gemma encoder is not offloaded to the CPU when `CPU_OFFLOAD` is set (only on GPUs with 48 GB or less).
- `inference/pipeline/video_generate.py`: new `KEEP_SR_ON_GPU` variable; when set with `CPU_OFFLOAD` and `SR2_1080`, the super-resolution model stays on the GPU instead of being moved to the CPU.
- `inference/pipeline/entry.py`: `--image_path` is optional in the upstream command-line entry point (T2V when omitted). Not used by `generate_t2v.py` and `generate_i2v.py`.
- `example/*/config.json`: checkpoint paths are relative to this folder; `example/distill_sr_1080p/config.json` is new.
