# Generators

Code used to generate the synthetic videos of SynthForensics. Each folder contains the upstream code of one generator at a pinned commit, our batch scripts, and a `GENERATION.md` with the environment, the model weights, the commands, and the settings. For questions or issues about a model itself (installation, model code, weights), please refer to its original repository, linked in the corresponding `GENERATION.md`.

| Generator | Modes | Scripts | Dataset folder | Docs |
|-----------|-------|---------|----------------|------|
| CogVideoX | T2V, I2V | `generate_t2v.py`, `generate_i2v.py` | `cogvideox` | [GENERATION.md](CogVideoX/GENERATION.md) |
| daVinci-MagiHuman | T2V, I2V | `generate_t2v.py`, `generate_i2v.py` | `daVinci-MagiHuman` | [GENERATION.md](daVinci-MagiHuman/GENERATION.md) |
| Helios | T2V, I2V | `generate_t2v.py`, `generate_i2v.py` | `helios` | [GENERATION.md](Helios/GENERATION.md) |
| LTX-2.3 | T2V, I2V | `generate_t2v.py`, `generate_i2v.py` | `ltx2-3` | [GENERATION.md](LTX-2/GENERATION.md) |
| MAGI-1 | T2V, I2V | `generate.py --mode t2v` / `--mode i2v` | `magi-1` | [GENERATION.md](MAGI-1/GENERATION.md) |
| Self-Forcing | T2V | `multipleVideo.py` | `self-forcing` | [GENERATION.md](Self-Forcing/GENERATION.md) |
| SkyReels-V2 | T2V, I2V | `generate_t2v.py`, `generate_i2v.py` | `skyreels-v2` | [GENERATION.md](SkyReels-V2/GENERATION.md) |
| Wan2.1 | T2V, I2V | `generate_t2v.py`, `generate_i2v.py` | `wan2-1` | [GENERATION.md](Wan2.1/GENERATION.md) |

## Inputs

All scripts read their inputs from the [SynthForensics dataset](https://huggingface.co/datasets/SynthForensics/SynthForensics) on Hugging Face. The dataset is gated: accept the terms on its page before downloading.

- `captions/{id}.json`: structured caption of each of the 1,363 source videos (T2V and I2V)
- `I2V/i2v_frames/{id}.png`: reference frame of each source video (I2V only)

```bash
hf download SynthForensics/SynthForensics --repo-type dataset \
    --include "captions/*" --local-dir /path/to/SynthForensics
hf download SynthForensics/SynthForensics --repo-type dataset \
    --include "I2V/i2v_frames/*" --local-dir /path/to/SynthForensics
```

## Outputs

For every video, the scripts write an `.mp4` and a `.json` with the same name as in the released dataset, `{id}_{generator}_{t2v|i2v}` (e.g. `000_cogvideox_t2v.mp4`). The `.json` holds the prompt, the seed, the hyperparameters, and the model and software versions.

## Compression versions

`utils/make_compression_versions.py` builds the Canonical, CRF23, and CRF40 versions of a folder of Raw videos with the FFmpeg command used for the dataset: the Canonical version re-encodes each Raw video with H.264 at CRF 0, YUV420p, and BT.709 color tags, and the CRF23 and CRF40 versions re-encode the Canonical one.

```bash
python src/generators/utils/make_compression_versions.py \
    --input_dir /path/to/raw_videos --output_dir /path/to/output
```

The released versions were produced with x264 core 164 r3095 and FFmpeg 6.0.1 (all I2V videos; T2V videos of LTX-2.3, Helios, and daVinci-MagiHuman) or FFmpeg 8.0 (T2V videos of CogVideoX, MAGI-1, Self-Forcing, SkyReels-V2, and Wan2.1); with the same versions, the script reproduces the released files byte for byte. The FFmpeg version matters because the Raw videos have no color tags: when BT.709 tags are requested, FFmpeg 8.0 converts the colors, while FFmpeg 6.0.1 keeps the pixel values. Use `--ffmpeg` to choose the executable.

## Reproducibility

Every released video ships with its full generation metadata (seed, prompt, negative prompt, resolution, and all sampling hyperparameters). Re-running our scripts with these settings is not guaranteed to reproduce the released videos bit-for-bit, since diffusion sampling depends on GPU model, driver, CUDA, and library versions.
