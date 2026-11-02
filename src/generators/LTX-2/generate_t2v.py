#!/usr/bin/env python3
"""
LTX-2.3 Text-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all 1363 videos on GPU 0
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0

    # Split across two GPUs
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --start-index 0 --end-index 681
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 1 --start-index 681 --end-index 1363

    # Regenerate rejected videos from validation CSV
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --regenerate validation_results.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# GPU must be set before importing torch/CUDA
def _set_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
_set_gpu_early()

import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "ltx23_t2v_raw"
OUTPUT_SUFFIX = "_ltx_2-3_t2v"  # released file names: {id}_ltx_2-3_t2v.mp4 / .json

# ─── Generation defaults ─────────────────────────────────────────────────────
NUM_FRAMES = 121
FRAME_RATE = 24
NUM_INFERENCE_STEPS = 30
VIDEO_CFG_SCALE = 3.0
VIDEO_STG_SCALE = 1.0
VIDEO_STG_BLOCKS = [28]
VIDEO_RESCALE_SCALE = 0.7
VIDEO_SKIP_STEP = 0
A2V_GUIDANCE_SCALE = 3.0
AUDIO_CFG_SCALE = 7.0
AUDIO_STG_SCALE = 1.0
AUDIO_STG_BLOCKS = [28]
AUDIO_RESCALE_SCALE = 0.7
AUDIO_SKIP_STEP = 0
V2A_GUIDANCE_SCALE = 3.0
DISTILLED_LORA_STRENGTH = 0.5

NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

# Orientations: landscape (1024x1536) or portrait (1536x1024)
ORIENTATIONS = {
    "landscape": {"height": 1024, "width": 1536},
    "portrait":  {"height": 1536, "width": 1024},
}

# Prompt field order for LTX-2.3
PROMPT_FIELDS = [
    "short_description",
    "dense_description",
    "background",
    "shot_type",
    "camera_movement",
    "lighting",
    "atmosphere",
]


def build_prompt(caption: dict) -> str:
    """Compose a single flowing prompt from structured caption fields."""
    parts = []
    for field in PROMPT_FIELDS:
        value = caption.get(field, "").strip()
        if value:
            # Remove trailing period to avoid double periods when joining
            if value.endswith("."):
                value = value[:-1]
            parts.append(value)
    return ". ".join(parts) + "."


def pick_orientation(seed: int) -> str:
    """Deterministically pick orientation from seed for reproducibility."""
    rng = random.Random(seed)
    return rng.choice(["landscape", "portrait"])


def generate_seed() -> int:
    """Generate a random seed."""
    return random.randint(0, 2**31 - 1)


def clean_stem(name: str) -> str:
    """Remove the output suffix and file extensions from a video/caption name."""
    name = name.replace(".mp4", "").replace(".json", "")
    name = name.replace(OUTPUT_SUFFIX, "")
    return name


def parse_regenerate_csv(csv_path: Path) -> dict:
    """Parse validation CSV and return a dict of stem -> {prompt_override, extra_negative_prompt}."""
    regen_info = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = clean_stem(row["video"].strip())
            regen_info[stem] = {
                "prompt_override": row.get("prompt_override", "").strip(),
                "extra_negative_prompt": row.get("extra_negative_prompt", "").strip(),
            }
    return regen_info


def get_caption_files(args, captions_dir: Path) -> list[Path]:
    """Get the list of caption files to process based on arguments."""
    all_files = sorted(captions_dir.glob("*.json"))
    start = args.start_index if args.start_index is not None else 0
    end = args.end_index if args.end_index is not None else len(all_files)
    return all_files[start:end]


def _get_pkg_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def get_environment_info() -> dict:
    """Collect reproducibility metadata about the environment."""
    import flash_attn
    return {
        "repo_url": "https://github.com/Lightricks/LTX-2.git",
        "repo_commit": "9e8a28e17ac4dd9e49695223d50753a1ebda36fe",
        "model_id": "ltx-2.3-22b-dev",
        "model_url": "https://huggingface.co/Lightricks/LTX-2.3",
        "model_hf_commit": "5a9c1c680bc66c159f708143bf274739961ecd08",
        "model_sha256": {
            "ltx-2.3-22b-dev.safetensors": "7ab7225325bc403448ea84b6db2269811a880e5118cd2ee2b6282a93d585016f",
            "ltx-2.3-22b-distilled-lora-384.safetensors": "2943ab994f3c9d88052e5a2a34cca14e4a2dfc36b1d8c407931d52d5c25dd72b",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": "93800de87dbc448b5b31f3c5c3a1579ba6335151de061a564f6f026b0fc770ad",
        },
        "text_encoder_id": "gemma-3-12b",
        "text_encoder_url": "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized",
        "text_encoder_hf_commit": "68f7ee4fbd59087436ada77ed2d62f373fdd4482",
        "text_encoder_sha256": {
            "model-00001-of-00005.safetensors": "e6fb899db428481aafb45a20130457df6e247e7cb03b7d9f01ee4bc2a9a08138",
            "model-00002-of-00005.safetensors": "d251e7fe9799d529405ddb61705a44cd700bd30a8b66a8d44ae26ddf8365dbc6",
            "model-00003-of-00005.safetensors": "0684ef801385f0669a0b3e4ab160c50877efdbfa40eb97788595985de2743e78",
            "model-00004-of-00005.safetensors": "b4b964e6526f81ccfa625c900b72ce92d5e0fd2debb75998763038ad06b9c541",
            "model-00005-of-00005.safetensors": "4ef2de8f93e165b4e02425769fc566000b0674256ef0c3a27b23a0d45eb12088",
        },
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "flash_attn_version": flash_attn.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "key_deps": {
            "ltx-core": _get_pkg_version("ltx-core"),
            "ltx-pipelines": _get_pkg_version("ltx-pipelines"),
            "xformers": _get_pkg_version("xformers"),
        },
    }


def build_metadata(caption_name: str, prompt: str, negative_prompt: str, seed: int, orientation: str, dims: dict, env_info: dict) -> dict:
    """Build the full metadata dict for reproducibility."""
    return {
        "generator": "LTX-2.3",
        "model_version": "ltx-2.3-22b-dev",
        "pipeline": "ti2vid_two_stages",
        "mode": "t2v",
        "checkpoint": "ltx-2.3-22b-dev.safetensors",
        "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "distilled_lora_strength": DISTILLED_LORA_STRENGTH,
        "spatial_upsampler": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "text_encoder": "gemma-3-12b",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "width": dims["width"],
        "height": dims["height"],
        "num_frames": NUM_FRAMES,
        "frame_rate": FRAME_RATE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "video_cfg_scale": VIDEO_CFG_SCALE,
        "video_stg_scale": VIDEO_STG_SCALE,
        "video_stg_blocks": VIDEO_STG_BLOCKS,
        "video_rescale_scale": VIDEO_RESCALE_SCALE,
        "video_skip_step": VIDEO_SKIP_STEP,
        "a2v_guidance_scale": A2V_GUIDANCE_SCALE,
        "audio_cfg_scale": AUDIO_CFG_SCALE,
        "audio_stg_scale": AUDIO_STG_SCALE,
        "audio_stg_blocks": AUDIO_STG_BLOCKS,
        "audio_rescale_scale": AUDIO_RESCALE_SCALE,
        "audio_skip_step": AUDIO_SKIP_STEP,
        "v2a_guidance_scale": V2A_GUIDANCE_SCALE,
        "quantization": None,
        "dtype": "bfloat16",
        "source_caption": caption_name,
        "image_conditioning_frame_idx": 0,
        "image_conditioning_strength": 1.0,
        "image_conditioning_crf": 0,
        "orientation": orientation,
        "environment": env_info,
    }


def generate_video(pipeline, prompt: str, seed: int, dims: dict, tiling_config, negative_prompt: str = NEGATIVE_PROMPT) -> tuple:
    """Run the pipeline and return (video, audio)."""
    from ltx_core.components.guiders import MultiModalGuiderParams

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=VIDEO_CFG_SCALE,
        stg_scale=VIDEO_STG_SCALE,
        rescale_scale=VIDEO_RESCALE_SCALE,
        modality_scale=A2V_GUIDANCE_SCALE,
        skip_step=VIDEO_SKIP_STEP,
        stg_blocks=VIDEO_STG_BLOCKS,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=AUDIO_CFG_SCALE,
        stg_scale=AUDIO_STG_SCALE,
        rescale_scale=AUDIO_RESCALE_SCALE,
        modality_scale=V2A_GUIDANCE_SCALE,
        skip_step=AUDIO_SKIP_STEP,
        stg_blocks=AUDIO_STG_BLOCKS,
    )

    video, audio = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=dims["height"],
        width=dims["width"],
        num_frames=NUM_FRAMES,
        frame_rate=FRAME_RATE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[],
        tiling_config=tiling_config,
    )
    return video, audio


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 T2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--captions-dir", type=str, required=True, help="Directory containing caption JSON files")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Directory containing model weights")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--start-index", type=int, default=None, help="Start index in sorted caption list (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index in sorted caption list (exclusive)")
    parser.add_argument("--regenerate", type=str, default=None, help="Path to validation CSV for regeneration (columns: video,prompt_override,extra_negative_prompt)")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed for all videos (default: random)")
    args = parser.parse_args()

    # Resolve paths
    captions_dir = Path(args.captions_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    if not captions_dir.exists():
        print(f"Error: captions directory not found: {captions_dir}")
        sys.exit(1)
    if not models_dir.exists():
        print(f"Error: models directory not found: {models_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = models_dir / "v2.3" / "ltx-2.3-22b-dev.safetensors"
    distilled_lora_path = models_dir / "v2.3" / "ltx-2.3-22b-distilled-lora-384.safetensors"
    spatial_upsampler = models_dir / "v2.3" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    gemma_root = models_dir / "gemma-3-12b"

    # Parse regeneration CSV if provided
    regen_info = {}
    if args.regenerate:
        regen_csv = Path(args.regenerate)
        if not regen_csv.exists():
            print(f"Error: regeneration CSV not found: {regen_csv}")
            sys.exit(1)
        regen_info = parse_regenerate_csv(regen_csv)
        print(f"Regeneration mode: {len(regen_info)} videos from {regen_csv.name}")

    # In regeneration mode, only process videos listed in the CSV
    if regen_info:
        todo = []
        for stem in sorted(regen_info.keys()):
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if not json_path.exists():
                print(f"Warning: metadata not found for {stem}, skipping")
                continue
            todo.append(stem)
    else:
        # Normal generation mode
        caption_files = get_caption_files(args, captions_dir)
        if not caption_files:
            print("No caption files to process.")
            return

        todo = []
        for cf in caption_files:
            stem = cf.stem
            mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if mp4_path.exists() and json_path.exists():
                continue  # Already generated, skip
            todo.append(stem)

    print(f"GPU {args.gpu} | Total: {len(todo)} videos to {'regenerate' if regen_info else 'generate'}")

    if not todo:
        print("Nothing to generate. All done!")
        return

    # Collect environment info
    env_info = get_environment_info()

    # Load pipeline (heavy, do once)
    print("Loading LTX-2.3 pipeline...")
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=str(distilled_lora_path),
            strength=DISTILLED_LORA_STRENGTH,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=str(checkpoint),
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(spatial_upsampler),
        gemma_root=str(gemma_root),
        loras=(),
        quantization=None,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(NUM_FRAMES, tiling_config)

    print(f"Pipeline loaded. Starting generation of {len(todo)} videos...\n")

    is_regen = bool(regen_info)

    for i, stem in enumerate(todo):
        mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
        json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"

        if is_regen:
            # Regeneration: read prompt from existing metadata JSON
            old_meta = json.loads(json_path.read_text())
            prompt = old_meta["prompt"]
            negative_prompt = old_meta.get("negative_prompt", NEGATIVE_PROMPT)

            # Apply prompt override if provided
            info = regen_info[stem]
            if info["prompt_override"]:
                prompt = info["prompt_override"]

            # Append extra negative prompt if provided
            if info["extra_negative_prompt"]:
                negative_prompt = negative_prompt.rstrip(". ") + ", " + info["extra_negative_prompt"]

            # Always use a new seed different from the old one
            if args.seed is not None:
                seed = args.seed
            else:
                old_seed = old_meta.get("seed", -1)
                seed = generate_seed()
                while seed == old_seed:
                    seed = generate_seed()
        else:
            # Normal generation: read from caption file
            caption_path = captions_dir / f"{stem}.json"
            with open(caption_path) as f:
                caption = json.load(f)
            prompt = build_prompt(caption)
            negative_prompt = NEGATIVE_PROMPT

            if args.seed is not None:
                seed = args.seed
            else:
                seed = generate_seed()

        orientation = pick_orientation(seed)
        dims = ORIENTATIONS[orientation]

        print(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | {orientation} ({dims['width']}x{dims['height']})")

        # Generate
        t0 = time.time()
        writing = False
        try:
            video, audio = generate_video(pipeline, prompt, seed, dims, tiling_config, negative_prompt)

            # Encode and save video
            writing = True
            encode_video(
                video=video,
                fps=FRAME_RATE,
                audio=audio,
                output_path=str(mp4_path),
                video_chunks_number=video_chunks_number,
            )

            # Save metadata
            metadata = build_metadata(stem + ".json", prompt, negative_prompt, seed, orientation, dims, env_info)
            metadata["generation_time_seconds"] = round(time.time() - t0, 1)
            json_path.write_text(json.dumps(metadata, indent=2))

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s -> {mp4_path.name}")

        except Exception as e:
            print(f"  ERROR generating {stem}: {e}")
            # Clean up partial files
            if writing:
                for p in (mp4_path, json_path):
                    if p.exists():
                        p.unlink()
            continue

    print(f"\nGeneration complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
