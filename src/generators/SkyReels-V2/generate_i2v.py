#!/usr/bin/env python3
"""
SkyReels-V2 Image-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all 1363 videos on GPU 0
    python generate_i2v.py --captions-dir /path/to/SynthForensics/captions \
        --images-dir /path/to/SynthForensics/I2V/i2v_frames --gpu 0

    # Split across two GPUs
    python generate_i2v.py --captions-dir /path/to/SynthForensics/captions \
        --images-dir /path/to/SynthForensics/I2V/i2v_frames --gpu 0 --start-index 0 --end-index 681
    python generate_i2v.py --captions-dir /path/to/SynthForensics/captions \
        --images-dir /path/to/SynthForensics/I2V/i2v_frames --gpu 1 --start-index 681 --end-index 1363

    # Regenerate specific videos from validation CSV
    python generate_i2v.py --captions-dir /path/to/SynthForensics/captions \
        --images-dir /path/to/SynthForensics/I2V/i2v_frames --gpu 0 --regenerate validation_results.csv
"""

import argparse
import csv
import json
import logging
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

import imageio
import torch
from diffusers.utils import load_image
from PIL import Image

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = SCRIPT_DIR / "models" / "SkyReels-V2-I2V-14B-540P"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "skyreels_v2_i2v_raw"
OUTPUT_SUFFIX = "_skyreels-v2_i2v"  # released file names: {id}_skyreels-v2_i2v.mp4 / .json
REPO_COMMIT = "9351d1315220"  # upstream SkyReels-V2 commit vendored in this folder

# ─── Generation defaults ─────────────────────────────────────────────────────
NUM_FRAMES = 97          # ~4s @ 24fps
FRAME_RATE = 24
SAMPLE_STEPS = 50
SAMPLE_SHIFT = 8.0
GUIDE_SCALE = 6.0
WIDTH = 960
HEIGHT = 544
OFFLOAD_MODEL = False
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

# Prompt field order
PROMPT_FIELDS = [
    "dense_description",
    "style",
    "shot_type",
    "camera_movement",
    "background",
    "atmosphere",
    "lighting",
]

PREFIXED_FIELDS = {
    "style": "The style is {}",
    "shot_type": "The shot type is {}",
    "camera_movement": "The camera movement is {}",
}


def build_prompt(caption: dict) -> str:
    """Compose a single flowing prompt from structured caption fields."""
    parts = []
    for field in PROMPT_FIELDS:
        value = caption.get(field, "").strip()
        if not value:
            continue
        if value.endswith("."):
            value = value[:-1]
        if field in PREFIXED_FIELDS:
            value = PREFIXED_FIELDS[field].format(value.lower())
        parts.append(value)
    return ". ".join(parts) + "."


def generate_seed() -> int:
    """Generate a random seed."""
    return random.randint(0, 2**63 - 1)


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


def resizecrop(image: Image.Image, th: int, tw: int) -> Image.Image:
    """
    Aspect-preserving center crop to target height (th) and width (tw).
    Same logic as SkyReels-V2's official resizecrop.
    """
    w, h = image.size
    if w == tw and h == th:
        return image
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    return image


def get_environment_info() -> dict:
    """Collect reproducibility metadata about the environment."""
    import flash_attn
    return {
        "repo_url": "https://github.com/SkyworkAI/SkyReels-V2.git",
        "repo_commit": REPO_COMMIT,
        "model_id": "SkyReels-V2-I2V-14B-540P",
        "model_url": "https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P",
        "model_hf_commit": "0205be4d0df6a2fbcf216bd8cdfb242a70c8a745",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "flash_attn_version": flash_attn.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "key_deps": {
            "skyreels_v2_infer": "local",
            "easydict": _get_pkg_version("easydict"),
            "imageio": _get_pkg_version("imageio"),
            "diffusers": _get_pkg_version("diffusers"),
        },
    }


def _get_pkg_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def get_caption_files(args, captions_dir: Path) -> list[Path]:
    """Get the list of caption files to process."""
    all_files = sorted(captions_dir.glob("*.json"))
    start = args.start_index if args.start_index is not None else 0
    end = args.end_index if args.end_index is not None else len(all_files)
    return all_files[start:end]


def build_metadata(
    caption_name: str, prompt: str, negative_prompt: str, seed: int,
    source_image: str, width: int, height: int,
    gen_time: float, env_info: dict,
) -> dict:
    """Build the full metadata dict for reproducibility."""
    return {
        "generator": "SkyReels-V2",
        "task": "i2v-14B-540P",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "source_image": source_image,
        "width": width,
        "height": height,
        "num_frames": NUM_FRAMES,
        "fps": FRAME_RATE,
        "sample_steps": SAMPLE_STEPS,
        "sample_shift": SAMPLE_SHIFT,
        "guide_scale": GUIDE_SCALE,
        "offload_model": OFFLOAD_MODEL,
        "dtype": "bfloat16",
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="SkyReels-V2 I2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--captions-dir", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--regenerate", type=str, default=None, help="Path to validation CSV for regeneration")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed for all videos")
    parser.add_argument("--offload-model", type=str, default=None, choices=["true", "false"],
                        help="Override offload_model (default: false)")
    args = parser.parse_args()

    global OFFLOAD_MODEL
    if args.offload_model is not None:
        OFFLOAD_MODEL = args.offload_model == "true"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    # Resolve paths
    captions_dir = Path(args.captions_dir)
    images_dir = Path(args.images_dir)
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)

    if not captions_dir.exists():
        logging.error(f"Captions directory not found: {captions_dir}")
        sys.exit(1)
    if not images_dir.exists():
        logging.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not ckpt_dir.exists():
        logging.error(f"Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse regeneration CSV if provided
    regen_info = {}
    if args.regenerate:
        regen_csv = Path(args.regenerate)
        if not regen_csv.exists():
            logging.error(f"Regeneration CSV not found: {regen_csv}")
            sys.exit(1)
        regen_info = parse_regenerate_csv(regen_csv)
        logging.info(f"Regeneration mode: {len(regen_info)} videos from {regen_csv.name}")

    # In regeneration mode, only process videos listed in the CSV
    if regen_info:
        todo = []
        for stem in sorted(regen_info.keys()):
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if not json_path.exists():
                logging.warning(f"Metadata not found for {stem}, skipping")
                continue
            todo.append(stem)
    else:
        # Normal generation mode
        caption_files = get_caption_files(args, captions_dir)
        if not caption_files:
            logging.info("No caption files to process.")
            return

        todo = []
        for cf in caption_files:
            stem = cf.stem
            if not (images_dir / f"{stem}.png").exists():
                continue
            mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if mp4_path.exists() and json_path.exists():
                continue
            todo.append(stem)

    logging.info(f"GPU {args.gpu} | Total: {len(todo)} videos to {'regenerate' if regen_info else 'generate'}")

    if not todo:
        logging.info("Nothing to generate. All done!")
        return

    # Collect environment info
    env_info = get_environment_info()

    # Load pipeline
    logging.info("Loading SkyReels-V2 I2V pipeline...")
    from skyreels_v2_infer.pipelines import Image2VideoPipeline

    pipe = Image2VideoPipeline(
        model_path=str(ckpt_dir),
        dit_path=str(ckpt_dir),
        use_usp=False,
        offload=OFFLOAD_MODEL,
    )

    logging.info(f"Pipeline loaded. Starting I2V generation of {len(todo)} videos...\n")

    is_regen = bool(regen_info)

    for i, stem in enumerate(todo):
        mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
        json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
        image_path = images_dir / f"{stem}.png"

        if is_regen:
            old_meta = json.loads(json_path.read_text())
            prompt = old_meta["prompt"]
            negative_prompt = old_meta.get("negative_prompt", NEGATIVE_PROMPT)

            info = regen_info[stem]
            if info["prompt_override"]:
                prompt = info["prompt_override"]
            if info["extra_negative_prompt"]:
                negative_prompt = negative_prompt.rstrip(". ") + ", " + info["extra_negative_prompt"]

            if args.seed is not None:
                seed = args.seed
            else:
                old_seed = old_meta.get("seed", -1)
                seed = generate_seed()
                while seed == old_seed:
                    seed = generate_seed()
        else:
            caption_path = captions_dir / f"{stem}.json"
            with open(caption_path) as f:
                caption = json.load(f)
            prompt = build_prompt(caption)
            negative_prompt = NEGATIVE_PROMPT

            if args.seed is not None:
                seed = args.seed
            else:
                seed = generate_seed()

        logging.info(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | img={image_path.name}")

        t0 = time.time()
        writing = False
        try:
            # Load and preprocess source image
            image = load_image(str(image_path)).convert("RGB")
            image_width, image_height = image.size

            # Determine orientation from source image
            if image_height > image_width:
                h, w = WIDTH, HEIGHT  # portrait: swap
            else:
                h, w = HEIGHT, WIDTH  # landscape: default

            # Scale-to-fill + center crop (same as official CLI)
            image = resizecrop(image, h, w)

            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                video_frames = pipe(
                    image=image.convert("RGB"),
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=h,
                    width=w,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=SAMPLE_STEPS,
                    guidance_scale=GUIDE_SCALE,
                    shift=SAMPLE_SHIFT,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )[0]

            # Save video
            writing = True
            imageio.mimwrite(
                str(mp4_path), video_frames, fps=FRAME_RATE, quality=8,
                output_params=["-loglevel", "error"],
            )

            # Save metadata
            gen_time = time.time() - t0
            metadata = build_metadata(
                stem + ".json", prompt, negative_prompt, seed,
                stem + ".png", w, h, gen_time, env_info,
            )
            json_path.write_text(json.dumps(metadata, indent=2))

            logging.info(f"  Done in {gen_time:.0f}s -> {mp4_path.name}")

        except Exception as e:
            logging.error(f"  ERROR generating {stem}: {e}")
            import traceback
            traceback.print_exc()
            if writing:
                for p in (mp4_path, json_path):
                    if p.exists():
                        p.unlink()
            continue

    logging.info(f"\nI2V generation complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
