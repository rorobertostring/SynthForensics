#!/usr/bin/env python3
"""
CogVideoX-5B Image-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all I2V on GPU 0
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0

    # Split across two GPUs
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0 --start-index 0 --end-index 681
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 1 --start-index 681 --end-index 1363

    # Regenerate specific videos from validation CSV
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0 --regenerate validation_results.csv
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

import torch
from PIL import Image, ImageOps
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_SUFFIX = "_cogvideox_i2v"  # released file names: {id}_cogvideox_i2v.mp4 / .json

# ─── Generation defaults ─────────────────────────────────────────────────────
NUM_FRAMES = 49          # 6s @ 8fps
FRAME_RATE = 8
SAMPLE_STEPS = 100
GUIDE_SCALE = 6.0
WIDTH = 720
HEIGHT = 480

NEGATIVE_PROMPT = (
    "Overexposed, static, blurred, worst quality, low quality, JPEG compression, "
    "ugly, incomplete, extra fingers, poorly drawn, deformed, disfigured, "
    "misshapen limbs, fused fingers, messy background, three legs, blurred eyes, blurred"
)

# ─── Prompt construction ─────────────────────────────────────────────────────
PROMPT_FIELDS = [
    "dense_description",
    "style",
    "shot_type",
    "background",
]

PREFIXED_FIELDS = {
    "style": "The style is {}",
    "shot_type": "The shot type is {}",
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


def get_environment_info() -> dict:
    """Collect reproducibility metadata about the environment."""
    import diffusers
    return {
        "repo_url": "https://github.com/zai-org/CogVideo.git",
        "repo_commit": "7a1af71",
        "model_id": "CogVideoX-5b-I2V",
        "model_url": "https://huggingface.co/THUDM/CogVideoX-5b-I2V",
        "model_hf_commit": "a6f0f4858a8395e7429d82493864ce92bf73af11",
        "model_sha256": {
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "f2e3060199c34a0d18892a19d687455f938b0ac3d2ea7d48f37cb4090e141965",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "1e8d0c62d366b0d9cc3476d2b21ca54afbecea154d54d923da120b2ec174c7e7",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "da91a0051da3f39caf10944b7c9aa66b14ddeffb37a25b087c49fc1692c1a361",
            "vae/diffusion_pytorch_model.safetensors": "a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e",
            "text_encoder/model-00001-of-00002.safetensors": "9162b8ae9152e7a8e3bbebc535c8692783f50aec8cd3bb8ef6a751c432dd6392",
            "text_encoder/model-00002-of-00002.safetensors": "d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0",
        },
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "key_deps": {
            "accelerate": _get_pkg_version("accelerate"),
            "transformers": _get_pkg_version("transformers"),
            "sentencepiece": _get_pkg_version("sentencepiece"),
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
    source_image: str, gen_time: float, env_info: dict,
) -> dict:
    """Build the full metadata dict for reproducibility."""
    return {
        "generator": "CogVideoX",
        "task": "i2v-5B",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "source_image": source_image,
        "num_frames": NUM_FRAMES,
        "fps": FRAME_RATE,
        "width": WIDTH,
        "height": HEIGHT,
        "sample_steps": SAMPLE_STEPS,
        "guide_scale": GUIDE_SCALE,
        "scheduler": "CogVideoXDPMScheduler",
        "use_dynamic_cfg": True,
        "dtype": "bfloat16",
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="CogVideoX-5B I2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--dataset-root", type=str, default=None, help="SynthForensics dataset root (contains captions/ and I2V/i2v_frames/)")
    parser.add_argument("--captions-dir", type=str, default=None, help="Override captions directory (default: <dataset-root>/captions)")
    parser.add_argument("--images-dir", type=str, default=None, help="Override images directory (default: <dataset-root>/I2V/i2v_frames)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--model-path", type=str, default=None, help="Override model path")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--regenerate", type=str, default=None, help="Path to validation CSV for regeneration (columns: video,prompt_override,extra_negative_prompt)")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed for all videos")
    args = parser.parse_args()
    if args.dataset_root is None and (args.captions_dir is None or args.images_dir is None):
        parser.error("--dataset-root is required unless both --captions-dir and --images-dir are given")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    model_path = Path(args.model_path) if args.model_path else SCRIPT_DIR / "models" / "CogVideoX-5b-I2V"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "outputs" / "cogvideox_i2v_raw"
    captions_dir = Path(args.captions_dir) if args.captions_dir else Path(args.dataset_root) / "captions"
    images_dir = Path(args.images_dir) if args.images_dir else Path(args.dataset_root) / "I2V" / "i2v_frames"

    if not captions_dir.exists():
        logging.error(f"Captions directory not found: {captions_dir}")
        sys.exit(1)
    if not images_dir.exists():
        logging.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    if not model_path.exists():
        logging.error(f"Model directory not found: {model_path}")
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
            # Must have matching image
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
    logging.info("Loading CogVideoX-5B I2V pipeline...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    logging.info(f"Pipeline loaded. Starting I2V generation of {len(todo)} videos...\n")

    is_regen = bool(regen_info)

    for i, stem in enumerate(todo):
        mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
        json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
        image_path = images_dir / f"{stem}.png"

        if is_regen:
            # Regeneration: read prompt from existing metadata JSON
            old_meta = json.loads(json_path.read_text())
            prompt = old_meta["prompt"]
            negative_prompt = old_meta.get("negative_prompt", NEGATIVE_PROMPT)

            # Apply prompt modification if provided
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

        logging.info(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | img={image_path.name}")

        t0 = time.time()
        writing = False
        try:
            # Load source image and scale-to-fill + center crop to target dims
            # (avoids aspect-ratio distortion from diffusers' default stretch resize)
            image = load_image(str(image_path))
            image = ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.LANCZOS)

            video_frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=SAMPLE_STEPS,
                guidance_scale=GUIDE_SCALE,
                use_dynamic_cfg=True,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]

            # Save video
            writing = True
            export_to_video(video_frames, str(mp4_path), fps=FRAME_RATE)

            # Save metadata
            gen_time = time.time() - t0
            metadata = build_metadata(
                stem + ".json", prompt, negative_prompt, seed, stem + ".png", gen_time, env_info,
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
