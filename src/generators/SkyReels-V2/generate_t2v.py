#!/usr/bin/env python3
"""
SkyReels-V2 Text-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all 1363 videos on GPU 0
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0

    # Split across two GPUs
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --start-index 0 --end-index 681
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 1 --start-index 681 --end-index 1363

    # Regenerate specific videos from validation CSV
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --regenerate validation_results.csv
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

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = SCRIPT_DIR / "models" / "SkyReels-V2-T2V-14B-540P"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "skyreels_v2_t2v_raw"
OUTPUT_SUFFIX = "_skyreels-v2_t2v"  # released file names: {id}_skyreels-v2_t2v.mp4 / .json
REPO_COMMIT = "9351d1315220"  # upstream SkyReels-V2 commit vendored in this folder

# ─── Generation defaults ─────────────────────────────────────────────────────
NUM_FRAMES = 97          # ~4s @ 24fps
FRAME_RATE = 24
SAMPLE_STEPS = 50
SAMPLE_SHIFT = 8.0
GUIDE_SCALE = 6.0
OFFLOAD_MODEL = False
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

# Orientations: landscape (960x544) or portrait (544x960)
ORIENTATIONS = {
    "landscape": {"width": 960, "height": 544},
    "portrait":  {"width": 544, "height": 960},
}

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


def pick_orientation(seed: int) -> str:
    """Deterministically pick orientation from seed for reproducibility."""
    rng = random.Random(seed)
    return rng.choice(["landscape", "portrait"])


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
    import flash_attn
    return {
        "repo_url": "https://github.com/SkyworkAI/SkyReels-V2.git",
        "repo_commit": REPO_COMMIT,
        "model_id": "SkyReels-V2-T2V-14B-540P",
        "model_url": "https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P",
        "model_hf_commit": "7031b32f2537572c1da0ce4de2b36ca69cb489f0",
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
    orientation: str, dims: dict, gen_time: float, env_info: dict,
) -> dict:
    """Build the full metadata dict for reproducibility."""
    return {
        "generator": "SkyReels-V2",
        "task": "t2v-14B-540P",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "width": dims["width"],
        "height": dims["height"],
        "num_frames": NUM_FRAMES,
        "fps": FRAME_RATE,
        "sample_steps": SAMPLE_STEPS,
        "sample_shift": SAMPLE_SHIFT,
        "guide_scale": GUIDE_SCALE,
        "offload_model": OFFLOAD_MODEL,
        "dtype": "bfloat16",
        "orientation": orientation,
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="SkyReels-V2 T2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--captions-dir", type=str, required=True)
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
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)

    if not captions_dir.exists():
        logging.error(f"Captions directory not found: {captions_dir}")
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
    logging.info("Loading SkyReels-V2 T2V pipeline...")
    from skyreels_v2_infer.pipelines import Text2VideoPipeline

    pipe = Text2VideoPipeline(
        model_path=str(ckpt_dir),
        dit_path=str(ckpt_dir),
        use_usp=False,
        offload=OFFLOAD_MODEL,
    )

    logging.info(f"Pipeline loaded. Starting T2V generation of {len(todo)} videos...\n")

    is_regen = bool(regen_info)

    for i, stem in enumerate(todo):
        mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
        json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"

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

        orientation = pick_orientation(seed)
        dims = ORIENTATIONS[orientation]

        logging.info(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | {orientation} ({dims['width']}x{dims['height']})")

        t0 = time.time()
        writing = False
        try:
            with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=dims["height"],
                    width=dims["width"],
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
                orientation, dims, gen_time, env_info,
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

    logging.info(f"\nT2V generation complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
