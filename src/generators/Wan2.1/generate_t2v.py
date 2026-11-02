#!/usr/bin/env python3
"""
Wan 2.1 Text-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all 1363 videos on GPU 0
    python generate_t2v.py --dataset-root /path/to/SynthForensics --gpu 0

    # Split across two GPUs
    python generate_t2v.py --dataset-root /path/to/SynthForensics --gpu 0 --start-index 0 --end-index 681
    python generate_t2v.py --dataset-root /path/to/SynthForensics --gpu 1 --start-index 681 --end-index 1363

    # Regenerate specific videos from validation CSV
    python generate_t2v.py --dataset-root /path/to/SynthForensics --gpu 0 --regenerate validation_results.csv
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

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = SCRIPT_DIR / "models" / "Wan2.1-T2V-14B"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "wan21_t2v_raw"
OUTPUT_SUFFIX = "_wan2-1_t2v"  # released file names: {id}_wan2-1_t2v.mp4 / .json

# ─── Generation defaults ─────────────────────────────────────────────────────
NUM_FRAMES = 81          # 5s @ 16fps
FRAME_RATE = 16
SAMPLE_STEPS = 50
SAMPLE_SHIFT = 5.0
GUIDE_SCALE = 5.0
SAMPLE_SOLVER = "unipc"
OFFLOAD_MODEL = False    # default; set True if VRAM is limited
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Orientations: landscape (832x480) or portrait (480x832)
ORIENTATIONS = {
    "landscape": {"width": 832, "height": 480, "size_str": "832*480"},
    "portrait":  {"width": 480, "height": 832, "size_str": "480*832"},
}

# Prompt field order — same structure as LTX, adapted prefix style for Wan
PROMPT_FIELDS = [
    "style",
    "shot_type",
    "short_description",
    "dense_description",
    "background",
    "lighting",
    "atmosphere",
    "camera_movement",
]

# Prefixed fields get "The {field} is {value}" format
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
        # Remove trailing period
        if value.endswith("."):
            value = value[:-1]
        # Apply prefix template if defined
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
        "repo_url": "https://github.com/Wan-Video/Wan2.1.git",
        "repo_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "model_id": "Wan2.1-T2V-14B",
        "model_url": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B",
        "model_hf_commit": "a064a6c71f5be440641209c07bf2a5ce7a2ff5e4",
        "model_sha256": {
            "diffusion_pytorch_model-00001-of-00006.safetensors": "569d54a07279b89f8281421fccf27ee2459ea853ce6845d3536b8664b0070078",
            "diffusion_pytorch_model-00002-of-00006.safetensors": "b17ff172f262b4da91c31e8c46d3b7707f62cecfff6e18dd0072ab29eb350e46",
            "diffusion_pytorch_model-00003-of-00006.safetensors": "741fd1508cd0288f0bb0f4fb3df6734da521d5ff23746170c848564a367e2cea",
            "diffusion_pytorch_model-00004-of-00006.safetensors": "47b08ba289b127fc64979bc877ee07e1c87aeb8eb8cd47bcfd112682847981ba",
            "diffusion_pytorch_model-00005-of-00006.safetensors": "29e35f4cf0e3a61f4726c960f1babf71d65677a8c99fe704257d665c9498ee88",
            "diffusion_pytorch_model-00006-of-00006.safetensors": "91ac087a40b814331e87f5f015af11ba999275d48d2af49591b6bf7727d3c146",
            "Wan2.1_VAE.pth": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
            "models_t5_umt5-xxl-enc-bf16.pth": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
        },
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "flash_attn_version": flash_attn.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "key_deps": {
            "wan": "editable-local",
            "easydict": _get_pkg_version("easydict"),
            "imageio": _get_pkg_version("imageio"),
        },
    }


def _get_pkg_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def save_video(tensor, save_path: str, fps: int = 16):
    """
    Save video tensor using Wan's cache_video (libx264, quality=8).
    tensor: (C, N, H, W) float in [-1, 1]
    """
    from wan.utils.utils import cache_video
    cache_video(
        tensor=tensor[None],  # cache_video expects (B, C, N, H, W)
        save_file=save_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


def get_caption_files(args, captions_dir: Path) -> list[Path]:
    """Get the list of caption files to process."""
    all_files = sorted(captions_dir.glob("*.json"))
    start = args.start_index if args.start_index is not None else 0
    end = args.end_index if args.end_index is not None else len(all_files)
    return all_files[start:end]


def build_metadata(
    caption_name: str, prompt: str, negative_prompt: str, seed: int,
    orientation: str, dims: dict, gen_time: float,
    env_info: dict,
) -> dict:
    """Build the full metadata dict for reproducibility."""
    return {
        "generator": "Wan2.1",
        "task": "t2v-14B",
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
        "sample_solver": SAMPLE_SOLVER,
        "offload_model": OFFLOAD_MODEL,
        "dtype": "bfloat16",
        "orientation": orientation,
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 T2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--dataset-root", type=str, default=None, help="SynthForensics dataset root (contains captions/)")
    parser.add_argument("--captions-dir", type=str, default=None, help="Override captions directory (default: <dataset-root>/captions)")
    parser.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--regenerate", type=str, default=None, help="Path to validation CSV for regeneration (columns: video,prompt_override,extra_negative_prompt)")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed for all videos")
    parser.add_argument("--offload-model", type=str, default=None, choices=["true", "false"],
                        help="Override offload_model (default: false)")
    args = parser.parse_args()
    if args.dataset_root is None and args.captions_dir is None:
        parser.error("--dataset-root is required unless --captions-dir is given")

    global OFFLOAD_MODEL
    if args.offload_model is not None:
        OFFLOAD_MODEL = args.offload_model == "true"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    # Resolve paths
    captions_dir = Path(args.captions_dir) if args.captions_dir else Path(args.dataset_root) / "captions"
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

    # Load pipeline (heavy, do once)
    logging.info("Loading Wan 2.1 T2V pipeline...")
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

    cfg = WAN_CONFIGS["t2v-14B"]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    logging.info(f"Pipeline loaded. Starting generation of {len(todo)} videos...\n")

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

        logging.info(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | {orientation} ({dims['width']}x{dims['height']})")

        t0 = time.time()
        writing = False
        try:
            video = wan_t2v.generate(
                prompt,
                n_prompt=negative_prompt,
                size=SIZE_CONFIGS[dims["size_str"]],
                frame_num=NUM_FRAMES,
                shift=SAMPLE_SHIFT,
                sample_solver=SAMPLE_SOLVER,
                sampling_steps=SAMPLE_STEPS,
                guide_scale=GUIDE_SCALE,
                seed=seed,
                offload_model=OFFLOAD_MODEL,
            )

            # Save video
            writing = True
            save_video(video, str(mp4_path), fps=FRAME_RATE)

            # Save metadata
            gen_time = time.time() - t0
            metadata = build_metadata(stem + ".json", prompt, negative_prompt, seed, orientation, dims, gen_time, env_info)
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

    logging.info(f"\nGeneration complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
