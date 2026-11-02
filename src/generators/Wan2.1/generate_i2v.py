#!/usr/bin/env python3
"""
Wan 2.1 Image-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all I2V at 480P on GPU 0
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0 --resolution 480p

    # Generate all I2V at 720P on GPU 1 (uses offload_model=True)
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 1 --resolution 720p

    # Split across two GPUs
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0 --resolution 480p --start-index 0 --end-index 681
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 1 --resolution 480p --start-index 681 --end-index 1363

    # Regenerate specific videos from validation CSV
    python generate_i2v.py --dataset-root /path/to/SynthForensics --gpu 0 --resolution 480p --regenerate validation_results.csv
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
from PIL import Image

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_SUFFIX = "_wan2-1_i2v"  # released file names: {id}_wan2-1_i2v.mp4 / .json

# ─── Generation defaults ─────────────────────────────────────────────────────
NUM_FRAMES = 81          # 5s @ 16fps
FRAME_RATE = 16
SAMPLE_STEPS = 40        # I2V default
GUIDE_SCALE = 5.0
SAMPLE_SOLVER = "unipc"
NEGATIVE_PROMPT = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Resolution-specific configs
RESOLUTION_CONFIGS = {
    "480p": {
        "ckpt_dir": SCRIPT_DIR / "models" / "Wan2.1-I2V-14B-480P",
        "output_dir": SCRIPT_DIR / "outputs" / "wan21_i2v_480p_raw",
        "size_str": "832*480",       # used for max_area lookup
        "sample_shift": 3.0,
        "offload_model": False,      # default at 480P; set True if VRAM is limited
        "model_id": "Wan2.1-I2V-14B-480P",
        "model_url": "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P",
        "model_hf_commit": "6b73f84e66371cdfe870c72acd6826e1d61cf279",
        "model_sha256": {
            "diffusion_pytorch_model-00001-of-00007.safetensors": "09c0170242cfe9598208724585196ca18f294928fe25971149e1d7b37b3b51d6",
            "diffusion_pytorch_model-00002-of-00007.safetensors": "9e9dd8069241f0cbfe8f269e784e915ddaa7f177272b9850ae88c494d418c181",
            "diffusion_pytorch_model-00003-of-00007.safetensors": "22af36eb620c381d57a2dd5d5e3a001bf90ba0ca32f2d5013d6ca45225b11d5f",
            "diffusion_pytorch_model-00004-of-00007.safetensors": "5c2be0421b8df4a8e7dd31cdbb6489567438ae14b812b57518e43c6f4e49a755",
            "diffusion_pytorch_model-00005-of-00007.safetensors": "0ba7f3ab69f414f23e5566e607b9385a8f67907302cbaaf86507b3a826a20dda",
            "diffusion_pytorch_model-00006-of-00007.safetensors": "d7287e9233cbf67a75fce8ef8be24232ec93d6336f148972465db00008d69e28",
            "diffusion_pytorch_model-00007-of-00007.safetensors": "d1bc30f07b162c34b90e4ee4a349f81b6e1a3a342187ea9ea153c9e4dbb07676",
            "Wan2.1_VAE.pth": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
            "models_t5_umt5-xxl-enc-bf16.pth": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38",
        },
    },
    "720p": {
        "ckpt_dir": SCRIPT_DIR / "models" / "Wan2.1-I2V-14B-720P",
        "output_dir": SCRIPT_DIR / "outputs" / "wan21_i2v_720p_raw",
        "size_str": "1280*720",      # used for max_area lookup
        "sample_shift": 5.0,
        "offload_model": True,       # 720P needs offload for VAE decode
        "model_id": "Wan2.1-I2V-14B-720P",
        "model_url": "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P",
        "model_hf_commit": "8823af45fcc58a8aa999a54b04be9abc7d2aac98",
        "model_sha256": {
            "diffusion_pytorch_model-00001-of-00007.safetensors": "3c36c371b3060931770f693f22253a7de7c76fc79cffb0ab08032fb5a04784e4",
            "diffusion_pytorch_model-00002-of-00007.safetensors": "eb465c3d3ec91bf5562f8f9c1a7cbb010266012f255d05d0d5c8815524a1dd43",
            "diffusion_pytorch_model-00003-of-00007.safetensors": "072ebcc282f2d35c016c22fbc46e1c8eefa85745b9038ca4eea9e34cf4a47186",
            "diffusion_pytorch_model-00004-of-00007.safetensors": "3b3e8561340ad1d43e4569912810fdc24f83ddbcf4f2863c968114479113324b",
            "diffusion_pytorch_model-00005-of-00007.safetensors": "9700f2ba6194aad23a458adc6e73352ea982a0b1c7606f459cd3e559eacc270d",
            "diffusion_pytorch_model-00006-of-00007.safetensors": "148e81b477494ef9590acd8983f45116d4684919659ea2c0e747050458aa2346",
            "diffusion_pytorch_model-00007-of-00007.safetensors": "19f7ff97683bdbe3ee9ddada6f85d3e5d91a9b871a3beffdb5f19a1cc6208647",
            "Wan2.1_VAE.pth": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
            "models_t5_umt5-xxl-enc-bf16.pth": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38",
        },
    },
}

# Prompt field order — same as T2V script
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


def get_environment_info(res_cfg: dict) -> dict:
    """Collect reproducibility metadata about the environment."""
    import flash_attn
    return {
        "repo_url": "https://github.com/Wan-Video/Wan2.1.git",
        "repo_commit": "9737cba9c1c3c4d04b33fcad41c111989865d315",
        "model_id": res_cfg["model_id"],
        "model_url": res_cfg["model_url"],
        "model_hf_commit": res_cfg["model_hf_commit"],
        "model_sha256": res_cfg["model_sha256"],
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
    width: int, height: int, orientation: str,
    source_image: str, resolution: str, res_cfg: dict,
    gen_time: float, env_info: dict,
) -> dict:
    """Build the full metadata dict for reproducibility."""
    return {
        "generator": "Wan2.1",
        "task": "i2v-14B",
        "resolution": resolution,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "orientation": orientation,
        "width": width,
        "height": height,
        "source_image": source_image,
        "num_frames": NUM_FRAMES,
        "fps": FRAME_RATE,
        "sample_steps": SAMPLE_STEPS,
        "sample_shift": res_cfg["sample_shift"],
        "guide_scale": GUIDE_SCALE,
        "sample_solver": SAMPLE_SOLVER,
        "offload_model": res_cfg["offload_model"],
        "dtype": "bfloat16",
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 I2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--resolution", type=str, required=True, choices=["480p", "720p"], help="Target resolution")
    parser.add_argument("--dataset-root", type=str, default=None, help="SynthForensics dataset root (contains captions/ and I2V/i2v_frames/)")
    parser.add_argument("--captions-dir", type=str, default=None, help="Override captions directory (default: <dataset-root>/captions)")
    parser.add_argument("--images-dir", type=str, default=None, help="Override images directory (default: <dataset-root>/I2V/i2v_frames)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--regenerate", type=str, default=None, help="Path to validation CSV for regeneration (columns: video,prompt_override,extra_negative_prompt)")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed for all videos")
    parser.add_argument("--offload-model", type=str, default=None, choices=["true", "false"],
                        help="Override offload_model (default: false for 480p, true for 720p)")
    args = parser.parse_args()
    if args.dataset_root is None and (args.captions_dir is None or args.images_dir is None):
        parser.error("--dataset-root is required unless both --captions-dir and --images-dir are given")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    # Resolution config
    res_cfg = RESOLUTION_CONFIGS[args.resolution]
    if args.offload_model is not None:
        res_cfg["offload_model"] = args.offload_model == "true"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else res_cfg["ckpt_dir"]
    output_dir = Path(args.output_dir) if args.output_dir else res_cfg["output_dir"]
    captions_dir = Path(args.captions_dir) if args.captions_dir else Path(args.dataset_root) / "captions"
    images_dir = Path(args.images_dir) if args.images_dir else Path(args.dataset_root) / "I2V" / "i2v_frames"

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
            # Must have matching image
            if not (images_dir / f"{stem}.png").exists():
                continue
            mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if mp4_path.exists() and json_path.exists():
                continue
            todo.append(stem)

    logging.info(f"GPU {args.gpu} | Resolution: {args.resolution} | Total: {len(todo)} videos to {'regenerate' if regen_info else 'generate'}")

    if not todo:
        logging.info("Nothing to generate. All done!")
        return

    # Collect environment info
    env_info = get_environment_info(res_cfg)

    # Load pipeline
    logging.info(f"Loading Wan 2.1 I2V pipeline ({args.resolution})...")
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

    cfg = WAN_CONFIGS["i2v-14B"]
    max_area = MAX_AREA_CONFIGS[res_cfg["size_str"]]

    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

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

        logging.info(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | {args.resolution} | img={image_path.name}")

        t0 = time.time()
        writing = False
        try:
            # Load source image
            img = Image.open(str(image_path)).convert("RGB")

            video = wan_i2v.generate(
                prompt,
                img,
                n_prompt=negative_prompt,
                max_area=max_area,
                frame_num=NUM_FRAMES,
                shift=res_cfg["sample_shift"],
                sample_solver=SAMPLE_SOLVER,
                sampling_steps=SAMPLE_STEPS,
                guide_scale=GUIDE_SCALE,
                seed=seed,
                offload_model=res_cfg["offload_model"],
            )

            # Save video
            writing = True
            save_video(video, str(mp4_path), fps=FRAME_RATE)

            # Extract video dimensions from the generated tensor (last two dims are H, W)
            height_px, width_px = int(video.shape[-2]), int(video.shape[-1])
            orientation = "landscape" if width_px >= height_px else "portrait"

            # Save metadata
            gen_time = time.time() - t0
            metadata = build_metadata(
                stem + ".json", prompt, negative_prompt, seed,
                width_px, height_px, orientation,
                stem + ".png",
                args.resolution, res_cfg, gen_time, env_info,
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
