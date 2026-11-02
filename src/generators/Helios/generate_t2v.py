#!/usr/bin/env python3
"""
Helios Text-to-Video batch generation script for SynthForensics benchmark.

Usage:
    # Generate all 1363 videos with Distilled on GPU 0
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --variant distilled --enable-compile

    # Split across GPUs
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --variant distilled --start-index 0 --end-index 681

    # Regenerate rejected videos from validation CSV
    python generate_t2v.py --captions-dir /path/to/SynthForensics/captions --gpu 0 --variant distilled --regenerate validation_results.csv
"""

import argparse
import csv
import json
import os
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

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import torch
from diffusers.utils import export_to_video

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "helios_t2v_raw"
OUTPUT_SUFFIX = "_helios_t2v"  # released file names: {id}_helios_t2v.mp4 / .json

# ─── Generation defaults ────────────────────────────────────────────────────
HEIGHT = 384
WIDTH = 640
NUM_FRAMES = 99
FPS = 24
IMAGE_NOISE_SIGMA_MIN = 0.111
IMAGE_NOISE_SIGMA_MAX = 0.135
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# Per-variant configs
MODEL_VARIANTS = {
    "distilled": {
        "model_dir": "Helios-Distilled",
        "guidance_scale": 1.0,
        "num_inference_steps": 50,
        "is_enable_stage2": True,
        "pyramid_num_inference_steps_list": [2, 2, 2],
        "is_amplify_first_chunk": True,
        "use_zero_init": False,
        "zero_steps": 1,
    },
}

# Caption fields combined into the final prompt, in the order they appear
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
        if value.endswith("."):
            value = value[:-1]
        if field in PREFIXED_FIELDS:
            value = PREFIXED_FIELDS[field].format(value.lower())
        parts.append(value)
    return ". ".join(parts) + "."


def generate_seed() -> int:
    return int.from_bytes(os.urandom(4), "big") % (2**31)


def clean_stem(name: str) -> str:
    name = name.replace(".mp4", "").replace(".json", "")
    name = name.replace(OUTPUT_SUFFIX, "")
    return name


def parse_regenerate_csv(csv_path: Path) -> dict:
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


def get_caption_files(start_index, end_index, captions_dir: Path) -> list[Path]:
    all_files = sorted(captions_dir.glob("*.json"))
    start = start_index if start_index is not None else 0
    end = end_index if end_index is not None else len(all_files)
    return all_files[start:end]


def _get_pkg_version(pkg_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def get_environment_info(variant: str) -> dict:
    vcfg = MODEL_VARIANTS[variant]
    model_dir = vcfg["model_dir"]
    return {
        "repo_url": "https://github.com/PKU-YuanGroup/Helios",
        "repo_commit": "5cb7ce9",
        "paper_url": "https://arxiv.org/abs/2603.04379",
        "model_id": model_dir,
        "model_url": f"https://huggingface.co/BestWishYsh/{model_dir}",
        "model_hf_commit": "1999182614cb08d3bdcc46b9827504af2914b87b",
        "model_parameter_count": "14B",
        "models": {
            "transformer": {
                "path": "Helios-Distilled/transformer",
                "files": 6,
                "sha256_shard1": "7be354f061682a7b2ddc96858ae488ea1944fa315109f1b3835b14286cd17df4",
            },
            "vae": {
                "path": "Helios-Distilled/vae",
                "sha256": "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793",
            },
            "text_encoder": {
                "path": "Helios-Distilled/text_encoder",
                "files": 5,
                "sha256_shard1": "c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99",
            },
        },
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "key_deps": {
            "torch": torch.__version__,
            "torchaudio": _get_pkg_version("torchaudio"),
            "torchvision": _get_pkg_version("torchvision"),
            "diffusers": _get_pkg_version("diffusers"),
            "transformers": _get_pkg_version("transformers"),
            "flash_attn": _get_pkg_version("flash_attn"),
            "triton": _get_pkg_version("triton"),
            "accelerate": _get_pkg_version("accelerate"),
            "xformers": _get_pkg_version("xformers"),
        },
    }


def build_metadata(
    caption_name: str, prompt: str, negative_prompt: str, seed: int,
    gen_time: float, variant: str, vcfg: dict, env_info: dict,
    enable_compile: bool = False,
) -> dict:
    return {
        "generator": "Helios",
        "task": "t2v-14B",
        "mode": variant,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "guidance_scale": vcfg["guidance_scale"],
        "num_inference_steps": vcfg["num_inference_steps"],
        "is_enable_stage2": vcfg["is_enable_stage2"],
        "pyramid_num_inference_steps_list": vcfg["pyramid_num_inference_steps_list"],
        "is_amplify_first_chunk": vcfg["is_amplify_first_chunk"],
        "use_zero_init": vcfg["use_zero_init"],
        "zero_steps": vcfg["zero_steps"],
        "enable_compile": enable_compile,
        "image_noise_sigma_min": IMAGE_NOISE_SIGMA_MIN,
        "image_noise_sigma_max": IMAGE_NOISE_SIGMA_MAX,
        "dtype": "bfloat16",
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


def load_pipeline(model_path: str, enable_compile: bool = False):
    """Load Helios pipeline once with all optimizations."""
    from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
    from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
    from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
    from diffusers.models import AutoencoderKLWan

    weight_dtype = torch.bfloat16

    transformer = HeliosTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=weight_dtype,
    )

    if not enable_compile:
        # Manual kernel optimizations (faster startup, no compilation overhead)
        from helios.modules.helios_kernels import (
            replace_all_norms_with_flash_norms,
            replace_rmsnorm_with_fp32,
            replace_rope_with_flash_rope,
        )
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()

    # Set attention backend based on GPU architecture
    cuda_major = torch.cuda.get_device_capability()[0]
    if cuda_major >= 9:
        try:
            transformer.set_attention_backend("_flash_3_hub")
        except Exception:
            transformer.set_attention_backend("flash_hub")
    else:
        transformer.set_attention_backend("flash_hub")

    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(model_path, subfolder="scheduler")

    pipe = HeliosPipeline.from_pretrained(
        model_path, transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=weight_dtype,
    )

    if enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    pipe = pipe.to("cuda")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Helios T2V batch generation for SynthForensics")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--variant", type=str, default="distilled", choices=["distilled"],
                        help="Model variant (default: distilled)")
    parser.add_argument("--captions-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None, help="Override model path (default: auto from variant)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir (default: auto from variant)")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--regenerate", type=str, default=None,
                        help="Path to validation CSV (columns: video,prompt_override,extra_negative_prompt)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enable-compile", action="store_true",
                        help="Use torch.compile for faster generation (slower first video)")
    args = parser.parse_args()

    vcfg = MODEL_VARIANTS[args.variant]
    model_path = args.model_path or str(DEFAULT_MODELS_DIR / vcfg["model_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR.parent / f"helios_{args.variant}_t2v_raw"
    captions_dir = Path(args.captions_dir)

    if not captions_dir.exists():
        print(f"Error: captions directory not found: {captions_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse regeneration CSV
    regen_info = {}
    if args.regenerate:
        regen_csv = Path(args.regenerate)
        if not regen_csv.exists():
            print(f"Error: regeneration CSV not found: {regen_csv}")
            sys.exit(1)
        regen_info = parse_regenerate_csv(regen_csv)
        print(f"Regeneration mode: {len(regen_info)} videos from {regen_csv.name}")

    # Build todo list
    if regen_info:
        todo = []
        for stem in sorted(regen_info.keys()):
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if not json_path.exists():
                print(f"Warning: metadata not found for {stem}, skipping")
                continue
            todo.append(stem)
    else:
        caption_files = get_caption_files(args.start_index, args.end_index, captions_dir)
        if not caption_files:
            print("No caption files to process.")
            return

        todo = []
        for cf in caption_files:
            stem = cf.stem
            mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
            json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"
            if mp4_path.exists() and json_path.exists():
                continue
            todo.append(stem)

    print(f"GPU {args.gpu} | Total: {len(todo)} videos to {'regenerate' if regen_info else 'generate'}")

    if not todo:
        print("Nothing to generate. All done!")
        return

    # Load pipeline
    print(f"Loading Helios {args.variant.capitalize()} pipeline (compile={'ON' if args.enable_compile else 'OFF'})...")
    pipe = load_pipeline(model_path, enable_compile=args.enable_compile)
    env_info = get_environment_info(args.variant)
    is_regen = bool(regen_info)

    print(f"Pipeline loaded. Starting generation of {len(todo)} videos...\n")

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

        print(f"[{i+1}/{len(todo)}] {stem} | seed={seed}")

        t0 = time.time()
        writing = False
        try:
            with torch.no_grad():
              output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=vcfg["num_inference_steps"],
                guidance_scale=vcfg["guidance_scale"],
                generator=torch.Generator(device="cuda").manual_seed(seed),
                history_sizes=[16, 2, 1],
                num_latent_frames_per_chunk=9,
                keep_first_frame=True,
                is_enable_stage2=vcfg["is_enable_stage2"],
                pyramid_num_inference_steps_list=vcfg["pyramid_num_inference_steps_list"],
                is_amplify_first_chunk=vcfg["is_amplify_first_chunk"],
                use_zero_init=vcfg["use_zero_init"],
                zero_steps=vcfg["zero_steps"],
                image_noise_sigma_min=IMAGE_NOISE_SIGMA_MIN,
                image_noise_sigma_max=IMAGE_NOISE_SIGMA_MAX,
              ).frames[0]

            writing = True
            export_to_video(output, str(mp4_path), fps=FPS)

            gen_time = time.time() - t0
            metadata = build_metadata(stem + ".json", prompt, negative_prompt, seed, gen_time, args.variant, vcfg, env_info, args.enable_compile)
            json_path.write_text(json.dumps(metadata, indent=2))

            print(f"  Done in {gen_time:.0f}s -> {mp4_path.name}")

        except Exception as e:
            print(f"  ERROR generating {stem}: {e}")
            import traceback
            traceback.print_exc()
            if writing:
                for p in (mp4_path, json_path):
                    if p.exists():
                        p.unlink()
            continue

    print(f"\nGeneration complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
