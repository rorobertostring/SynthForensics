#!/usr/bin/env python3
"""
daVinci-MagiHuman Text-to-Video batch generation script for SynthForensics benchmark.

NOTE: This generator requires torchrun (distributed launch).

Usage:
    # Generate all 1363 videos on GPU 0
    CUDA_VISIBLE_DEVICES=0 CPU_OFFLOAD=true SR2_1080=true KEEP_SR_ON_GPU=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=1 --master_port=29502 \
      generate_t2v.py --config-load-path example/distill_sr_1080p/config.json \
      --captions-dir /path/to/SynthForensics/captions

    # Split across two GPUs (use different master_port!)
    CUDA_VISIBLE_DEVICES=0 ... torchrun --master_port=29500 generate_t2v.py \
      --config-load-path example/distill_sr_1080p/config.json --captions-dir /path/to/SynthForensics/captions \
      --start-index 0 --end-index 681
    CUDA_VISIBLE_DEVICES=1 ... torchrun --master_port=29502 generate_t2v.py \
      --config-load-path example/distill_sr_1080p/config.json --captions-dir /path/to/SynthForensics/captions \
      --start-index 681 --end-index 1363

    # Regenerate rejected videos from validation CSV
    CUDA_VISIBLE_DEVICES=0 ... torchrun --master_port=29502 generate_t2v.py \
      --config-load-path example/distill_sr_1080p/config.json --captions-dir /path/to/SynthForensics/captions \
      --regenerate validation_results.csv
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "davinci_t2v_raw"
OUTPUT_SUFFIX = "_daVinci-MagiHuman_t2v"  # released file names: {id}_daVinci-MagiHuman_t2v.mp4 / .json

# ─── Generation defaults ─────────────────────────────────────────────────────
SECONDS = 5
FPS = 25

# Orientations: landscape or portrait (base → SR)
ORIENTATIONS = {
    "landscape": {"br_width": 448, "br_height": 256, "sr_width": 1920, "sr_height": 1088},
    "portrait":  {"br_width": 256, "br_height": 448, "sr_width": 1088, "sr_height": 1920},
}

# Hardcoded negative prompt from video_generate.py:222-224 (not configurable per-video)
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards, "
    "low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, "
    "compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, "
    "room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, "
    "thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, "
    "looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, "
    "disorganized, messy, cheap sound, "
    "emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, "
    "undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, "
    "exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, "
    "stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky "
    "voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, "
    "coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural "
    "pauses, awkward silence, choppy, disjointed, multiple speakers, two voices, background talking, out of tune, "
    "off-key, autotune artifacts"
)

# Prompt fields used from captions (6 of 8: excluded short_description and camera_movement)
PROMPT_FIELDS = [
    "dense_description",
    "background",
    "shot_type",
    "lighting",
    "atmosphere",
    "style",
]


def _strip_prefix(text: str, prefixes: list[str]) -> str:
    """Remove known verbose prefixes from caption fields (case-insensitive)."""
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def build_prompt(caption: dict) -> str:
    """
    Build a daVinci-MagiHuman prompt from structured caption fields.

    Uses 6 of 8 caption fields:
    - dense_description: main character/scene description
    - background: environment
    - shot_type: camera framing (prefixed with "Captured in a static")
    - lighting: lighting description
    - atmosphere: mood/atmosphere
    - style: visual style

    Excluded:
    - short_description: redundant with dense_description
    - camera_movement: always static for daVinci

    Background Sound section appended with fixed "No prominent background sound".
    No Dialogue section (no transcription data available).
    """
    dense = caption.get("dense_description", "").strip().rstrip(".")
    background = caption.get("background", "").strip().rstrip(".")
    shot_type = caption.get("shot_type", "medium shot").strip().rstrip(".")
    lighting = caption.get("lighting", "").strip().rstrip(".")
    atmosphere = caption.get("atmosphere", "").strip().rstrip(".")
    style = caption.get("style", "").strip().rstrip(".")

    # Clean verbose prefixes that some captions (DFD) include in their fields
    shot_type = _strip_prefix(shot_type, ["the shot type is ", "the shot is "])
    style = _strip_prefix(style, [
        "the video style appears to be ", "the video style is ",
        "the style appears to be ", "the style is ",
    ])
    lighting = _strip_prefix(lighting, ["the lighting is ", "lighting is "])
    atmosphere = _strip_prefix(atmosphere, ["the atmosphere is ", "atmosphere is "])

    parts = []
    if dense:
        parts.append(dense)
    if background:
        parts.append(background)
    if shot_type:
        parts.append(f"Captured in a static {shot_type.lower()}")
    if lighting:
        parts.append(f"The lighting is {lighting[0].lower() + lighting[1:]}")
    if atmosphere:
        parts.append(f"The atmosphere is {atmosphere[0].lower() + atmosphere[1:]}")
    if style:
        parts.append(f"The style is {style.lower()}")

    main_body = ". ".join(parts) + "."

    return f"{main_body}\n\nBackground Sound:\n<No prominent background sound>"


def pick_orientation(seed: int) -> str:
    """Deterministically pick orientation from seed for reproducibility."""
    rng = random.Random(seed)
    return rng.choice(["landscape", "portrait"])


def generate_seed() -> int:
    """Generate a random seed using os.urandom (not affected by set_random_seed)."""
    return int.from_bytes(os.urandom(4), "big") % (2**31)


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


def get_caption_files(start_index, end_index, captions_dir: Path) -> list[Path]:
    """Get the list of caption files to process based on index range."""
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


def get_environment_info(config) -> dict:
    """Collect reproducibility metadata about the environment."""
    return {
        "repo_url": "https://github.com/GAIR-NLP/daVinci-MagiHuman",
        "repo_commit": "f9764ce",
        "model_id": "daVinci-MagiHuman-15B",
        "model_url": "https://huggingface.co/GAIR/daVinci-MagiHuman",
        "model_hf_commit": "7fe95e50c11bd92bdadf94cd51dbec18427f8e0c",
        "model_parameter_count": "15.3B",
        "models": {
            "distill_dit": {
                "path": "davinci-magihuman/distill",
                "files": 13,
                "total_size_bytes": 61197676608,
                "sha256_shard1": "9db62d550ae1cd3314387aaa725a2b4326d01accd102dccabfad8e3ed7b55c95",
            },
            "sr_1080p_dit": {
                "path": "davinci-magihuman/1080p_sr",
                "files": 13,
                "total_size_bytes": 61197676608,
                "sha256_shard1": "46bf7f311131721e9b21f3b17de0538231e155758d4de4f6df37eed3153fad1f",
            },
            "turbo_vae": {
                "path": "davinci-magihuman/turbo_vae/checkpoint-340000.ckpt",
                "sha256": "90a676de37f35f3acaca69bffe1955c9d94cfe6acf767b340e1b10f7e06938f7",
            },
            "video_vae": {
                "path": "Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
                "sha256": "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
            },
            "text_encoder": {
                "path": "t5gemma-9b-9b-ul2",
                "files": 9,
                "sha256_shard1": "31263111cf868589497869bceab00e07fe3caacb38d2fae427c34b1cf58aa335",
            },
            "audio_vae": {
                "path": "stable-audio-open-1.0/model.safetensors",
                "sha256": "7b20458a071231aaf32613b6fbc7945f28f34dbba4f295bb49bad56f5f66b57e",
            },
        },
        "distill": "distill" in str(config.engine_config.load),
        "use_sr_model": config.evaluation_config.use_sr_model,
        "use_turbo_vae": config.evaluation_config.use_turbo_vae,
        "cpu_offload": os.environ.get("CPU_OFFLOAD", "false"),
        "sr2_1080": os.environ.get("SR2_1080", "false"),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "key_deps": {
            "torch": torch.__version__,
            "torchaudio": _get_pkg_version("torchaudio"),
            "torchvision": _get_pkg_version("torchvision"),
            "flash_attn": _get_pkg_version("flash_attn"),
            "magi_compiler": "0.0.0 (installed from source: github.com/SandAI-org/MagiCompiler)",
            "magi_attention": _get_pkg_version("magi_attention"),
            "imageio": _get_pkg_version("imageio"),
            "transformers": _get_pkg_version("transformers"),
        },
    }


def build_metadata(
    caption_name: str, prompt: str, seed: int, orientation: str, dims: dict,
    gen_time: float, config, env_info: dict,
) -> dict:
    """Build the full metadata dict for reproducibility."""
    eval_cfg = config.evaluation_config
    return {
        "generator": "daVinci-MagiHuman",
        "task": "t2v-15B",
        "mode": "distill" if "distill" in str(config.engine_config.load) else "base",
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": seed,
        "orientation": orientation,
        "width": dims["sr_width"],
        "height": dims["sr_height"],
        "base_width": dims["br_width"],
        "base_height": dims["br_height"],
        "seconds": SECONDS,
        "fps": FPS,
        "num_frames": SECONDS * FPS + 1,
        "num_inference_steps": eval_cfg.num_inference_steps,
        "sr_num_inference_steps": eval_cfg.sr_num_inference_steps,
        "cfg_number": eval_cfg.cfg_number,
        "sr_cfg_number": eval_cfg.sr_cfg_number,
        "video_txt_guidance_scale": eval_cfg.video_txt_guidance_scale,
        "sr_video_txt_guidance_scale": eval_cfg.sr_video_txt_guidance_scale,
        "shift": eval_cfg.shift,
        "noise_value": eval_cfg.noise_value,
        "use_cfg_trick": eval_cfg.use_cfg_trick,
        "cfg_trick_start_frame": eval_cfg.cfg_trick_start_frame,
        "cfg_trick_value": eval_cfg.cfg_trick_value,
        "dtype": "bfloat16",
        "source_caption": caption_name,
        "generation_time_seconds": round(gen_time, 1),
        "environment": env_info,
    }


def parse_batch_args():
    """Parse batch-specific arguments (coexists with pydantic config args)."""
    parser = argparse.ArgumentParser(
        description="daVinci-MagiHuman T2V batch generation for SynthForensics",
        allow_abbrev=False,
    )
    parser.add_argument("--captions-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--regenerate", type=str, default=None,
                        help="Path to validation CSV (columns: video,prompt_override,extra_negative_prompt)")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific seed for all videos")
    args, _ = parser.parse_known_args()
    return args


@torch.inference_mode()
def main():
    args = parse_batch_args()

    # Resolve paths
    captions_dir = Path(args.captions_dir)
    output_dir = Path(args.output_dir)

    if not captions_dir.exists():
        print(f"Error: captions directory not found: {captions_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse regeneration CSV if provided
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
                continue  # Already generated, skip
            todo.append(stem)

    print(f"Total: {len(todo)} videos to {'regenerate' if regen_info else 'generate'}")

    if not todo:
        print("Nothing to generate. All done!")
        return

    # ─── Load pipeline (heavy, do once) ──────────────────────────────────────
    print("Loading daVinci-MagiHuman pipeline...")
    from inference.common import parse_config
    from inference.model.dit import get_dit
    from inference.pipeline.pipeline import MagiPipeline

    config = parse_config()
    model = get_dit(config.arch_config, config.engine_config)
    pipeline = MagiPipeline(model, config.evaluation_config)

    env_info = get_environment_info(config)
    is_regen = bool(regen_info)

    print(f"Pipeline loaded. Starting generation of {len(todo)} videos...\n")

    for i, stem in enumerate(todo):
        mp4_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.mp4"
        json_path = output_dir / f"{stem}{OUTPUT_SUFFIX}.json"

        if is_regen:
            # Regeneration: read prompt from existing metadata JSON
            old_meta = json.loads(json_path.read_text())
            prompt = old_meta["prompt"]

            # Apply prompt modification if provided
            info = regen_info[stem]
            if info["prompt_override"]:
                prompt = info["prompt_override"]

            # Note: extra_negative_prompt is ignored for daVinci because
            # the negative prompt is hardcoded in video_generate.py (not configurable)
            if info.get("extra_negative_prompt"):
                print(f"  Warning: extra_negative_prompt ignored for {stem} "
                      "(daVinci negative prompt is hardcoded)")

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

            if args.seed is not None:
                seed = args.seed
            else:
                seed = generate_seed()

        orientation = pick_orientation(seed)
        dims = ORIENTATIONS[orientation]

        print(f"[{i+1}/{len(todo)}] {stem} | seed={seed} | {orientation} ({dims['sr_width']}x{dims['sr_height']})")

        t0 = time.time()
        writing = False
        try:
            # Generate video — run_offline saves to a path with resolution suffix
            generated_path = pipeline.run_offline(
                prompt=prompt,
                image=None,     # T2V mode
                audio=None,
                save_path_prefix=str(output_dir / f"_tmp_{stem}"),
                seed=seed,
                seconds=SECONDS,
                br_width=dims["br_width"],
                br_height=dims["br_height"],
                sr_width=dims["sr_width"],
                sr_height=dims["sr_height"],
            )

            # Rename from _tmp_{stem}_5s_448x256_1920x1088.mp4 → {stem}.mp4
            generated_file = Path(generated_path)
            if generated_file.exists():
                writing = True
                shutil.move(str(generated_file), str(mp4_path))
            else:
                # Fallback: find the temp file by glob
                candidates = list(output_dir.glob(f"_tmp_{stem}_*.mp4"))
                if candidates:
                    writing = True
                    shutil.move(str(candidates[0]), str(mp4_path))
                else:
                    raise FileNotFoundError(f"Generated video not found for {stem}")

            gen_time = time.time() - t0

            # Save metadata
            metadata = build_metadata(stem + ".json", prompt, seed, orientation, dims, gen_time, config, env_info)
            json_path.write_text(json.dumps(metadata, indent=2))

            print(f"  Done in {gen_time:.0f}s -> {mp4_path.name}")

        except Exception as e:
            print(f"  ERROR generating {stem}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up partial files
            if writing:
                for p in (mp4_path, json_path):
                    if p.exists():
                        p.unlink()
            # Clean up temp files
            for tmp in output_dir.glob(f"_tmp_{stem}*"):
                tmp.unlink()
            continue

    print(f"\nGeneration complete. Output: {output_dir}")


if __name__ == "__main__":
    from inference.infra import initialize_infra
    initialize_infra()
    main()
