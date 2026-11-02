#!/usr/bin/env python3
"""
MAGI-1 batch inference script.
Can be launched directly with python — torchrun is invoked automatically.

Examples:
  # Text-to-video
  python generate.py --input_dir captions/ --output_dir output/ --mode t2v

  # Image-to-video (captions and images in same folder)
  python generate.py --input_dir captions/ --output_dir output/ --mode i2v

  # Image-to-video (captions and images in separate folders)
  python generate.py --input_dir captions/ --image_dir images/ --output_dir output/ --mode i2v

  # Regenerate specific videos with a new random seed
  python generate.py --input_dir captions/ --output_dir output/ --regenerate to_regen.txt

  # Override config params
  python generate.py --input_dir captions/ --gpu 2 --num_frames 120 --seed -1

  # Process a specific slice (e.g. to split the work across GPUs)
  python generate.py --input_dir captions/ --output_dir output/ --mode i2v --gpu 1 --start-index 100 --end-index 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Auto-relaunch via torchrun if called directly (LOCAL_RANK not set)
# ---------------------------------------------------------------------------

def _relaunch_via_torchrun() -> None:
    """Re-exec this script under torchrun with the correct environment."""
    args      = parse_args()
    conda_env = os.environ.get(
        "CONDA_PREFIX",
        str(Path(sys.executable).parent.parent),
    )
    env = os.environ.copy()
    env["PATH"]         = f"{conda_env}/bin:{env.get('PATH', '')}"
    # Only override CUDA_HOME when inside conda
    if "CONDA_PREFIX" in os.environ:
        env["CUDA_HOME"] = conda_env
    env["LIBRARY_PATH"] = (
        f"/usr/lib/x86_64-linux-gnu"
        f":{conda_env}/targets/x86_64-linux/lib/stubs"
        f":{env.get('LIBRARY_PATH', '')}"
    )
    env["CUDA_DEVICE_MAX_CONNECTIONS"]  = "1"
    env["NCCL_ALGO"]                    = "^NVLS"
    env["PAD_HQ"]                       = "1"
    env["PAD_DURATION"]                 = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"]      = "expandable_segments:True"
    env["OFFLOAD_T5_CACHE"]             = "true"
    env["OFFLOAD_VAE_CACHE"]            = "true"
    env["TORCH_CUDA_ARCH_LIST"]         = "8.0;8.9;9.0;10.0"
    env["CUDA_VISIBLE_DEVICES"]         = str(args.gpu)

    magi_root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = f"{magi_root}:{env.get('PYTHONPATH', '')}"

    import shutil
    torchrun = (
        shutil.which("torchrun")                          # found in PATH
        or str(Path(sys.executable).parent / "torchrun") # sibling of python (conda)
        or str(Path(conda_env) / "bin" / "torchrun")     # explicit conda fallback
    )
    cmd = [
        torchrun,
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint=localhost:{args.rdzv_port}",
        "--nnodes=1",
        "--nproc_per_node=1",
        str(Path(__file__).resolve()),
    ] + sys.argv[1:]  # forward all original args

    log_file = f"log_{time.strftime('%Y-%m-%d_%H:%M:%S')}.log"
    print(f"[MAGI] Relaunching via torchrun | GPU={args.gpu} | log → {log_file}", flush=True)
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
    sys.exit(proc.wait())


if "LOCAL_RANK" not in os.environ:
    # Not inside a torchrun worker — relaunch
    # We need parse_args available, so we define it before this block runs.
    # Use a deferred call inside __main__ guard instead.
    pass


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
OUTPUT_SUFFIX = {"t2v": "_magi-1_t2v", "i2v": "_magi-1_i2v"}  # released file names: {id}_magi-1_{mode}.mp4 / .json


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt_from_json(json_data: dict) -> str:
    ordered_keys = [
        "short_description",
        "dense_description",
        "background",
        "style",
        "shot_type",
        "lighting",
        "atmosphere",
        "camera_movement",
    ]
    prompt_parts = []
    for key in ordered_keys:
        value = json_data.get(key, "").strip()
        if not value:
            continue
        if len(value.split()) < 4:
            prompt_parts.append(f"The {key.replace('_', ' ')} is {value.lower()}")
        else:
            prompt_parts.append(value)
    prompt = ". ".join(prompt_parts)
    while ".." in prompt:
        prompt = prompt.replace("..", ".")
    return prompt


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def get_env_metadata() -> dict:
    import platform
    env: dict = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    try:
        import flash_attn
        env["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or ""
        env["gpu_name"] = torch.cuda.get_device_name(0)
    env["repo_commit"] = "6ff822e74ded50611e81e1d0e115146b5c4dd2a5"
    return env


def _hf_cache_base(config) -> Path:
    """Return the root of the HuggingFace download cache for model weights."""
    base = Path(config.runtime_config.load).parent.parent  # e.g. weights/ckpt
    return base.parent / ".cache" / "huggingface" / "download" / base.name


def read_hf_commit(config) -> str | None:
    """Read the HuggingFace commit hash from the first available cache metadata file."""
    cache_base = _hf_cache_base(config)
    load_path = Path(config.runtime_config.load)
    base = load_path.parent.parent
    for meta in sorted((cache_base / load_path.relative_to(base)).rglob("*.metadata")):
        lines = meta.read_text().splitlines()
        if lines and lines[0].strip():
            return lines[0].strip()
    return None


def compute_model_sha256(config) -> dict:
    """
    Return SHA256 for each model weight file.
    Reads from HuggingFace download cache metadata (line 2) when available,
    falling back to full SHA256 computation over the actual file.
    """
    import hashlib
    rc = config.runtime_config
    load_path = Path(rc.load)
    base = load_path.parent.parent  # e.g. weights/ckpt
    cache_base = _hf_cache_base(config)

    files_to_hash: list[Path] = []
    files_to_hash += sorted(load_path.rglob("*.safetensors"))
    files_to_hash += sorted(Path(rc.vae_pretrained).glob("*.safetensors"))
    files_to_hash += sorted(Path(rc.t5_pretrained).rglob("pytorch_model*.bin"))

    sha256_dict: dict = {}
    for fpath in files_to_hash:
        rel = fpath.relative_to(base)
        cache_meta = cache_base / f"{rel}.metadata"
        if cache_meta.exists():
            sha256_dict[str(rel)] = cache_meta.read_text().splitlines()[1].strip()
        else:
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            sha256_dict[str(rel)] = h.hexdigest()
    return sha256_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image(json_path: Path, image_dir: Path | None = None) -> Path | None:
    """
    Given 'foo.json', looks for 'foo.png', 'foo.jpg', etc.
    If image_dir is given, searches there; otherwise looks alongside the JSON.
    """
    image_id = json_path.stem
    search_dir = image_dir if image_dir is not None else json_path.parent
    for ext in IMAGE_EXTENSIONS:
        candidate = search_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def prepare_i2v_image(
    src_path: Path,
    target_w: int,
    target_h: int,
    top_bias: float = 0.25,
) -> tuple[str, bool]:
    """
    Ensure src_path matches the target_w:target_h aspect ratio without stretching.

    If the image is already at the correct ratio (within 1%), returns the original
    path untouched. Otherwise crops with ffmpeg and writes to a NamedTemporaryFile.

    top_bias controls where the vertical crop is anchored:
      0.0 → keep the very top (all excess removed from the bottom)
      0.5 → centre crop
      1.0 → keep the very bottom (all excess removed from the top)
    Default 0.25 biases toward the top.

    Returns (path_to_use: str, is_temp: bool).
    The caller must delete the file when is_temp is True.
    """
    import ffmpeg as _ffmpeg

    target_ratio = target_w / target_h

    probe = _ffmpeg.probe(str(src_path))
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    src_ratio = w / h

    if abs(src_ratio - target_ratio) / target_ratio < 0.01:
        return str(src_path), False  # already correct ratio — no crop needed

    if src_ratio > target_ratio:
        # Image is wider than target: crop left/right symmetrically
        crop_w = int(h * target_ratio)
        crop_w -= crop_w % 2
        crop_h = h
        x = (w - crop_w) // 2
        y = 0
    else:
        # Image is narrower than target: crop top/bottom with top_bias
        crop_w = w
        crop_h = int(w / target_ratio)
        crop_h -= crop_h % 2
        excess = h - crop_h
        y = int(excess * top_bias)
        y -= y % 2
        x = 0

    tmp = tempfile.NamedTemporaryFile(suffix=src_path.suffix, delete=False)
    tmp.close()
    (
        _ffmpeg.input(str(src_path))
        .filter("crop", crop_w, crop_h, x, y)
        .output(tmp.name, vframes=1)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    return tmp.name, True


def infer_model_version(load_path: str) -> str:
    for tag, name in [
        ("24B_distill", "MAGI-1-24B-distill"),
        ("24B_base",    "MAGI-1-24B-base"),
        ("4.5B_distill","MAGI-1-4.5B-distill"),
        ("4.5B_base",   "MAGI-1-4.5B-base"),
    ]:
        if tag in load_path:
            return name
    return "MAGI-1"


def _serialize(obj):
    """Recursively convert a config object to a JSON-safe dict."""
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj


def build_metadata(
    config,
    prompt: str,
    source_caption: str,
    generation_time: float,
    mode: str,
    env_meta: dict,
    image_path: Path | None = None,
) -> dict:
    rc = config.runtime_config
    meta: dict = {
        "generator": "MAGI-1",
        "model_version": infer_model_version(rc.load),
        "mode": mode,
        "prompt": prompt,
        "source_caption": source_caption,
    }
    if image_path is not None:
        meta["source_image"] = image_path.name
    meta["generation_time_seconds"] = round(generation_time, 1)
    meta["config"] = {
        "model_config":   _serialize(config.model_config),
        "runtime_config": _serialize(config.runtime_config),
        "engine_config":  _serialize(config.engine_config),
    }
    meta["environment"] = env_meta
    return meta


def apply_overrides(config, args: argparse.Namespace) -> None:
    rc = config.runtime_config
    ec = config.engine_config
    for obj, attr, val in [
        (rc, "num_frames",   args.num_frames),
        (rc, "video_size_h", args.video_size_h),
        (rc, "video_size_w", args.video_size_w),
        (rc, "num_steps",    args.num_steps),
        (rc, "fps",          args.fps),
        (rc, "window_size",  args.window_size),
        (rc, "chunk_width",  args.chunk_width),
        (rc, "cfg_number",   args.cfg_number),
        (rc, "t5_device",    args.t5_device),
        (ec, "kv_offload",   args.kv_offload),
        (ec, "fp8_quant",    args.fp8_quant),
    ]:
        if val is not None:
            setattr(obj, attr, val)


def is_rank0() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def log(msg: str) -> None:
    if is_rank0():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAGI-1 batch video generation")

    # I/O
    parser.add_argument("--input_dir",   required=True,
                        help="Folder containing caption JSON files")
    parser.add_argument("--image_dir",   default=None,
                        help="(i2v only) Folder containing input images; if omitted, images are "
                             "looked up alongside each JSON file. Image names must share the same "
                             "prefix as the JSON (e.g. '000.png' for '000.json').")
    parser.add_argument("--crop_top_bias", type=float, default=0.25, metavar="BIAS",
                        help="(i2v only) Vertical crop anchor when the image aspect ratio does not "
                             "match the target: 0.0=keep top, 0.5=centre, 1.0=keep bottom "
                             "(default: 0.25 — biased toward top to preserve faces)")
    parser.add_argument("--output_dir",  default="output",
                        help="Destination folder for videos and metadata (default: output/)")
    parser.add_argument("--mode",        choices=["t2v", "i2v"], default="t2v",
                        help="Generation mode: text-to-video or image-to-video")
    parser.add_argument("--config_file", default="example/24B/24B_distill_config_1gpu.json",
                        help="MAGI config JSON (default: example/24B/24B_distill_config_1gpu.json)")
    parser.add_argument("--regenerate",  default=None, metavar="CSV_FILE",
                        help="CSV file (columns: video, prompt_override, ...) listing videos to regenerate "
                             "with a new random seed; if 'prompt_override' is non-empty for a row, that "
                             "prompt overrides the one built from the caption JSON")

    # Launcher options (used only when relaunching via torchrun)
    parser.add_argument("--gpu",         type=int, default=0,
                        help="CUDA device index to use (default: 0)")
    parser.add_argument("--rdzv_port",   type=int, default=6009,
                        help="torchrun rendezvous port (default: 6009)")
    parser.add_argument("--start-index", type=int, default=None, dest="start_index",
                        help="First video index to process (inclusive, 0-based); used to split "
                             "the work across GPUs")
    parser.add_argument("--end-index",   type=int, default=None, dest="end_index",
                        help="Last video index to process (exclusive); used to split the work "
                             "across GPUs")

    # runtime_config overrides
    parser.add_argument("--num_frames",   type=int,   default=None)
    parser.add_argument("--video_size_h", type=int,   default=None)
    parser.add_argument("--video_size_w", type=int,   default=None)
    parser.add_argument("--num_steps",    type=int,   default=None)
    parser.add_argument("--fps",          type=int,   default=None)
    parser.add_argument("--seed",         type=int,   default=None,
                        help="Fixed seed for all videos; -1 for random per video")
    parser.add_argument("--window_size",  type=int,   default=None)
    parser.add_argument("--chunk_width",  type=int,   default=None)
    parser.add_argument("--cfg_number",   type=int,   default=None)
    parser.add_argument("--t5_device",    type=str,   default=None)

    # engine_config overrides
    parser.add_argument("--kv_offload", default=None, action=argparse.BooleanOptionalAction,
                        help="Enable/disable KV offload (default: from config)")
    parser.add_argument("--fp8_quant",  default=None, action=argparse.BooleanOptionalAction,
                        help="Enable/disable FP8 quantization (default: from config)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from inference.pipeline import MagiPipeline
    from inference.common import set_random_seed

    # Patch get_dit and _t5 so both models are loaded only once and reused across
    # all generations, regardless of OFFLOAD_T5_CACHE / OFFLOAD_VAE_CACHE env vars.

    import inference.pipeline.pipeline as _pipeline_mod
    from inference.model.dit import get_dit as _get_dit_orig
    _dit_cache: list = []
    def _get_dit_cached(config):
        if not _dit_cache:
            _dit_cache.append(_get_dit_orig(config))
        return _dit_cache[0]
    _pipeline_mod.get_dit = _get_dit_cached

    import inference.pipeline.prompt_process as _pp_mod
    _orig_t5_fn = _pp_mod._t5
    _t5_cache: list = []
    def _t5_cached(model_cache_dir, model_device, model_max_length):
        if not _t5_cache:
            _t5_cache.append(_orig_t5_fn(model_cache_dir, model_device, model_max_length))
        return _t5_cache[0]
    _pp_mod._t5 = _t5_cached

    log(f"[MAGI] Loading pipeline from {args.config_file}")
    pipeline = MagiPipeline(args.config_file)
    apply_overrides(pipeline.config, args)

    log("[MAGI] Reading model metadata…")
    runtime_env = get_env_metadata()
    hf_commit = read_hf_commit(pipeline.config)
    env_meta: dict = {}
    if "repo_commit" in runtime_env:
        env_meta["repo_commit"] = runtime_env.pop("repo_commit")
    env_meta["model_id"] = "MAGI-1"
    env_meta["model_url"] = "https://huggingface.co/sand-ai/MAGI-1"
    if hf_commit:
        env_meta["model_hf_commit"] = hf_commit
    env_meta["model_sha256"] = compute_model_sha256(pipeline.config)
    env_meta.update(runtime_env)

    input_dir  = Path(args.input_dir)
    image_dir  = Path(args.image_dir) if args.image_dir else None
    output_dir = Path(args.output_dir)
    if is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    # Regenerate mode: restrict to listed stems, always use a new random seed.
    # CSV must have a 'video' column; 'prompt_override' column (if present and non-empty)
    # overrides the prompt built from the caption JSON.
    regenerate_set: set[str] = set()
    regenerate_prompts: dict[str, str] = {}  # stem -> prompt override (only if non-empty)
    if args.regenerate:
        with open(args.regenerate, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_val = row.get("video", "").strip()
                if not video_val:
                    continue
                stem = Path(video_val).stem.removesuffix(OUTPUT_SUFFIX[args.mode])  # accept "foo", "foo.mp4" and released names
                regenerate_set.add(stem)
                override = row.get("prompt_override", "").strip()
                if override:
                    regenerate_prompts[stem] = override
        json_files = [j for j in json_files if j.stem in regenerate_set]

    # Apply index slice
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index or 0
        end   = args.end_index if args.end_index is not None else len(json_files)
        json_files = json_files[start:end]

    if not json_files:
        log("[MAGI] No JSON files found to process.")
        return

    log(f"[MAGI] {len(json_files)} video(s) to generate | mode={args.mode}")

    for json_path in json_files:
        stem         = json_path.stem
        output_video = output_dir / f"{stem}{OUTPUT_SUFFIX[args.mode]}.mp4"
        output_meta  = output_dir / f"{stem}{OUTPUT_SUFFIX[args.mode]}.json"

        # Skip if already generated (both .mp4 and .json present), unless regenerating
        if stem not in regenerate_set and output_video.exists() and output_meta.exists():
            log(f"[MAGI] Skipping {stem} (already generated)")
            continue

        # Seed resolution
        if stem in regenerate_set or args.seed == -1:
            seed = secrets.randbelow(2**32)  # OS entropy, unaffected by the seeded PRNG state
        elif args.seed is not None:
            seed = args.seed
        else:
            seed = pipeline.config.runtime_config.seed

        pipeline.config.runtime_config.seed = seed
        set_random_seed(seed)

        # Build prompt (may be overridden by CSV prompt_override)
        with open(json_path) as f:
            caption_data = json.load(f)
        prompt = build_prompt_from_json(caption_data)
        if stem in regenerate_prompts:
            prompt = regenerate_prompts[stem]
            log(f"  [regen] prompt overridden from CSV")

        # Image path (i2v only)
        image_path: Path | None = None
        prepared_image_path: str | None = None
        prepared_is_temp: bool = False
        if args.mode == "i2v":
            image_path = find_image(json_path, image_dir)
            if image_path is None:
                search_desc = str(image_dir) if image_dir else str(json_path.parent)
                log(f"[WARN] No image found for {json_path.name} in {search_desc} — skipping.")
                continue
            prepared_image_path, prepared_is_temp = prepare_i2v_image(
                image_path,
                target_w=pipeline.config.runtime_config.video_size_w,
                target_h=pipeline.config.runtime_config.video_size_h,
                top_bias=args.crop_top_bias,
            )

        log(f"\n[{stem}] seed={seed}")
        log(f"  prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        t_start = time.time()
        try:
            if args.mode == "i2v":
                pipeline.run_image_to_video(
                    prompt=prompt,
                    image_path=prepared_image_path,
                    output_path=str(output_video),
                )
            else:
                pipeline.run_text_to_video(
                    prompt=prompt,
                    output_path=str(output_video),
                )
        finally:
            if prepared_is_temp and prepared_image_path:
                try:
                    os.unlink(prepared_image_path)
                except OSError:
                    pass
        gen_time = time.time() - t_start

        if is_rank0():
            meta = build_metadata(
                config=pipeline.config,
                prompt=prompt,
                source_caption=json_path.name,
                generation_time=gen_time,
                mode=args.mode,
                env_meta=env_meta,
                image_path=image_path,
            )
            with open(output_meta, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            log(f"  → {output_video} ({gen_time:.1f}s)")


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        _relaunch_via_torchrun()
    else:
        main()
