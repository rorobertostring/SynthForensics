#-*- coding: utf-8 -*-
"""
Preprocessing pipeline for SynthForensics → FakeSTormer format.

Extracts face crops (RetinaFace) from real and fake videos and saves them in the
structure expected by the FakeSTormer dataset loader:

    {output_root}/{sf_dataset}/{compression}/test/videos/{faketype}/{video_id}/*.png

Where {faketype} is:
    - "real"          → real videos (DFD/FF++)
    - {generator}     → fake videos per generator (cogvideox, magi-1, ...)

Modes:
    t2v (default): fake videos from {data_root}/{fake_videos_subdir}/{compression}/{generator}/
    i2v:           fake videos from {data_root}/{fake_videos_subdir}/{compression}/{generator}/
                   (different fake_videos_subdir, e.g. SynthForensics/I2V/videos)
                   Real is not re-extracted: a symlink to --real_preprocessed_root is created instead.

Idempotent:
    - Single video: skipped if the output folder already contains at least 1 PNG
    - Full subset: skipped if extraction_stats_{type}.json exists in the subset root

Usage (t2v):
    CUDA_VISIBLE_DEVICES=0 python preprocess_synthforensics.py \\
        --dataset SF-FF --compression crf23 \\
        --output_root preprocessed \\
        --data_root /path/to/Data

Usage (i2v):
    CUDA_VISIBLE_DEVICES=0 python preprocess_synthforensics.py \\
        --mode i2v \\
        --fake_videos_subdir SynthForensics/I2V/videos \\
        --real_preprocessed_root preprocessed \\
        --output_root preprocessed_i2v \\
        --data_root /path/to/Data

Usage (single combination):
    CUDA_VISIBLE_DEVICES=0 python preprocess_synthforensics.py \\
        --dataset SF-FF --compression crf23 --generator cogvideox \\
        --output_root preprocessed --data_root /path/to/Data
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())


# ─────────────────────────── constants ──────────────────────────────────────

GENERATORS = None  # auto-discovered from fake_base at runtime

# Maps sf_dataset → (real data dir relative to DATA_ROOT, split txt file)
DATASET_CFG = {
    "SF-FF": {
        "real_dir": "FF++/real_{compression}_test_set",
        "split_file": "SF-FF_fake_ids_140.txt",
    },
    "SF-DFD": {
        "real_dir": "DFD/real_{compression}_test_set",
        "split_file": "SF-DFD_fake_ids_363.txt",
    },
}

IMAGE_SIZE = 256
PADDING = 0.25


# ─────────────────────────── helpers ────────────────────────────────────────

def load_split_ids(split_file: str) -> set:
    """Load valid video IDs from a .txt split file, ignoring comment lines."""
    ids = set()
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def get_base_id(filename: str, valid_ids: set) -> str:
    """
    Extract base video ID by trying increasing underscore-prefixes of the stem
    until one matches valid_ids.

    Examples:
      '000_cogvideox.mp4'           → '000'
      '01__exit_phone_room_cogvideox.mp4' → '01__exit_phone_room'
      '000_ltx_2-3_t2v.mp4'        → '000'
    """
    name = os.path.splitext(filename)[0]  # strip .mp4
    parts = name.split('_')
    for i in range(1, len(parts)):
        candidate = '_'.join(parts[:i])
        if candidate in valid_ids:
            return candidate
    return name  # fallback: no match found


def extract_faces(video_path: str, out_dir: str, num_frames: int, retina_model) -> dict:
    """
    Extract face crops from a video and save as PNG.
    Returns stats dict with n_requested, n_extracted, n_no_face.
    Skips if out_dir already contains at least 1 PNG (idempotent).
    """
    existing = [f for f in os.listdir(out_dir) if f.endswith(".png")] if os.path.exists(out_dir) else []
    if existing:
        return {"skipped": True, "n_extracted": len(existing)}

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return {"error": f"Cannot read {video_path}"}

    frame_idxs = set(np.linspace(0, total - 1, num_frames, endpoint=True, dtype=np.int64))
    os.makedirs(out_dir, exist_ok=True)

    n_extracted = 0
    n_no_face = 0

    for cnt in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt not in frame_idxs:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        faces = retina_model.predict_jsons(frame_rgb)
        faces = [f for f in faces if len(f.get("bbox", [])) == 4 and f.get("score", -1) > 0]
        if not faces:
            n_no_face += 1
            continue

        best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        x0, y0, x1, y1 = best["bbox"]
        fw, fh = x1 - x0, y1 - y0
        x0c = max(0, int(x0 - fw * PADDING))
        x1c = min(w,  int(x1 + fw * PADDING))
        y0c = max(0, int(y0 - fh * PADDING))
        y1c = min(h,  int(y1 + fh * PADDING))

        crop = frame[y0c:y1c, x0c:x1c]
        crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, f"{n_extracted:03d}.png"), crop)
        n_extracted += 1

    cap.release()
    return {"n_requested": num_frames, "n_extracted": n_extracted, "n_no_face": n_no_face}


def process_subset(video_files: list, video_dir: str, out_videos_dir: str,
                   num_frames: int, retina_model, stats_key: str) -> dict:
    """
    Process a list of video files → extract faces → save stats JSON.
    Returns aggregated stats.
    """
    stats = {"total": len(video_files), "processed": 0, "skipped": 0, "errors": 0, "videos": {}}

    for vf in tqdm(video_files, desc=stats_key, leave=True):
        video_id = os.path.splitext(vf)[0]
        video_path = os.path.join(video_dir, vf)
        out_dir = os.path.join(out_videos_dir, video_id)

        result = extract_faces(video_path, out_dir, num_frames, retina_model)

        if result.get("skipped"):
            stats["skipped"] += 1
        elif "error" in result:
            stats["errors"] += 1
            print(f"  [error] {vf}: {result['error']}")
        else:
            stats["processed"] += 1
        stats["videos"][vf] = result

    return stats


def ensure_real_symlink(out_base: str, real_preprocessed_root: str,
                        sf_dataset: str, compression: str) -> bool:
    """
    In i2v mode: create a symlink {out_base}/real → {real_preprocessed_root}/{ds}/{comp}/test/videos/real
    so that eval can find real frames without re-extraction.
    Returns True if symlink was created or already exists, False if source is missing.
    """
    real_link = os.path.join(out_base, "real")
    if os.path.exists(real_link) or os.path.islink(real_link):
        return True

    real_src = os.path.abspath(os.path.join(
        real_preprocessed_root, sf_dataset, compression, "test", "videos", "real"
    ))
    if not os.path.exists(real_src):
        print(f"  [warn] real_preprocessed_root does not contain: {real_src}")
        print(f"         Run t2v preprocessing first, or omit --real_preprocessed_root to extract real.")
        return False

    os.makedirs(os.path.dirname(real_link), exist_ok=True)
    os.symlink(real_src, real_link)
    print(f"  [symlink] real → {real_src}")
    return True


# ─────────────────────────── main ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("SynthForensics → FakeSTormer preprocessing")

    # ── mode ─────────────────────────────────────────────────────────────────
    p.add_argument("--mode", default="t2v", choices=["t2v", "i2v"],
                   help="Generation mode: t2v (text-to-video, default) or i2v (image-to-video). "
                        "Controls which fake video source is used and whether real is re-extracted.")

    # ── SynthForensics mode ───────────────────────────────────────────────────
    p.add_argument("--dataset",      "-d", default="all",
                   choices=["SF-FF", "SF-DFD", "all"],
                   help="Which SF subset to process (default: all)")
    p.add_argument("--compression",  "-c", default="crf23",
                   choices=["crf23", "crf40", "raw", "canonical", "all"],
                   help="Compression level (default: crf23)")
    p.add_argument("--generator",    "-g", default=None,
                   help="Process only this generator (default: all, auto-discovered)")
    p.add_argument("--output_root",  "-o", default=None,
                   help="Root folder for extracted frames output")
    p.add_argument("--data_root",    default="/path/to/SynthForensics/Data",
                   help="Root of the SynthForensics Data directory")
    p.add_argument("--fake_videos_subdir", default="SynthForensics/T2V/videos",
                   help="Subdirectory under data_root containing fake videos per compression. "
                        "Default: SynthForensics/T2V/videos (t2v). "
                        "For i2v: SynthForensics/I2V/videos")
    p.add_argument("--real_preprocessed_root", default=None,
                   help="(i2v mode) Path to already-extracted t2v real frames. "
                        "A symlink {output_root}/../real → this path is created instead of re-extracting. "
                        "If omitted, real frames are extracted normally.")
    p.add_argument("--fake_only",    action="store_true",
                   help="Process only fake videos (skip real)")
    p.add_argument("--real_only",    action="store_true",
                   help="Process only real videos (skip fake)")

    # ── path_list mode (explicit video list) ───────────────────────────────────
    p.add_argument("--path_list_file", default=None,
                   help="JSON file with list of absolute video paths (path_list mode)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir for path_list mode: frames saved to "
                        "{output_dir}/{video_id}/*.png")

    # ── common ───────────────────────────────────────────────────────────────
    p.add_argument("--num_frames",   default=32, type=int,
                   help="Frames to extract per video (default: 32)")
    p.add_argument("--gpu",          default=0, type=int,
                   help="GPU index for RetinaFace (default: 0)")
    return p.parse_args()


def video_id_from_path(video_path: str, common_prefix: str) -> str:
    rel = os.path.relpath(video_path, common_prefix)
    parts = rel.replace("\\", "/").split("/")

    if len(parts) == 1 or ".." in parts:
        full_parts = video_path.replace("\\", "/").split("/")
        fallback = full_parts[-3:]
        fallback[-1] = os.path.splitext(fallback[-1])[0]
        return "_".join(fallback)

    stem = os.path.splitext(parts[-1])[0]
    return "_".join(parts[:-1] + [stem])


def process_path_list(path_list_file: str, output_dir: str,
                      num_frames: int, retina_model) -> None:
    sentinel = os.path.join(output_dir, "extraction_stats_fake.json")
    if os.path.exists(sentinel):
        print(f"[skip] sentinel exists: {sentinel}")
        return

    with open(path_list_file) as f:
        video_paths = json.load(f)

    if not video_paths:
        print("[warn] path_list_file is empty")
        return

    n_raw = len(video_paths)
    video_paths = list(dict.fromkeys(video_paths))
    n_dupes = n_raw - len(video_paths)
    if n_dupes:
        print(f"  [warn] removed {n_dupes} duplicate paths from JSON")

    common_prefix = os.path.commonpath(video_paths)
    print(f"  Common prefix : {common_prefix}")
    print(f"  Unique videos : {len(video_paths)}  (raw: {n_raw})")

    os.makedirs(output_dir, exist_ok=True)

    stats = {"total": len(video_paths), "processed": 0, "skipped": 0,
             "errors": 0, "no_face": 0, "videos": {}}

    for video_path in tqdm(video_paths, desc="path_list", unit="video",
                           leave=True, dynamic_ncols=True):
        vid_id = video_id_from_path(video_path, common_prefix)
        out_dir = os.path.join(output_dir, vid_id)

        if not os.path.exists(video_path):
            tqdm.write(f"  [error] not found: {video_path}")
            stats["errors"] += 1
            stats["videos"][vid_id] = {"error": "file not found"}
            continue

        result = extract_faces(video_path, out_dir, num_frames, retina_model)

        if result.get("skipped"):
            stats["skipped"] += 1
        elif "error" in result:
            stats["errors"] += 1
            tqdm.write(f"  [error] {vid_id}: {result['error']}")
        else:
            stats["processed"] += 1
            if result.get("n_no_face", 0) == result.get("n_requested", 1):
                stats["no_face"] += 1
        stats["videos"][vid_id] = result

    with open(sentinel, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\npath_list done: {stats['processed']} processed, "
          f"{stats['skipped']} skipped, {stats['errors']} errors, "
          f"{stats['no_face']} all-no-face")
    print(f"Sentinel → {sentinel}")


def main():
    args = parse_args()

    # ── path_list mode ───────────────────────────────────────────────────────
    if args.path_list_file is not None:
        if args.output_dir is None:
            raise ValueError("--output_dir is required when --path_list_file is set")

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        try:
            from retinaface.pre_trained_models import get_model as get_retina
        except ImportError:
            raise ImportError("Install retinaface-pytorch:  pip install retinaface-pytorch")

        print("Loading RetinaFace model...")
        retina = get_retina("resnet50_2020-07-20", max_size=2048, device=device)
        retina.eval()

        process_path_list(args.path_list_file, args.output_dir, args.num_frames, retina)
        return

    # ── SynthForensics mode ───────────────────────────────────────────────────
    if args.output_root is None:
        raise ValueError("--output_root is required in SynthForensics mode")

    datasets     = list(DATASET_CFG.keys()) if args.dataset == "all" else [args.dataset]
    compressions = ["crf23", "crf40", "raw", "canonical"] if args.compression == "all" else [args.compression]

    # In i2v mode real is shared → skip unless explicitly requested
    skip_real = (args.mode == "i2v" and args.real_preprocessed_root is not None)
    process_real = not args.fake_only and not skip_real
    process_fake = not args.real_only

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  mode: {args.mode}")

    # Load RetinaFace only if we actually need it
    retina = None
    if process_real or process_fake:
        try:
            from retinaface.pre_trained_models import get_model as get_retina
        except ImportError:
            raise ImportError("Install retinaface-pytorch:  pip install retinaface-pytorch")
        print("Loading RetinaFace model...")
        retina = get_retina("resnet50_2020-07-20", max_size=2048, device=device)
        retina.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\nWork plan: {datasets} x {compressions}")
    print(f"  mode={args.mode}  fake_videos_subdir={args.fake_videos_subdir}")
    print(f"  generator filter={args.generator}  fake_only={args.fake_only}  real_only={args.real_only}")
    print(f"  num_frames={args.num_frames}  output={args.output_root}\n")

    for sf_dataset in datasets:
        cfg = DATASET_CFG[sf_dataset]
        split_file = os.path.join(script_dir, cfg["split_file"])
        valid_ids = load_split_ids(split_file)
        print(f"\n{'='*60}")
        print(f"Dataset: {sf_dataset}  |  {len(valid_ids)} valid IDs in split")

        for compression in compressions:
            print(f"\n  Compression: {compression}")

            out_base = os.path.join(args.output_root, sf_dataset, compression, "test", "videos")

            # ── REAL ────────────────────────────────────────────────────────
            if skip_real:
                # i2v mode with real_preprocessed_root: create symlink instead of extracting
                ensure_real_symlink(out_base, args.real_preprocessed_root, sf_dataset, compression)
            elif process_real:
                real_dir = os.path.join(
                    args.data_root,
                    cfg["real_dir"].format(compression=compression)
                )
                if not os.path.exists(real_dir):
                    print(f"  [skip] Real dir not found: {real_dir}")
                else:
                    stats_path = os.path.join(out_base, "real", "extraction_stats_real.json")
                    if os.path.exists(stats_path):
                        print(f"  [skip] Real already processed ({stats_path})")
                    else:
                        real_videos = sorted([f for f in os.listdir(real_dir) if f.endswith(".mp4")])
                        print(f"  Real videos: {len(real_videos)}")
                        out_real = os.path.join(out_base, "real")
                        os.makedirs(out_real, exist_ok=True)
                        stats = process_subset(
                            real_videos, real_dir, out_real, args.num_frames, retina,
                            f"{sf_dataset}/{compression}/real ({len(real_videos)} videos)"
                        )
                        with open(stats_path, "w") as f:
                            json.dump(stats, f, indent=2)
                        print(f"  Real done: {stats['processed']} processed, "
                              f"{stats['skipped']} skipped, {stats['errors']} errors")

            # ── FAKE (per generator) ─────────────────────────────────────────
            if not process_fake:
                continue

            fake_base = os.path.join(args.data_root, args.fake_videos_subdir, compression)
            if not os.path.exists(fake_base):
                print(f"  [skip] Fake base not found: {fake_base}")
                continue

            # Auto-discover generators, then apply optional --generator filter
            all_generators = sorted([
                d for d in os.listdir(fake_base)
                if os.path.isdir(os.path.join(fake_base, d))
            ])
            generators = [args.generator] if args.generator else all_generators
            # Validate requested generator exists
            generators = [g for g in generators if os.path.isdir(os.path.join(fake_base, g))]
            if args.generator and not generators:
                print(f"  [warn] Generator '{args.generator}' not found in {fake_base}")
                continue
            print(f"  Generators: {generators}  (all available: {all_generators})")

            for generator in generators:
                gen_dir = os.path.join(fake_base, generator)

                stats_path = os.path.join(out_base, generator, "extraction_stats_fake.json")
                if os.path.exists(stats_path):
                    print(f"  [skip] {generator} already processed")
                    continue

                all_videos = sorted([f for f in os.listdir(gen_dir) if f.endswith(".mp4")])
                filtered = [
                    vf for vf in all_videos
                    if get_base_id(vf, valid_ids) in valid_ids
                ]
                print(f"  {generator}: {len(filtered)}/{len(all_videos)} videos match split")

                out_gen = os.path.join(out_base, generator)
                os.makedirs(out_gen, exist_ok=True)

                gen_stats = {"total": len(filtered), "processed": 0, "skipped": 0, "errors": 0, "videos": {}}
                for vf in tqdm(filtered, desc=f"{sf_dataset}/{compression}/{generator}",
                               unit="video", leave=True, dynamic_ncols=True):
                    base_id = get_base_id(vf, valid_ids)
                    video_path = os.path.join(gen_dir, vf)
                    out_dir = os.path.join(out_gen, base_id)
                    result = extract_faces(video_path, out_dir, args.num_frames, retina)
                    if result.get("skipped"):
                        gen_stats["skipped"] += 1
                    elif "error" in result:
                        gen_stats["errors"] += 1
                        tqdm.write(f"  [error] {vf}: {result['error']}")
                    else:
                        gen_stats["processed"] += 1
                    gen_stats["videos"][vf] = result

                with open(stats_path, "w") as f:
                    json.dump(gen_stats, f, indent=2)
                print(f"  {generator} done: {gen_stats['processed']} processed, "
                      f"{gen_stats['skipped']} skipped, {gen_stats['errors']} errors")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
