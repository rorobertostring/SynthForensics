# -*- coding: utf-8 -*-
"""
Extract frames for GenD training/fine-tuning with optional multi-worker GPU parallelism.

Each worker process holds its own RetinaFace ONNX session (CUDAExecutionProvider when
available), so N workers keep the GPU busy while other workers decode/save concurrently.

Output structure:
  <output_dir>/<video_id>/frame_XXXXXX.png

Modes:
  real  — video_id = file stem (no suffix stripping)
  fake  — video_id extracted from stem by stripping trailing generator suffix;
          optionally filtered by --split_file

Usage:
  # Sequential (1 worker, default)
  python SF_scripts/extract_frames_training.py \\
      --mode real --real_dir /path/to/FF++_Full --output_dir preprocessed_gend/training/real_ffpp/real

  # 4 parallel workers
  python SF_scripts/extract_frames_training.py \\
      --mode fake --workers 4 \\
      --fake_dir /path/to/SynthForensics/videos/raw/cogvideox \\
      --split_file combined_split.json \\
      --output_dir preprocessed_gend/training/t2v/cogvideox
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from glob import glob

import av
import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detector import align_face
from src.retinaface import prepare_model

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp")
DEFAULT_NUM_FRAMES = 32
DEFAULT_TARGET_SIZE = (256, 256)
DEFAULT_SCALE = 1.3
DEFAULT_DET_THRES = 0.1


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def get_video_paths(folder):
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths += glob(os.path.join(folder, f"*{ext}"))
        paths += glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(paths))


def load_split_ids(split_file):
    with open(split_file) as f:
        content = f.read().strip()
    if content.startswith("[") or content.startswith("{"):
        return json.loads(content)
    return [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]


def _extract_vid_id(stem, video_suffix=None, split_set=None):
    if video_suffix and stem.endswith("_" + video_suffix):
        return stem[: -(len(video_suffix) + 1)]
    if split_set is not None:
        parts = stem.split("_")
        for i in range(1, len(parts)):
            candidate = "_".join(parts[:i])
            if candidate in split_set:
                return candidate
    return stem.rsplit("_", 1)[0]


def filter_by_split(video_paths, split_ids, video_suffix=None):
    split_set = set(split_ids)
    filtered = []
    for vp in video_paths:
        stem = os.path.splitext(os.path.basename(vp))[0]
        vid_id = _extract_vid_id(stem, video_suffix, split_set)
        if vid_id in split_set:
            filtered.append(vp)
    return filtered


def strip_generator_suffix(stem, video_suffix=None, split_set=None):
    return _extract_vid_id(stem, video_suffix, split_set)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_uniform_frames(video_path, num_frames):
    """
    Decode the video in a single sequential pass with pyav (no seeking).
    Collects num_frames uniformly spaced frames. Much faster than per-frame seek.
    """
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"[WARN] Cannot open video {video_path}: {e}", flush=True)
        return []

    stream = container.streams.video[0]
    total = stream.frames
    if total == 0:
        # some containers don't report frame count; fall back to decoding all
        try:
            total = sum(1 for _ in container.decode(video=0))
            container.seek(0)
        except Exception:
            container.close()
            return []

    if total <= 0:
        container.close()
        return []

    target_idxs = set(np.linspace(0, total - 1, num_frames, endpoint=True, dtype=int).tolist())
    frames = []
    try:
        for fi, frame in enumerate(container.decode(video=0)):
            if fi in target_idxs:
                img = frame.to_ndarray(format="bgr24")
                frames.append((fi, img))
            if fi >= max(target_idxs):
                break
    except Exception as e:
        print(f"[WARN] Decode error {video_path}: {e}", flush=True)
    finally:
        container.close()

    return frames


def detect_and_align(model, frame_bgr, target_size, scale):
    try:
        xyxy, landmarks = model.detect(frame_bgr)
    except Exception:
        return None
    if xyxy is None or len(xyxy) == 0:
        return None
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))
    try:
        aligned, _ = align_face(frame_bgr, landmarks[idx], target_size=target_size, scale=scale)
    except Exception:
        return None
    return aligned


# ---------------------------------------------------------------------------
# Multi-worker support
# ---------------------------------------------------------------------------

_g_model = None  # per-worker model, initialized by _worker_init


def _worker_init(det_thres):
    global _g_model
    _g_model = prepare_model(det_thres)


def _process_video_task(task):
    """Process one video in a worker process."""
    global _g_model
    vpath, vid_id, output_dir, num_frames, target_size, scale, label_str = task

    vid_save_dir = os.path.join(output_dir, vid_id)
    existing = glob(os.path.join(vid_save_dir, "*.png"))
    if existing:
        return {
            "video": vpath, "label": label_str,
            "extracted": len(existing), "no_face_frames": 0,
            "no_face_at_all": False, "skipped": True,
        }

    os.makedirs(vid_save_dir, exist_ok=True)
    raw_frames = extract_uniform_frames(vpath, num_frames)

    if not raw_frames:
        return {
            "video": vpath, "label": label_str,
            "extracted": 0, "no_face_frames": num_frames,
            "no_face_at_all": True, "skipped": False,
            "_warn": f"[WARN] Cannot read video: {vpath}",
        }

    no_face_count = 0
    saved_count = 0
    for frame_idx, frame_bgr in raw_frames:
        crop = detect_and_align(_g_model, frame_bgr, target_size, scale)
        if crop is None:
            no_face_count += 1
            continue
        cv2.imwrite(os.path.join(vid_save_dir, f"frame_{frame_idx:06d}.png"), crop)
        saved_count += 1

    no_face_at_all = (saved_count == 0)
    result = {
        "video": vpath, "label": label_str,
        "extracted": saved_count, "no_face_frames": no_face_count,
        "no_face_at_all": no_face_at_all, "skipped": False,
    }
    if no_face_at_all:
        result["_warn"] = f"[WARN] No face in any frame: {vid_id}"
    elif no_face_count > 0:
        result["_info"] = f"[INFO] {no_face_count}/{len(raw_frames)} frames missing: {vid_id}"
    return result


# ---------------------------------------------------------------------------
# Core processing dispatcher
# ---------------------------------------------------------------------------

def _build_tasks(video_paths, label_str, output_dir, num_frames, target_size, scale,
                 strip_suffix, video_suffix, split_set):
    tasks = []
    for vpath in video_paths:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        if strip_suffix:
            vid_id = strip_generator_suffix(stem, video_suffix, split_set)
        else:
            vid_id = stem
        tasks.append((vpath, vid_id, output_dir, num_frames, target_size, scale, label_str))
    return tasks


def process_videos(model, video_paths, label_str, output_dir, num_frames, target_size, scale,
                   strip_suffix=False, video_suffix=None, split_set=None,
                   num_workers=1, det_thres=DEFAULT_DET_THRES):

    tasks = _build_tasks(video_paths, label_str, output_dir, num_frames,
                         target_size, scale, strip_suffix, video_suffix, split_set)

    if num_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers, initializer=_worker_init, initargs=(det_thres,)) as pool:
            stats = []
            for result in tqdm(
                pool.imap_unordered(_process_video_task, tasks),
                total=len(tasks), desc=label_str, unit="video",
            ):
                warn = result.pop("_warn", None)
                info = result.pop("_info", None)
                if warn:
                    tqdm.write(warn)
                if info:
                    tqdm.write(info)
                stats.append(result)
        return stats

    # Sequential
    stats = []
    for task in tqdm(tasks, desc=label_str, unit="video"):
        vpath, vid_id, output_dir_, num_frames_, target_size_, scale_, lbl = task
        vid_save_dir = os.path.join(output_dir_, vid_id)

        existing = glob(os.path.join(vid_save_dir, "*.png"))
        if existing:
            stats.append({
                "video": vpath, "label": lbl,
                "extracted": len(existing), "no_face_frames": 0,
                "no_face_at_all": False, "skipped": True,
            })
            continue

        os.makedirs(vid_save_dir, exist_ok=True)
        raw_frames = extract_uniform_frames(vpath, num_frames_)

        if not raw_frames:
            stats.append({
                "video": vpath, "label": lbl,
                "extracted": 0, "no_face_frames": num_frames_,
                "no_face_at_all": True, "skipped": False,
            })
            tqdm.write(f"[WARN] Cannot read video: {vpath}")
            continue

        no_face_count = 0
        saved_count = 0
        for frame_idx, frame_bgr in raw_frames:
            crop = detect_and_align(model, frame_bgr, target_size_, scale_)
            if crop is None:
                no_face_count += 1
                continue
            cv2.imwrite(os.path.join(vid_save_dir, f"frame_{frame_idx:06d}.png"), crop)
            saved_count += 1

        no_face_at_all = (saved_count == 0)
        stats.append({
            "video": vpath, "label": lbl,
            "extracted": saved_count, "no_face_frames": no_face_count,
            "no_face_at_all": no_face_at_all, "skipped": False,
        })
        if no_face_at_all:
            tqdm.write(f"[WARN] No face in any frame: {vid_id}")
        elif no_face_count > 0:
            tqdm.write(f"[INFO] {no_face_count}/{len(raw_frames)} frames missing: {vid_id}")

    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(stats, num_frames):
    no_face = [s for s in stats if s["no_face_at_all"]]
    partial = [s for s in stats if not s["no_face_at_all"] and s["no_face_frames"] > 0 and not s.get("skipped")]
    skipped = [s for s in stats if s.get("skipped")]

    print("\n" + "=" * 60)
    print("FACE EXTRACTION REPORT")
    print("=" * 60)
    print(f"Total    : {len(stats)}")
    print(f"Skipped  : {len(skipped)}  (already processed)")
    print(f"No face  : {len(no_face)}")
    print(f"Partial  : {len(partial)}")

    if no_face:
        print("\n--- No face detected ---")
        for s in no_face:
            print(f"  [{s['label']}] {os.path.basename(s['video'])}")

    if partial:
        print("\n--- Partial detection ---")
        for s in partial:
            pct = s["no_face_frames"] / num_frames * 100
            print(f"  [{s['label']}] {os.path.basename(s['video'])}: "
                  f"{s['no_face_frames']}/{num_frames} missing ({pct:.1f}%)")

    for lbl in ("real", "fake"):
        group = [s for s in stats if s["label"] == lbl and not s.get("skipped")]
        if not group:
            continue
        avg_saved = np.mean([s["extracted"] for s in group])
        avg_missing = np.mean([s["no_face_frames"] for s in group])
        print(f"\n{lbl.upper()}: avg saved={avg_saved:.1f}/{num_frames}, "
              f"avg missing={avg_missing:.1f}/{num_frames}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_target_size(s):
    try:
        w, h = map(int, s.split(","))
        return (w, h)
    except ValueError:
        raise argparse.ArgumentTypeError("Use 'W,H' format, e.g. '256,256'")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SynthForensics videos for GenD training (multi-worker capable)"
    )
    parser.add_argument("--mode", choices=["real", "fake"], required=True)
    parser.add_argument("--real_dir", default=None)
    parser.add_argument("--fake_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_file", default=None,
                        help="JSON with video IDs to keep (for --mode fake)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--target_size", type=parse_target_size, default="256,256")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--det_thres", type=float, default=DEFAULT_DET_THRES)
    parser.add_argument("--txt_out", default=None,
                        help="Output .txt listing saved PNG paths (default: <output_dir>.txt)")
    parser.add_argument("--video_suffix", default=None,
                        help="Explicit generator suffix to strip from filenames")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (each with its own ONNX session). "
                             "Default: 1. Recommended: 2-4 for GPU preprocessing.")
    args = parser.parse_args()

    if args.mode == "real" and not args.real_dir:
        parser.error("--real_dir required for --mode real")
    if args.mode == "fake" and not args.fake_dir:
        parser.error("--fake_dir required for --mode fake")

    os.makedirs(args.output_dir, exist_ok=True)
    num_workers = max(1, args.workers)

    if num_workers == 1:
        print("Loading RetinaFace ONNX model...")
        model = prepare_model(args.det_thres)
    else:
        print(f"Multi-worker mode: {num_workers} workers (each loads its own ONNX session).")
        model = None

    print(f"Target size: {args.target_size}, scale: {args.scale}, det_thres: {args.det_thres}")

    if args.mode == "real":
        videos = get_video_paths(args.real_dir)
        print(f"Real videos found: {len(videos)}")
        stats = process_videos(
            model, videos, "real", args.output_dir,
            args.num_frames, args.target_size, args.scale,
            strip_suffix=False,
            num_workers=num_workers, det_thres=args.det_thres,
        )
        stats_path = os.path.join(args.output_dir, "extraction_stats_real.json")

    else:  # fake
        videos = get_video_paths(args.fake_dir)
        split_ids = None
        split_set = None
        if args.split_file:
            split_ids = load_split_ids(args.split_file)
            split_set = set(split_ids)
            videos = filter_by_split(videos, split_ids, args.video_suffix)
            print(f"Fake videos: {len(videos)} (after split filter)")
        else:
            print(f"Fake videos: {len(videos)} (no split filter)")
        if args.video_suffix:
            print(f"Video suffix: '{args.video_suffix}'")
        stats = process_videos(
            model, videos, "fake", args.output_dir,
            args.num_frames, args.target_size, args.scale,
            strip_suffix=True, video_suffix=args.video_suffix, split_set=split_set,
            num_workers=num_workers, det_thres=args.det_thres,
        )
        stats_path = os.path.join(args.output_dir, "extraction_stats_fake.json")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print_report(stats, args.num_frames)
    print(f"Total frames saved: {sum(s['extracted'] for s in stats)}")

    txt_out = args.txt_out or (args.output_dir.rstrip("/") + ".txt")
    os.makedirs(os.path.dirname(os.path.abspath(txt_out)), exist_ok=True)
    png_paths = sorted(glob(os.path.join(args.output_dir, "**", "*.png"), recursive=True))
    with open(txt_out, "w") as f:
        f.write("\n".join(png_paths) + "\n")
    print(f"Written {len(png_paths)} paths → {txt_out}")


if __name__ == "__main__":
    main()
