# -*- coding: utf-8 -*-
"""
Extract uniformly sampled frames from videos, detect and crop faces with
RetinaFace, save crops as PNG, and report face detection stats.

Modes (--mode):
  both       [default] — process real and fake, saves to output_dir/real/ and output_dir/fake/
  real                 — process only real videos, saves to output_dir/<video_id>/
  fake                 — process only fake videos, saves to output_dir/<video_id>/
  path_list            — process a JSON list of absolute video paths, saves to output_dir/<video_id>/

Usage examples:
  python scripts/extract_frames_custom.py \\
      --mode real \\
      --real_dir path/to/real_videos \\
      --output_dir path/to/output

  python scripts/extract_frames_custom.py \\
      --mode fake \\
      --fake_dir path/to/fake_videos \\
      --split_file path/to/split.txt \\
      --output_dir path/to/output

  python scripts/extract_frames_custom.py \\
      --mode path_list \\
      --path_list_file path/to/video_list.json \\
      --output_dir path/to/output
"""

import os
import sys
import json
import argparse
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp')
IMAGE_H, IMAGE_W = 256, 256


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_video_paths(folder):
    """Return sorted list of all video files directly inside folder."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths += glob(os.path.join(folder, f'*{ext}'))
        paths += glob(os.path.join(folder, f'*{ext.upper()}'))
    return sorted(set(paths))


def load_split_ids(split_file):
    """
    Load a list of video IDs from a JSON array or a plain/commented TXT file.
    TXT lines starting with '#' or empty lines are ignored.
    """
    with open(split_file) as f:
        content = f.read().strip()
    if content.startswith('[') or content.startswith('{'):
        return json.loads(content)
    # Plain text / commented text format
    ids = []
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            ids.append(line)
    return ids


def filter_by_split(video_paths, split_ids):
    """
    Keep only videos whose ID is in split_ids.

    Handles any generator suffix (including multi-underscore ones like '_ltx_2-3_t2v')
    by trying increasingly long underscore-joined prefixes until one matches split_ids.

    Examples:
      '006_cogvideox.mp4'           → tries '006' → match
      '01__exit_phone_room_cog.mp4' → tries '01', '01_', '01__exit', ... → match
      '000_ltx_2-3_t2v.mp4'        → tries '000' → match
    """
    split_set = set(split_ids)
    filtered = []
    for vp in video_paths:
        stem = os.path.splitext(os.path.basename(vp))[0]
        parts = stem.split('_')
        for i in range(1, len(parts) + 1):
            candidate = '_'.join(parts[:i])
            if candidate in split_set:
                filtered.append(vp)
                break
    return filtered


def extract_uniform_frames(video_path, num_frames=32):
    """Extract num_frames uniformly spaced frames. Returns list of (frame_idx, bgr_array)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, num_frames, endpoint=True, dtype=np.int64)
    frames = {}
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames[int(idx)] = frame
    cap.release()
    return [(idx, frames[idx]) for idx in sorted(frames.keys())]


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def detect_and_crop_face(model, frame_bgr, padding=0.25):
    """Detect largest high-confidence face; return 256×256 BGR crop or None."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    faces = model.predict_jsons(frame_rgb)
    if not faces:
        return None

    best, best_area = None, -1
    for face in faces:
        bbox = face.get('bbox', [])
        if len(bbox) < 4:
            continue
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        if area > best_area and face.get('score', 0) > 0:
            best, best_area = face, area

    if best is None:
        return None

    x0, y0, x1, y1 = best['bbox']
    fw, fh = x1 - x0, y1 - y0
    cx0 = max(0, int(x0 - fw * padding))
    cy0 = max(0, int(y0 - fh * padding))
    cx1 = min(w, int(x1 + fw * padding))
    cy1 = min(h, int(y1 + fh * padding))
    crop = frame_bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_videos(model, video_paths, label_str, save_root, num_frames, padding,
                   vid_names=None):
    """
    Extract faces from all videos, save PNGs directly under save_root/<video_id>/.
    Returns list of per-video stat dicts.

    vid_names: optional dict {video_path: unique_id} used when filenames alone are
               not unique across the list (e.g. multiple generators all have 'a.mp4').
               When None the stem of the filename is used (original behaviour).
    """
    stats = []

    for vpath in tqdm(video_paths, desc=label_str, unit="video"):
        if vid_names is not None:
            vid_name = vid_names[vpath]
        else:
            vid_name = os.path.splitext(os.path.basename(vpath))[0]
        vid_save_dir = os.path.join(save_root, vid_name)

        # Resume: skip if already processed
        existing = glob(os.path.join(vid_save_dir, '*.png'))
        if existing:
            stats.append({
                "video": vpath, "label": label_str,
                "extracted": len(existing), "no_face_frames": 0,
                "no_face_at_all": False, "skipped": True,
            })
            continue

        os.makedirs(vid_save_dir, exist_ok=True)
        raw_frames = extract_uniform_frames(vpath, num_frames)

        if not raw_frames:
            tqdm.write(f"[WARN] Cannot read video: {vpath}")
            stats.append({
                "video": vpath, "label": label_str,
                "extracted": 0, "no_face_frames": num_frames,
                "no_face_at_all": True, "skipped": False,
            })
            continue

        no_face_count = 0
        saved_count = 0

        for frame_idx, frame_bgr in raw_frames:
            crop = detect_and_crop_face(model, frame_bgr, padding)
            if crop is None:
                no_face_count += 1
                continue
            cv2.imwrite(os.path.join(vid_save_dir, f"{str(frame_idx).zfill(6)}.png"), crop)
            saved_count += 1

        no_face_at_all = (saved_count == 0)
        stats.append({
            "video": vpath, "label": label_str,
            "extracted": saved_count, "no_face_frames": no_face_count,
            "no_face_at_all": no_face_at_all, "skipped": False,
        })

        if no_face_at_all:
            tqdm.write(f"[WARN] No face in any frame: {vid_name}")
        elif no_face_count > 0:
            tqdm.write(f"[INFO] {no_face_count}/{len(raw_frames)} frames missing: {vid_name}")

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

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SynthForensics videos with LAA-Net face crop pipeline"
    )
    parser.add_argument("--mode", choices=["both", "real", "fake", "path_list"], default="both",
                        help="'real': only real videos → output_dir/<vid>/*.png | "
                             "'fake': only fake videos → output_dir/<vid>/*.png | "
                             "'both': both → output_dir/real/<vid>/ and output_dir/fake/<vid>/ | "
                             "'path_list': JSON list of absolute paths → output_dir/<vid>/*.png")
    parser.add_argument("--real_dir", default=None,
                        help="Flat folder of real .mp4 (required for mode=real|both)")
    parser.add_argument("--fake_dir", default=None,
                        help="Flat folder of fake .mp4 (required for mode=fake|both)")
    parser.add_argument("--output_dir", required=True,
                        help="Output root directory for PNG crops")
    parser.add_argument("--split_file", default=None,
                        help="File with video IDs to keep (JSON array or plain TXT), for mode=fake|both")
    parser.add_argument("--path_list_file", default=None,
                        help="JSON file containing a list of absolute video paths (required for mode=path_list)")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--padding", type=float, default=0.25,
                        help="Face bbox padding ratio (default 0.25)")
    args = parser.parse_args()

    # Validate args
    if args.mode in ("real", "both") and not args.real_dir:
        parser.error("--real_dir is required for mode=real|both")
    if args.mode in ("fake", "both") and not args.fake_dir:
        parser.error("--fake_dir is required for mode=fake|both")
    if args.mode == "path_list" and not args.path_list_file:
        parser.error("--path_list_file is required for mode=path_list")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading RetinaFace model...")
    from retinaface.pre_trained_models import get_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    retina = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    retina.eval()

    all_stats = []

    # --- Real ---
    if args.mode in ("real", "both"):
        real_videos = get_video_paths(args.real_dir)
        print(f"Real videos: {len(real_videos)}")
        save_root = os.path.join(args.output_dir, "real") if args.mode == "both" else args.output_dir
        real_stats = process_videos(retina, real_videos, "real", save_root, args.num_frames, args.padding)
        all_stats += real_stats
        stats_path = os.path.join(args.output_dir if args.mode == "both" else args.output_dir,
                                  "extraction_stats_real.json")
        with open(stats_path, "w") as f:
            json.dump(real_stats, f, indent=2)

    # --- Fake ---
    if args.mode in ("fake", "both"):
        fake_videos = get_video_paths(args.fake_dir)
        if args.split_file:
            split_ids = load_split_ids(args.split_file)
            fake_videos_filtered = filter_by_split(fake_videos, split_ids)
            print(f"Fake videos: {len(fake_videos_filtered)} / {len(fake_videos)} (after split filter)")
            fake_videos = fake_videos_filtered
        else:
            print(f"Fake videos: {len(fake_videos)} (no split filter)")
        save_root = os.path.join(args.output_dir, "fake") if args.mode == "both" else args.output_dir
        fake_stats = process_videos(retina, fake_videos, "fake", save_root, args.num_frames, args.padding)
        all_stats += fake_stats
        stats_path = os.path.join(args.output_dir, "extraction_stats_fake.json")
        with open(stats_path, "w") as f:
            json.dump(fake_stats, f, indent=2)

    # --- Path List ---
    if args.mode == "path_list":
        with open(args.path_list_file) as f:
            video_paths = json.load(f)
        print(f"Videos from path list: {len(video_paths)}")

        # Build unique IDs from relative paths to avoid collisions when multiple
        # subdirectories contain files with the same name.
        common = os.path.commonpath(video_paths)
        vid_names = {
            vp: os.path.relpath(os.path.splitext(vp)[0], common).replace(os.sep, "_")
            for vp in video_paths
        }

        fake_stats = process_videos(retina, video_paths, "fake", args.output_dir,
                                    args.num_frames, args.padding, vid_names=vid_names)
        all_stats += fake_stats
        stats_path = os.path.join(args.output_dir, "extraction_stats_fake.json")
        with open(stats_path, "w") as f:
            json.dump(fake_stats, f, indent=2)

    print_report(all_stats, args.num_frames)
    n_real = sum(s["extracted"] for s in all_stats if s["label"] == "real")
    n_fake = sum(s["extracted"] for s in all_stats if s["label"] == "fake")
    print(f"Total frames saved: {n_real + n_fake}  ({n_real} real, {n_fake} fake)")


if __name__ == "__main__":
    main()
