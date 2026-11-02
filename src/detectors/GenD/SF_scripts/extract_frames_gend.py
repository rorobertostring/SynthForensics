# -*- coding: utf-8 -*-
"""
Extract frames from SynthForensics real/fake videos using GenD's face pipeline:
  - RetinaFace ONNX (same model used by detector.py)
  - 5-point landmark alignment via detector.align_face

Output structure (4-level, compatible with DeepfakeDataset loader):
  <output_dir>/<video_id>/frame_XXXXXX.png

Where output_dir should already include the source-level folder, e.g.:
  preprocessed_gend/real_ffpp_canonical/real/      ← run with --mode real
  preprocessed_gend/fake_SF-FF_canonical/cogvideox/  ← run with --mode fake

Modes:
  real  — process real videos; video_id = stem of each file
  fake  — process fake videos; video_id = stem with trailing _{generator} stripped
          optionally filtered by --split_file

Usage:
  # Real videos
  python SF_scripts/extract_frames_gend.py \\
      --mode real \\
      --real_dir /path/to/FF++/real_canonical_test_set \\
      --output_dir preprocessed_gend/real_ffpp_canonical/real

  # Fake videos
  python SF_scripts/extract_frames_gend.py \\
      --mode fake \\
      --fake_dir /path/to/SynthForensics/videos/canonical/cogvideox \\
      --split_file /path/to/SynthForensics/splits/test.json \\
      --output_dir preprocessed_gend/fake_SF-FF_canonical/cogvideox
"""

import argparse
import json
import os
import sys
from glob import glob

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path
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
    ids = []
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def _extract_vid_id(stem, video_suffix=None, split_set=None):
    """
    Strip the generator suffix from a video stem to get the video ID.
    - Explicit suffix : '006_ltx_2-3_t2v'       → '006'  (video_suffix='ltx_2-3_t2v')
    - Prefix matching : '000_ltx_2-3_t2v'        → tries '000' → match in split_set → '000'
                        '02__exit_phone_room_cogvideox' → tries growing prefixes until match
    - Fallback        : rsplit('_', 1)[0] when no split_set provided
    """
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
    """
    Keep videos whose ID (stem without trailing generator suffix) is in split_ids.
    e.g.  '006_cogvideox.mp4'                 → ID = '006'
          '02__exit_phone_room_cogvideox.mp4'  → ID = '02__exit_phone_room'
          '006_ltx_2-3_t2v.mp4'               → ID = '006'
    Uses prefix matching against split_ids, no need for explicit video_suffix.
    """
    split_set = set(split_ids)
    filtered = []
    for vp in video_paths:
        stem = os.path.splitext(os.path.basename(vp))[0]
        vid_id = _extract_vid_id(stem, video_suffix, split_set)
        if vid_id in split_set:
            filtered.append(vp)
    return filtered


def strip_generator_suffix(stem, video_suffix=None):
    """'006_cogvideox' → '006';  '02__exit_phone_room_cogvideox' → '02__exit_phone_room'"""
    return _extract_vid_id(stem, video_suffix)


# ---------------------------------------------------------------------------
# Frame extraction (using imageio + pyav, same as detector.py)
# ---------------------------------------------------------------------------

def extract_uniform_frames(video_path, num_frames):
    try:
        props = iio.improps(video_path, plugin="pyav")
        total = props.shape[0]
    except Exception as e:
        print(f"[WARN] Cannot read video {video_path}: {e}")
        return []

    if total <= 0:
        return []

    idxs = np.linspace(0, total - 1, num_frames, endpoint=True, dtype=int)
    frames = []
    for idx in idxs:
        try:
            frame_rgb = iio.imread(video_path, index=int(idx), plugin="pyav")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames.append((int(idx), frame_bgr))
        except Exception:
            continue
    return frames


# ---------------------------------------------------------------------------
# Face detection + alignment (GenD pipeline)
# ---------------------------------------------------------------------------

def detect_and_align(model, frame_bgr, target_size, scale):
    """
    Detect largest face, align with 5-point landmarks.
    Returns aligned BGR crop or None.
    """
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
# Core processing
# ---------------------------------------------------------------------------

def process_videos(model, video_paths, label_str, output_dir, num_frames, target_size, scale,
                   strip_suffix=False, video_suffix=None, vid_id_map=None):
    """
    For each video: extract faces → save PNGs under output_dir/<video_id>/.
    Returns list of per-video stat dicts.

    vid_id_map: optional dict {vpath → vid_id} that overrides the default stem-based id.
                Used by path_list mode to guarantee uniqueness across subdirectories.
    """
    stats = []

    for vpath in tqdm(video_paths, desc=label_str, unit="video"):
        stem = os.path.splitext(os.path.basename(vpath))[0]
        if vid_id_map is not None:
            vid_id = vid_id_map[vpath]
        elif strip_suffix:
            vid_id = strip_generator_suffix(stem, video_suffix)
        else:
            vid_id = stem
        vid_save_dir = os.path.join(output_dir, vid_id)

        # Resume: skip if already processed
        existing = glob(os.path.join(vid_save_dir, "*.png"))
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
            stats.append({
                "video": vpath, "label": label_str,
                "extracted": 0, "no_face_frames": num_frames,
                "no_face_at_all": True, "skipped": False,
            })
            tqdm.write(f"[WARN] Cannot read video: {vpath}")
            continue

        no_face_count = 0
        saved_count = 0

        for frame_idx, frame_bgr in raw_frames:
            crop = detect_and_align(model, frame_bgr, target_size, scale)
            if crop is None:
                no_face_count += 1
                continue
            cv2.imwrite(os.path.join(vid_save_dir, f"frame_{frame_idx:06d}.png"), crop)
            saved_count += 1

        no_face_at_all = (saved_count == 0)
        stats.append({
            "video": vpath, "label": label_str,
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
        description="Preprocess SynthForensics videos with GenD face pipeline"
    )
    parser.add_argument("--mode", choices=["real", "fake", "path_list"], required=True,
                        help="'real': process real videos | 'fake': process fake videos | "
                             "'path_list': process absolute paths from a JSON file")
    parser.add_argument("--real_dir", default=None,
                        help="Folder of real .mp4 files (required for --mode real)")
    parser.add_argument("--fake_dir", default=None,
                        help="Folder of fake .mp4 files (required for --mode fake)")
    parser.add_argument("--path_list_file", default=None,
                        help="JSON file with a list of absolute video paths (required for --mode path_list)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (frames saved as <output_dir>/<video_id>/frame_XXXXXX.png)")
    parser.add_argument("--split_file", default=None,
                        help="JSON/TXT with fake video IDs to keep (for --mode fake)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--target_size", type=parse_target_size, default="256,256",
                        help="Face crop size as W,H (default: 256,256)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help="Face align scale factor (default: 1.3)")
    parser.add_argument("--det_thres", type=float, default=DEFAULT_DET_THRES,
                        help="RetinaFace detection threshold (default: 0.1)")
    parser.add_argument("--txt_out", default=None,
                        help="Path of the .txt file listing all saved PNG paths (auto-derived if not set)")
    parser.add_argument("--video_suffix", default=None,
                        help="Explicit generator suffix in filenames (e.g. 'ltx_2-3_t2v' for ltx2-3). "
                             "Auto-derived via rsplit if not set.")
    args = parser.parse_args()

    if args.mode == "real" and not args.real_dir:
        parser.error("--real_dir is required for --mode real")
    if args.mode == "fake" and not args.fake_dir:
        parser.error("--fake_dir is required for --mode fake")
    if args.mode == "path_list" and not args.path_list_file:
        parser.error("--path_list_file is required for --mode path_list")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GenD RetinaFace ONNX model...")
    model = prepare_model(args.det_thres)
    print(f"Target size: {args.target_size}, scale: {args.scale}, det_thres: {args.det_thres}")

    all_stats = []

    if args.mode == "real":
        videos = get_video_paths(args.real_dir)
        print(f"Real videos found: {len(videos)}")
        stats = process_videos(
            model, videos, "real", args.output_dir,
            args.num_frames, args.target_size, args.scale,
            strip_suffix=False,
        )
        all_stats += stats
        stats_path = os.path.join(args.output_dir, "extraction_stats_real.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    elif args.mode == "path_list":
        with open(args.path_list_file) as f:
            video_paths = json.load(f)
        print(f"Videos from path list: {len(video_paths)}")

        # Build vid_id_map: use path relative to common prefix, with '/' replaced by '_',
        # so that videos with the same filename in different subdirectories get unique IDs.
        # e.g. .../videos/A/1/clip.mp4 → A_1_clip
        #      .../videos/B/2/clip.mp4 → B_2_clip
        common = os.path.commonpath(video_paths) if len(video_paths) > 1 else os.path.dirname(video_paths[0])
        vid_id_map = {
            vpath: os.path.splitext(os.path.relpath(vpath, common))[0].replace(os.sep, "_")
            for vpath in video_paths
        }

        stats = process_videos(
            model, video_paths, "fake", args.output_dir,
            args.num_frames, args.target_size, args.scale,
            strip_suffix=False, vid_id_map=vid_id_map,
        )
        all_stats += stats
        stats_path = os.path.join(args.output_dir, "extraction_stats_fake.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    else:  # fake
        videos = get_video_paths(args.fake_dir)
        if args.video_suffix:
            print(f"Video suffix: '{args.video_suffix}'")
        if args.split_file:
            split_ids = load_split_ids(args.split_file)
            videos = filter_by_split(videos, split_ids, args.video_suffix)
            print(f"Fake videos: {len(videos)} (after split filter)")
        else:
            print(f"Fake videos: {len(videos)} (no split filter)")
        stats = process_videos(
            model, videos, "fake", args.output_dir,
            args.num_frames, args.target_size, args.scale,
            strip_suffix=True, video_suffix=args.video_suffix,
        )
        all_stats += stats
        stats_path = os.path.join(args.output_dir, "extraction_stats_fake.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    print_report(all_stats, args.num_frames)
    n_saved = sum(s["extracted"] for s in all_stats)
    print(f"Total frames saved: {n_saved}")

    # -----------------------------------------------------------------
    # Write txt file listing all saved PNG paths (for GenD's data loader)
    # -----------------------------------------------------------------
    txt_out = args.txt_out
    if txt_out is None:
        # default: <output_dir>.txt  (sibling of the output directory)
        txt_out = args.output_dir.rstrip("/") + ".txt"

    os.makedirs(os.path.dirname(os.path.abspath(txt_out)), exist_ok=True)
    png_paths = sorted(glob(os.path.join(args.output_dir, "**", "*.png"), recursive=True))
    with open(txt_out, "w") as f:
        f.write("\n".join(png_paths) + "\n")
    print(f"Written {len(png_paths)} paths → {txt_out}")


if __name__ == "__main__":
    main()
