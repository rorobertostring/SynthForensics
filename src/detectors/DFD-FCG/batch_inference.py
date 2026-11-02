#!/usr/bin/env python3
"""
DFD-FCG batch inference.

Loads the model ONCE and runs inference on all videos in a folder.
Preprocessing results (landmarks + cropped faces) are cached in
  <script_dir>/already_processed/<cache_key>/
so that re-running on the same folder skips the expensive preprocessing.

Usage (single fake folder):
    python batch_inference.py <FAKE_FOLDER> --real-folder <REAL_FOLDER> --cache-key <KEY> [options]

Usage (multiple fake folders, model loaded once):
    python batch_inference.py <FAKE1> <FAKE2> ... \
        --real-folder <REAL_FOLDER> \
        --cache-keys <KEY1> <KEY2> ... \
        --real-cache-key <REAL_KEY>

The output CSV is saved automatically to results/<cache_key>.csv per fake folder.
The CSV starts with metric rows (metric,value), then a blank row, then per-video rows.
"""

import os
import sys
import csv
import glob
import math
import tempfile
import logging
import argparse
import warnings
import subprocess

import yaml
import numpy as np
import torch
from sklearn import metrics as sk_metrics
from tqdm import tqdm
from torchvision.io import VideoReader

from src.utility.builtin import ODTrainer, ODLightningCLI


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DFD-FCG batch inference")
    parser.add_argument("video_folder", type=str, nargs="+",
                        help="One or more folders containing FAKE input videos")
    parser.add_argument("--real-folder", type=str, default=None,
                        help="Folder containing REAL input videos. "
                             "If provided, metrics (AUC, EER, AP, ACC) are computed and printed.")
    parser.add_argument("--real-cache-key", type=str, default=None,
                        help="Cache key for the real folder (default: basename of real_folder)")
    parser.add_argument("--model-cfg", type=str,
                        default="weights/zero_shot/setting.yaml")
    parser.add_argument("--model-ckpt", type=str,
                        default="weights/zero_shot/weights.ckpt")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for REAL/FAKE label")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Clip batch size (~10 GB VRAM at 30)")
    parser.add_argument("--precision", type=str, default="16")
    parser.add_argument("--ext", type=str, default="mp4",
                        help="Extension of input videos (without dot)")
    parser.add_argument("--cache-keys", type=str, nargs="+", default=None,
                        help="Cache key(s) for fake folder(s), one per folder (default: <parent>_<folder>, "
                             "unique across splits and compressions). "
                             "Also accepted as --cache-key for single-folder usage.")
    parser.add_argument("--real-csv", type=str, default=None,
                        help="Path to an existing CSV with real video rows — skips real inference entirely.")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="Re-run preprocessing even if cache exists")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for result CSVs (default: <script_dir>/results/)")
    parser.add_argument("--output-keys", type=str, nargs="+", default=None,
                        help="Output CSV names, one per fake folder. If omitted, falls back to --cache-keys.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics (same formulas as DeepfakeBench training/metrics/utils.py)
# ---------------------------------------------------------------------------

def compute_metrics(y_pred, y_true):
    """
    y_pred: np.array of fake-probability scores (float, 0=real 1=fake)
    y_true: np.array of ground-truth labels    (int,   0=real 1=fake)
    Since DFD-FCG is video-level, video_auc == auc.
    """
    y_pred = np.array(y_pred, dtype=float).squeeze()
    y_true = np.array(y_true, dtype=int)
    y_true[y_true >= 1] = 1

    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = sk_metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    eer = float(fpr[np.nanargmin(np.absolute(fnr - fpr))])
    ap  = sk_metrics.average_precision_score(y_true, y_pred)
    pred_class = (y_pred > 0.5).astype(int)
    acc = (pred_class == np.clip(y_true, 0, 1)).sum().item() / len(pred_class)
    return {"acc": round(acc, 4), "auc": round(auc, 4),
            "eer": round(eer, 4), "ap": round(ap, 4), "video_auc": round(auc, 4)}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def default_cache_key(folder):
    """<parent>_<folder>, e.g. SF-FF_fake_test_set_crf23_cogvideox."""
    folder = folder.rstrip("/")
    return f"{os.path.basename(os.path.dirname(folder))}_{os.path.basename(folder)}"


def setup_cache(video_folder, script_dir, cache_key, ext):
    """
    Creates (or reuses) the cache directory for this video folder.
    Returns (cache_dir, videos, needs_preprocessing).

      cache_dir/
        videos/      -> symlink to video_folder
        frame_data/  <- persists across runs
        cropped/     <- persists across runs
    """
    cache_base = os.path.join(script_dir, "already_processed")
    cache_dir = os.path.join(cache_base, cache_key)
    videos_link = os.path.join(cache_dir, "videos")

    os.makedirs(cache_dir, exist_ok=True)

    # Create/update symlink to the actual video folder
    if os.path.islink(videos_link):
        if os.readlink(videos_link) != video_folder:
            os.remove(videos_link)
            os.symlink(video_folder, videos_link)
    else:
        os.symlink(video_folder, videos_link)

    # List all videos in the folder
    videos = sorted(glob.glob(os.path.join(video_folder, f"*.{ext}")))
    if not videos:
        raise FileNotFoundError(
            f"No .{ext} files found in {video_folder}"
        )

    # Check if preprocessing is complete (one pickle per video)
    fdata_dir = os.path.join(cache_dir, "frame_data")
    existing_pickles = glob.glob(os.path.join(fdata_dir, "*.pickle"))
    missing = len(videos) - len(existing_pickles)
    needs_preprocessing = missing > 0

    return cache_dir, videos, needs_preprocessing, missing


# ---------------------------------------------------------------------------
# Preprocessing (subprocess to handle the interactive input() in fetch_landmark_bbox)
# ---------------------------------------------------------------------------

def run_preprocessing(cache_dir, script_dir, mean_face):
    print("[1/3] Fetching landmarks & bboxes...")
    subprocess.run(
        [sys.executable, "-m", "src.preprocess.fetch_landmark_bbox",
         "--root-dir", cache_dir,
         "--video-dir", "videos",
         "--fdata-dir", "frame_data",
         "--glob-exp", "*"],
        input="y\n",
        text=True,
        check=True,
        cwd=script_dir,
    )

    print("[2/3] Cropping faces...")
    subprocess.run(
        [sys.executable, "-m", "src.preprocess.crop_main_face",
         "--root-dir", cache_dir,
         "--video-dir", "videos",
         "--fdata-dir", "frame_data",
         "--crop-dir", "cropped",
         "--glob-exp", "*",
         "--mean-face", mean_face],
        check=True,
        cwd=script_dir,
    )


# ---------------------------------------------------------------------------
# Per-video inference (model already in GPU)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def infer_video(model, video_path, batch_size):
    """
    Runs inference on a single pre-processed video.
    Returns the mean fake probability score (0=real, 1=fake).
    """
    transforms = model.transform
    stride = 0.333
    vid_ext = os.path.splitext(video_path)[-1]

    # Load cropped video from cache (produced by crop_main_face)
    cropped_path = (video_path
                    .replace("/videos", "/cropped/videos")
                    .replace(vid_ext, ".avi"))

    vid_reader = VideoReader(cropped_path, "video", num_threads=8)
    fps = vid_reader.get_metadata()["video"]["fps"][0]
    cropped_frames = torch.stack([f["data"] for f in vid_reader]).pin_memory()
    del vid_reader

    indices = torch.tensor(
        [int(math.floor(i * stride * fps)) for i in range(10)],
        dtype=torch.long
    )
    probs = []
    i = 0
    clip_count = len(cropped_frames) - indices[-1]
    while i < clip_count:
        batch = min(clip_count - i, batch_size)
        batch_frames = torch.stack([
            cropped_frames[indices + i + j] for j in range(batch)
        ]).to("cuda", non_blocking=True)
        clips = torch.stack([transforms(f) for f in batch_frames])
        results = model.evaluate(clips)
        probs.extend(
            results["logits"].softmax(dim=-1)[:, 1].flatten().cpu().tolist()
        )
        i += batch

    return float(np.mean(probs)) if probs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_folder(folder, script_dir, model, args, true_label, label_name, output_csv=None, cache_key=None):
    """Run inference on all videos in folder. Returns list of (name, score, true_label) rows.
    If output_csv is provided, writes rows incrementally (flush after each video).
    """
    if cache_key is None:
        cache_key = default_cache_key(folder)
    cache_dir, videos, needs_preprocessing, missing = setup_cache(
        folder, script_dir, cache_key, args.ext
    )
    print(f"\n[{label_name}] folder  : {folder}")
    print(f"[{label_name}] cache   : {cache_dir}")
    print(f"[{label_name}] videos  : {len(videos)}")

    mean_face = os.path.join(script_dir, "misc", "20words_mean_face.npy")
    if args.force_preprocess or needs_preprocessing:
        if needs_preprocessing:
            print(f"[{label_name}] Preprocessing cache incomplete, running... ({missing} videos missing)")
        run_preprocessing(cache_dir, script_dir, mean_face)
    else:
        print(f"[{label_name}] Preprocessing cache found, skipping.")

    cached_videos = sorted(glob.glob(os.path.join(cache_dir, "videos", f"*.{args.ext}")))
    rows = []

    # Resume: read already-processed rows from existing CSV
    already_done = {}  # name -> row dict
    csv_f = None
    writer = None
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        if os.path.exists(output_csv):
            with open(output_csv, newline="") as existing_f:
                reader = csv.DictReader(existing_f)
                for r in reader:
                    if r.get("video"):
                        already_done[r["video"]] = {
                            "video": r["video"],
                            "score": float(r["score"]) if r["score"] not in (None, "", "None") else None,
                            "true_label": int(r["true_label"]) if r["true_label"] not in (None, "", "None") else true_label,
                            "pred_label": r["pred_label"],
                            "error": r["error"],
                        }
            rows = list(already_done.values())
            print(f"[{label_name}] Resume: {len(already_done)} videos already processed, "
                  f"{len(cached_videos) - len(already_done)} remaining.")
            csv_f = open(output_csv, "a", newline="")  # append the remaining videos
        else:
            csv_f = open(output_csv, "w", newline="")
            writer = csv.writer(csv_f)
            writer.writerow(["video", "score", "true_label", "pred_label", "error"])
            csv_f.flush()
        if writer is None:
            writer = csv.writer(csv_f)

    try:
        pbar = tqdm(cached_videos, unit="video", desc=label_name)
        for video_path in pbar:
            name = os.path.splitext(os.path.basename(video_path))[0]
            vid_ext = os.path.splitext(video_path)[-1]
            cropped_path = (video_path
                            .replace("/videos", "/cropped/videos")
                            .replace(vid_ext, ".avi"))
            pbar.set_postfix_str(name[:40])
            if name in already_done:
                continue
            if not os.path.exists(cropped_path):
                tqdm.write(f"  {name} -> NO_FACE")
                row = {"video": name, "score": None, "true_label": true_label,
                       "pred_label": "NO_FACE", "error": ""}
            else:
                try:
                    score = infer_video(model, video_path, args.batch_size)
                    pred = "FAKE" if score is not None and score >= args.threshold else "REAL"
                    row = {"video": name, "score": round(score, 4) if score is not None else None,
                           "true_label": true_label, "pred_label": pred, "error": ""}
                except Exception as e:
                    tqdm.write(f"  {name} -> ERROR: {e}")
                    row = {"video": name, "score": None, "true_label": true_label,
                           "pred_label": "ERROR", "error": str(e)}
            rows.append(row)
            if writer is not None:
                writer.writerow([row["video"], row["score"], row["true_label"],
                                 row["pred_label"], row["error"]])
                csv_f.flush()
    finally:
        if csv_f is not None:
            csv_f.close()

    return rows


def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level="WARNING")

    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    fake_folders = [os.path.abspath(f) for f in args.video_folder]
    cache_keys   = args.cache_keys or [default_cache_key(f) for f in fake_folders]
    output_keys  = args.output_keys or cache_keys
    if len(cache_keys) != len(fake_folders):
        raise ValueError(f"--cache-keys must have the same number of entries as fake folders "
                         f"({len(fake_folders)} folders, {len(cache_keys)} keys)")

    # Summary and confirmation
    print("\n========== Run Summary ==========")
    for folder, key in zip(fake_folders, cache_keys):
        n = len(glob.glob(os.path.join(folder, f"*.{args.ext}")))
        print(f"  [FAKE] {key}: {n} videos  ({folder})")
    if args.real_folder:
        real_abs = os.path.abspath(args.real_folder)
        n_real = len(glob.glob(os.path.join(real_abs, f"*.{args.ext}")))
        print(f"  [REAL] {n_real} videos  ({real_abs})")
    print("=================================")
    print("Proceeding automatically (non-interactive mode).")

    model_cfg = args.model_cfg if os.path.isabs(args.model_cfg) \
        else os.path.join(script_dir, args.model_cfg)
    model_ckpt = args.model_ckpt if os.path.isabs(args.model_ckpt) \
        else os.path.join(script_dir, args.model_ckpt)

    # Load model ONCE
    with open(model_cfg) as f:
        cfg = yaml.safe_load(f)
    if "lr_scheduler" in cfg and "verbose" in cfg["lr_scheduler"]:
        del cfg["lr_scheduler"]["verbose"]
    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=script_dir
    )
    yaml.dump(cfg, tmp_cfg)
    tmp_cfg.close()
    patched_cfg = tmp_cfg.name

    print("Loading model...")
    try:
        cli = ODLightningCLI(
            run=False,
            trainer_class=ODTrainer,
            save_config_callback=None,
            parser_kwargs={"parser_mode": "omegaconf"},
            auto_configure_optimizers=False,
            seed_everything_default=1019,
            args=[
                "-c", patched_cfg,
                "--trainer.logger=null",
                "--trainer.devices=1",
                f"--trainer.precision={args.precision}",
            ],
        )
    finally:
        os.unlink(patched_cfg)
    try:
        model = cli.model.__class__.load_from_checkpoint(model_ckpt)
    except Exception as e:
        print(f"  Strict load failed ({e}), retrying non-strict...")
        model = cli.model.__class__.load_from_checkpoint(model_ckpt, strict=False)
    model.eval()
    print("  Model ready.\n")

    # Run on real folder ONCE (cached), reused across all fake folders
    real_rows = []
    real_cache_key = None
    if args.real_csv:
        # Load real rows directly from existing CSV — skip inference entirely
        real_csv_path = os.path.abspath(args.real_csv)
        print(f"\n[REAL] Loading scores from existing CSV: {real_csv_path}")
        with open(real_csv_path, newline="") as f:
            # Skip metrics section — find the header line starting with "video"
            for line in f:
                if line.strip().startswith("video,"):
                    break
            reader = csv.DictReader(f, fieldnames=["video", "score", "true_label", "pred_label", "error"])
            for r in reader:
                if r.get("true_label") == "0" and r.get("video"):
                    real_rows.append({
                        "video": r["video"],
                        "score": float(r["score"]) if r["score"] not in (None, "", "None") else None,
                        "true_label": 0,
                        "pred_label": r["pred_label"],
                        "error": r["error"],
                    })
        print(f"[REAL] Loaded {len(real_rows)} real video scores.")
    elif args.real_folder:
        real_folder    = os.path.abspath(args.real_folder)
        real_cache_key = args.real_cache_key or os.path.basename(real_folder.rstrip("/"))
        real_rows = run_folder(real_folder, script_dir, model, args,
                               true_label=0, label_name="REAL", cache_key=real_cache_key)

    # Loop over fake folders — one CSV per folder
    for fake_folder, cache_key, out_key in zip(fake_folders, cache_keys, output_keys):
        print(f"\n{'='*60}")
        print(f"Processing: {cache_key}")
        print(f"{'='*60}")

        results_base = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(script_dir, "results")
        output_csv = os.path.join(results_base, f"{out_key}.csv")

        # Fake rows written incrementally during processing
        fake_rows = run_folder(fake_folder, script_dir, model, args,
                               true_label=1, label_name="FAKE",
                               output_csv=output_csv, cache_key=cache_key)
        all_rows = fake_rows + real_rows

        # Compute metrics on all rows (fake + real)
        m = None
        if real_rows:
            valid = [(r["score"], r["true_label"]) for r in all_rows if r["score"] is not None]
            if len(valid) >= 2:
                y_pred = np.array([v[0] for v in valid])
                y_true = np.array([v[1] for v in valid])
                m = compute_metrics(y_pred, y_true)

        # Rewrite the entire CSV: metrics on top, then all video rows
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            if m is not None:
                writer.writerow(["metric", "value"])
                for k, v in m.items():
                    writer.writerow([k, v])
                writer.writerow([])
            writer.writerow(["video", "score", "true_label", "pred_label", "error"])
            for r in all_rows:
                writer.writerow([r["video"], r["score"], r["true_label"],
                                 r["pred_label"], r["error"]])
        print(f"\nResults saved to: {output_csv}")

        if m is not None:
            print("\n========== Metrics ==========")
            for k, v in m.items():
                print(f"  {k}: {v}")
            print("==============================")


if __name__ == "__main__":
    main()
