#!/usr/bin/env python3
"""
GenConViT batch inference.

Runs detection on one or more FAKE video folders and one REAL video folder,
then computes metrics (AUC, EER, AP, ACC) matching DFD-FCG / DeepFakeBench.

Usage (single fake folder):
    python batch_inference.py \
        --fake-folder /path/to/fake \
        --real-folder /path/to/real \
        --output sf-ff

Usage (multiple fake folders, model loaded once):
    python batch_inference.py \
        --fake-folder /path/fake1 /path/fake2 \
        --real-folder /path/to/real \
        --output sf-ff sf-dfd

CSVs are saved to results/<output>.csv
"""

import os
import csv
import argparse

import numpy as np
from sklearn import metrics as sk_metrics

from model.pred_func import (
    load_genconvit,
    df_face,
    df_face_from_folder,
    pred_vid,
    is_video,
    is_video_folder,
    real_or_fake,
)
from model.config import load_config
import torch

config = load_config()


# ---------------------------------------------------------------------------
# Metrics (same formulas as DFD-FCG / DeepFakeBench)
# ---------------------------------------------------------------------------

def compute_metrics(y_pred, y_true):
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
# Inference loop over a folder (like vids() in prediction.py)
# ---------------------------------------------------------------------------

def run_folder(root_dir, model, num_frames, net, fp16, true_label):
    """
    Iterates over videos in root_dir and returns a list of dicts
    {video, score, true_label, pred_label, error}.
    true_label: 0 = REAL, 1 = FAKE
    """
    rows = []
    count = 0

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            is_vid_folder = is_video_folder(curr_vid)
            if not (is_video(curr_vid) or is_vid_folder):
                continue

            count += 1
            print(f"\n{count} Loading... {curr_vid}")

            if is_vid_folder:
                df = df_face_from_folder(curr_vid, num_frames)
            else:
                df = df_face(curr_vid, num_frames)

            if fp16 and len(df) >= 1:
                df = df.half()

            if len(df) < 1:
                print(f"NO_FACE: {filename}")
                rows.append({
                    "video":      filename,
                    "score":      None,
                    "true_label": true_label,
                    "pred_label": "NO_FACE",
                    "error":      "",
                })
                continue

            y, y_val = pred_vid(df, model)
            pred_label = real_or_fake(y)
            print(f"Prediction: {y_val:.4f}  {pred_label}")

            rows.append({
                "video":      filename,
                "score":      round(y_val, 4),
                "true_label": true_label,
                "pred_label": pred_label,
                "error":      "",
            })

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            rows.append({
                "video":      filename,
                "score":      None,
                "true_label": true_label,
                "pred_label": "ERROR",
                "error":      str(e),
            })

    return rows


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GenConViT batch inference")

    parser.add_argument("--fake-folder", type=str, nargs="+", required=True,
                        help="One or more FAKE video folders")
    parser.add_argument("--real-folder", type=str, required=True,
                        help="REAL video folder")
    parser.add_argument("--output", type=str, nargs="+", default=None,
                        help="Output CSV name(s), one per fake folder. Default: folder basename.")
    parser.add_argument("--f", dest="num_frames", type=int, default=15,
                        help="Frames per video (default: 15)")
    parser.add_argument("--e", dest="ed_weight", nargs="?", const="genconvit_ed_inference", default=None,
                        help="ED branch weights")
    parser.add_argument("--v", dest="vae_weight", nargs="?", const="genconvit_vae_inference", default=None,
                        help="VAE branch weights")
    parser.add_argument("--s", dest="size", choices=["tiny", "large"], default=None,
                        help="Backbone size")
    parser.add_argument("--fp16", action="store_true", help="Half precision")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for CSVs (default: results/)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    fake_folders = [os.path.abspath(f) for f in args.fake_folder]
    real_folder  = os.path.abspath(args.real_folder)

    if args.output:
        if len(args.output) != len(fake_folders):
            raise ValueError("--output must have the same number of entries as --fake-folder")
        output_names = args.output
    else:
        output_names = [os.path.basename(f.rstrip("/")) for f in fake_folders]

    # Determine net and weights
    ed_weight  = args.ed_weight
    vae_weight = args.vae_weight
    if ed_weight and vae_weight:
        net = "genconvit"
    elif ed_weight:
        net = "ed"
    elif vae_weight:
        net = "vae"
    else:
        net = "genconvit"
        ed_weight  = "genconvit_ed_inference"
        vae_weight = "genconvit_vae_inference"

    if args.size:
        config["model"]["backbone"] = f"convnext_{args.size}"
        config["model"]["embedder"] = f"swin_{args.size}_patch4_window7_224"
        config["model"]["type"]     = args.size

    print(f"\nUsing {net}\n")

    # Load model ONCE
    model = load_genconvit(config, net, ed_weight, vae_weight, args.fp16)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Run REAL inference once
    print(f"\n{'='*50}\n[REAL] {real_folder}\n{'='*50}")
    real_rows = run_folder(real_folder, model, args.num_frames, net, args.fp16, true_label=0)

    # Loop over fake folders — one CSV per folder
    for fake_folder, out_name in zip(fake_folders, output_names):
        print(f"\n{'='*50}\n[FAKE] {fake_folder}\n{'='*50}")
        fake_rows = run_folder(fake_folder, model, args.num_frames, net, args.fp16, true_label=1)

        all_rows = fake_rows + real_rows

        # Metrics
        valid = [(r["score"], r["true_label"]) for r in all_rows if r["score"] is not None]
        m = None
        if len(valid) >= 2:
            y_pred = np.array([v[0] for v in valid])
            y_true = np.array([v[1] for v in valid])
            m = compute_metrics(y_pred, y_true)

        # Save CSV: metrics → blank row → video rows
        output_csv = os.path.join(results_dir, f"{out_name}.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            if m is not None:
                writer.writerow(["metric", "value"])
                for k, v in m.items():
                    writer.writerow([k, v])
                writer.writerow([])
            writer.writerow(["video", "score", "true_label", "pred_label", "error"])
            for r in all_rows:
                writer.writerow([r["video"], r["score"], r["true_label"], r["pred_label"], r["error"]])

        print(f"\nResults saved to: {output_csv}")

        if m is not None:
            print("\n========== Metrics ==========")
            for k, v in m.items():
                print(f"  {k}: {v}")
            print("==============================")


if __name__ == "__main__":
    main()
