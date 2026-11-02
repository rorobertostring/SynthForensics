#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-level deepfake detection with LAA-Net.

Pipeline (mirrors the official eval mode):
  1. Sample N frames uniformly from the video
  2. Detect the largest face per frame with RetinaFace (padding 25%)
  3. Preprocess each crop exactly as test.py test_img mode does
  4. Run LAA-Net inference per frame
  5. Aggregate frame scores with mean  →  video verdict

Usage:
    python predict_video.py \
        --video  path/to/video.mp4 \
        --cfg    configs/efn4_fpn_hm_adv.yaml \
        --weights logs/<your_weights>.pth \
        [--num_frames 32] \
        [--threshold 0.5] \
        [--gpu 0] \
        [--save_heatmap]
"""

import os
import sys
import argparse

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from retinaface.pre_trained_models import get_model as get_retinaface

from configs.get_config import load_config
from models import build_model, load_pretrained, MODELS
from package_utils.transform import final_transform, get_center_scale, get_affine_transform
from package_utils.utils import vis_heatmap
from losses.losses import _sigmoid

# Face crop parameters — same values as images_crop.py
FACE_PADDING   = 0.25
FACE_SAVE_SIZE = 256   # intermediate size before the affine warp (px)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LAA-Net single-video inference")
    p.add_argument("--video",       required=True,  help="Path to input video file")
    p.add_argument("--cfg",         required=True,  help="Config YAML (e.g. configs/efn4_fpn_hm_adv.yaml)")
    p.add_argument("--weights",     required=True,  help="Path to pretrained .pth weights")
    p.add_argument("--num_frames",  type=int, default=32,
                   help="Number of frames sampled uniformly (default: 32)")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Classification threshold (default: 0.5)")
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index to use (default: 0)")
    p.add_argument("--save_heatmap", action="store_true",
                   help="Save artifact heatmap overlay for each frame to ./heatmaps/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Face detection helpers  (mirrors images_crop.py facecrop logic)
# ---------------------------------------------------------------------------

def detect_largest_face(retina_model, frame_rgb, padding=FACE_PADDING):
    """
    Run RetinaFace on an RGB frame, return the crop of the largest detected face
    (with padding) as an RGB numpy array, or None if no face is found.
    """
    h, w = frame_rgb.shape[:2]
    faces = retina_model.predict_jsons(frame_rgb)

    if not faces:
        return None

    best_crop  = None
    best_area  = -1
    best_score = -1

    for face in faces:
        x0, y0, x1, y1 = face["bbox"]
        fw   = x1 - x0
        fh   = y1 - y0
        area = fw * fh
        score = face["score"]

        if area > best_area and score > best_score:
            # add padding and clamp to frame boundaries
            cx0 = max(0, int(x0 - fw * padding))
            cx1 = min(w, int(x1 + fw * padding))
            cy0 = max(0, int(y0 - fh * padding))
            cy1 = min(h, int(y1 + fh * padding))

            best_crop  = frame_rgb[cy0:cy1, cx0:cx1, :]
            best_area  = area
            best_score = score

    return best_crop


# ---------------------------------------------------------------------------
# Preprocessing  (mirrors test.py test_img preprocessing exactly)
# ---------------------------------------------------------------------------

def preprocess_face(face_rgb, cfg, transforms):
    """
    Replicate the preprocessing chain used in test.py test_img mode:
      1. Resize to 317×317
      2. Crop borders  [60:317, 30:287]
      3. Affine-warp to IMAGE_SIZE
      4. Normalise /255 + torchvision Normalize
    Returns a (1, 3, H, W) float64 tensor ready for the model.
    """
    # 1. Resize to intermediate size (same as test.py)
    img = cv2.resize(face_rgb, (317, 317), interpolation=cv2.INTER_LINEAR)

    # 2. Hardcoded border crop (same as test.py)
    img = img[60:317, 30:287, :]   # → ~257 × 257

    # 3. Compute affine transform and warp to model input size
    image_size   = cfg.DATASET.IMAGE_SIZE          # [384, 384]
    aspect_ratio = image_size[1] / image_size[0]   # 1.0 for square
    pixel_std    = 200
    rot          = 0

    c, s = get_center_scale(img.shape[:2], aspect_ratio, pixel_std)
    trans = get_affine_transform(c, s, rot, image_size)
    img = cv2.warpAffine(
        img, trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR,
    )

    # 4. To tensor + normalize
    tensor = transforms(img / 255.0).to(torch.float64)
    return tensor.unsqueeze(0)   # add batch dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.cfg)
    # Override pretrained path from CLI so user doesn't need to edit YAML
    cfg.TEST.pretrained = args.weights
    transforms = final_transform(cfg.DATASET)

    # ── Model ───────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model(cfg.MODEL, MODELS).to(torch.float64)
    model = load_pretrained(model, cfg.TEST.pretrained)
    model = model.to(device)
    model.eval()
    print(f"[INFO] Weights loaded from: {cfg.TEST.pretrained}")

    # ── RetinaFace ──────────────────────────────────────────────────────────
    retina = get_retinaface("resnet50_2020-07-20", max_size=2048, device=device)
    retina.eval()

    # ── Video ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video has 0 frames or could not be read.")

    # Uniform frame indices (same as images_crop.py / common.py sampler)
    frame_idxs = set(
        np.linspace(0, total_frames - 1, args.num_frames, endpoint=True, dtype=np.int64).tolist()
    )
    print(f"[INFO] Video: {args.video}  |  total frames: {total_frames}  |  sampling: {len(frame_idxs)}")

    if args.save_heatmap:
        os.makedirs("heatmaps", exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────────────
    frame_scores = []
    frames_processed = 0
    frames_no_face   = 0

    pbar = tqdm(total=len(frame_idxs), desc="Analysing frames", unit="frame")
    cnt  = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if cnt not in frame_idxs:
            cnt += 1
            continue

        # Convert BGR → RGB (RetinaFace and preprocessing expect RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_crop = detect_largest_face(retina, frame_rgb)
        if face_crop is None:
            tqdm.write(f"  [WARN] No face detected at frame {cnt} — skipped")
            frames_no_face += 1
            cnt += 1
            pbar.update(1)
            continue

        # Resize to FACE_SAVE_SIZE like images_crop.py saves crops
        face_crop = cv2.resize(face_crop, (FACE_SAVE_SIZE, FACE_SAVE_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        tensor = preprocess_face(face_crop, cfg, transforms).to(device)

        with torch.no_grad():
            outputs = model(tensor)

        # Model returns list of dicts; first element has 'hm' and 'cls'
        out = outputs[0] if isinstance(outputs, list) else outputs
        if isinstance(out, dict):
            hm_out  = out["hm"]
            cls_out = out["cls"]
        else:
            raise RuntimeError("Unexpected model output format.")

        # The model outputs raw logits: apply sigmoid so scores are in [0, 1].
        score = float(torch.sigmoid(cls_out).cpu().numpy()[0][-1])
        frame_scores.append(score)
        frames_processed += 1

        if args.save_heatmap:
            hm_pred = _sigmoid(hm_out).cpu().numpy()
            heatmap_path = os.path.join("heatmaps", f"frame_{cnt:05d}.jpg")
            vis_heatmap(face_crop, hm_pred[0], heatmap_path)

        cnt += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # ── Aggregation (mean — standard in FF++ / LAA-Net eval) ─────────────────
    if not frame_scores:
        print("[ERROR] No frames with a detected face were processed. Cannot classify.")
        return

    video_score = float(np.mean(frame_scores))
    verdict     = "FAKE" if video_score > args.threshold else "REAL"

    # ── Results ──────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print(f"  Video         : {os.path.basename(args.video)}")
    print(f"  Frames used   : {frames_processed}  (skipped no-face: {frames_no_face})")
    print(f"  Video score   : {video_score:.4f}  (threshold: {args.threshold})")
    print(f"  Verdict       : {verdict}")
    print("=" * 50)

    if args.save_heatmap:
        print(f"  Heatmaps saved to: ./heatmaps/")


if __name__ == "__main__":
    main()
