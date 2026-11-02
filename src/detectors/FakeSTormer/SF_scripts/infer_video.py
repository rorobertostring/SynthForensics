#-*- coding: utf-8 -*-
"""
Single-video deepfake inference with FakeSTormer.

Usage:
    python infer_video.py --cfg configs/temporal/FakeSFormer_base_c23.yaml \
                          --video path/to/video.mp4 \
                          [--num_frames 32] [--padding 0.25] [--gpu 0]

The script:
  1. Extracts face crops from the video using RetinaFace (--num_frames crops, default 32)
  2. Splits them into non-overlapping chunks of 4 frames (from config NUM_FRAMES)
  3. Runs inference on every chunk
  4. Averages the per-chunk scores → final verdict
"""

import os
import sys
import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from configs.get_config import load_config
from models import MODELS, build_model, load_pretrained
from package_utils.transform import final_transform
from losses.losses import _sigmoid


# ─────────────────────────── helpers ────────────────────────────────────────

def extract_face_crops(video_path: str, num_frames: int, padding: float, device: torch.device):
    """
    Extract face crops from a video using RetinaFace.
    Returns a list of PIL Images (RGB, 256×256), one per extracted frame.
    Keeps only the largest/highest-confidence face per frame.
    """
    try:
        from retinaface.pre_trained_models import get_model as get_retina
    except ImportError:
        raise ImportError("Install retinaface-pytorch:  pip install retinaface-pytorch")

    retina = get_retina("resnet50_2020-07-20", max_size=2048, device=device)
    retina.eval()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Cannot read video: {video_path}")

    # Uniform sampling across the whole video
    frame_idxs = set(np.linspace(0, total_frames - 1, num_frames, endpoint=True, dtype=np.int64))

    crops = []
    for cnt in tqdm(range(total_frames), desc="Extracting faces", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt not in frame_idxs:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        faces = retina.predict_jsons(frame_rgb)
        if not faces:
            continue

        # Pick largest face with highest score
        best = max(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0]) * (f['bbox'][3]-f['bbox'][1]))
        x0, y0, x1, y1 = best['bbox']
        fw, fh = x1 - x0, y1 - y0
        x0c = max(0, int(x0 - fw * padding))
        x1c = min(w,  int(x1 + fw * padding))
        y0c = max(0, int(y0 - fh * padding))
        y1c = min(h,  int(y1 + fh * padding))

        crop_bgr = frame[y0c:y1c, x0c:x1c]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(cv2.resize(crop_rgb, (256, 256)))
        crops.append(crop_pil)

    cap.release()
    return crops


def build_clips(crops: list, clip_len: int):
    """
    Split face crops into non-overlapping chunks of clip_len frames.
    Trailing crops that don't fill a full chunk are discarded.
    """
    n = len(crops)
    if n < clip_len:
        raise ValueError(f"Only {n} face crops extracted, need at least {clip_len}. "
                         "Check face detection on this video.")
    n_clips = n // clip_len
    return [crops[i * clip_len : (i + 1) * clip_len] for i in range(n_clips)]


def preprocess_clip(frames: list, img_size: list, transform) -> torch.Tensor:
    """
    Resize frames, apply normalization, return tensor (C, T, H, W).
    Mirrors what test.py does for test_vid.
    """
    tensors = []
    for pil_img in frames:
        img = pil_img.resize((img_size[0], img_size[1]))
        # Crop 15px border
        W, H = img.size
        img = img.crop((15, 15, W - 15, H - 15))
        img = img.resize((img_size[0], img_size[1]))
        arr = np.array(img) / 255.0
        t = transform(arr).to(torch.float)
        tensors.append(t)

    # Stack: (T, C, H, W) → transpose → (C, T, H, W) → unsqueeze batch → (1, C, T, H, W)
    stacked = torch.stack(tensors, dim=0)           # (T, C, H, W)
    stacked = stacked.transpose(0, 1).unsqueeze(0)  # (1, C, T, H, W)
    return stacked


# ─────────────────────────── main ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("FakeSTormer — single video inference")
    p.add_argument("--cfg",        "-c", required=True,  help="Config YAML file")
    p.add_argument("--video",      "-v", required=True,  help="Path to input video")
    p.add_argument("--num_frames",       default=32, type=int,   help="Face frames to extract from the video (default 32)")
    p.add_argument("--padding",          default=0.25, type=float, help="Face bbox padding for RetinaFace (default 0.25)")
    p.add_argument("--gpu",              default=0,  type=int,   help="GPU index (default 0)")
    p.add_argument("--threshold",        default=None, type=float, help="Override config threshold")
    return p.parse_args()


def main():
    args = parse_args()
    assert os.path.isfile(args.video), f"Video not found: {args.video}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.cfg)
    clip_len   = cfg.DATASET.DATA.SAMPLES_PER_VIDEO.NUM_FRAMES  # 4
    img_size   = cfg.DATASET.IMAGE_SIZE                          # [224, 224]
    threshold  = args.threshold if args.threshold is not None else cfg.TEST.threshold
    transform  = final_transform(cfg.DATASET)

    # ── build model ──────────────────────────────────────────────────────────
    print(f"Loading model weights: {cfg.TEST.pretrained}")
    model = build_model(cfg.MODEL, MODELS).to(torch.float)
    model = load_pretrained(model, cfg.TEST.pretrained)
    model = nn.DataParallel(model, device_ids=[args.gpu]).to(device)
    model.eval()

    # ── extract faces ────────────────────────────────────────────────────────
    print(f"Extracting {args.num_frames} face crops from: {args.video}")
    crops = extract_face_crops(args.video, args.num_frames, args.padding, device)
    print(f"  → {len(crops)} faces detected")

    # ── build clips ──────────────────────────────────────────────────────────
    clips = build_clips(crops, clip_len)
    print(f"Running inference on {len(clips)} clips of {clip_len} frames each ({len(crops)} crops extracted)...")

    # ── inference ────────────────────────────────────────────────────────────
    scores = []
    with torch.no_grad():
        for clip in tqdm(clips, desc="Clips"):
            inp = preprocess_clip(clip, img_size, transform).to(device)
            outputs = model(inp)
            out = outputs[0] if isinstance(outputs, list) else outputs
            if isinstance(out, dict):
                # Use temporal localisation score
                score = out['temp_loc'].sigmoid().cpu().numpy()[0][-1]
            else:
                score = out.sigmoid().cpu().numpy()[0][-1]
            scores.append(float(score))

    # ── aggregate ────────────────────────────────────────────────────────────
    mean_score = float(np.mean(scores))
    label = "FAKE" if mean_score > threshold else "REAL"

    print("\n" + "="*50)
    print(f"  Video   : {os.path.basename(args.video)}")
    print(f"  Score   : {mean_score:.4f}  (threshold: {threshold})")
    print(f"  Verdict : {label}")
    print(f"  Clips   : {[round(s,4) for s in scores]}")
    print("="*50)


if __name__ == "__main__":
    main()
