"""
Step 1 — Preprocess real videos.

Extracts frames and generates VQVAE reconstructions for every
(split, compression) combination.

Output layout:
  preprocessing/recons/reals/{split}/{compression}/0_real/
    original/  {video_id}_{frame_idx}.jpg
    recons/    {video_id}_{frame_idx}.jpg

A .done sentinel file is written on completion; existing combinations
that already have .done are skipped automatically.

Usage (from MM-Det root):
  python preprocessing/01_preprocess_reals.py
"""
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.pipeline_config import (
    VQVAE_CKPT, COMPRESSIONS, SPLITS,
    REAL_DATASET_TEMPLATES, reals_recons_dir, is_done, mark_done,
)
from models import VectorQuantizedVAE

SUPPORTED_EXTS = {".mp4", ".avi", ".mpeg", ".wmv", ".mov", ".flv"}
RECONS_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def denormalize(img_t):
    return (img_t * 0.5 + 0.5).clamp(0, 1)


def process_video(video_path, out_dir, model, device):
    orig_dir  = os.path.join(out_dir, "original")
    recon_dir = os.path.join(out_dir, "recons")
    os.makedirs(orig_dir,  exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save original frame
        cv2.imwrite(os.path.join(orig_dir, f"{video_id}_{frame_idx}.jpg"), frame)
        # VQVAE reconstruction
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = RECONS_TRANSFORM(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _ = model(img_t)
            if recon.shape != img_t.shape:
                recon = F.interpolate(recon, img_t.shape[-2:], mode="nearest")
            recon_np = rearrange(denormalize(recon.squeeze(0)).cpu().numpy(), "c h w -> h w c")
            Image.fromarray(np.uint8(recon_np * 255)).save(
                os.path.join(recon_dir, f"{video_id}_{frame_idx}.jpg")
            )
        frame_idx += 1
    cap.release()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading VQVAE...")
    vqvae = VectorQuantizedVAE(3, 256, 512)
    vqvae.load_state_dict(torch.load(VQVAE_CKPT, map_location=device))
    vqvae.to(device).eval()

    total = len(SPLITS) * len(COMPRESSIONS)
    done_count = skipped = 0

    for split in SPLITS:
        for compression in COMPRESSIONS:
            out_dir = reals_recons_dir(split, compression)
            if is_done(out_dir):
                print(f"  [SKIP] reals/{split}/{compression}")
                skipped += 1
                continue

            src = REAL_DATASET_TEMPLATES[split].format(compression=compression)
            if not os.path.isdir(src):
                print(f"  [WARN] Not found: {src}")
                continue

            videos = sorted(
                f for f in os.listdir(src)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
            )
            print(f"  [{split}/{compression}] {len(videos)} videos")
            for v in tqdm(videos, desc=f"reals/{split}/{compression}", leave=False):
                process_video(os.path.join(src, v), out_dir, vqvae, device)

            mark_done(out_dir)
            done_count += 1
            print(f"  [DONE] reals/{split}/{compression} ({done_count}/{total - skipped})")

    print(f"\nReals preprocessing complete. Done: {done_count}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
