"""
Step 2 — Preprocess fake videos.

Extracts frames and generates VQVAE reconstructions for every
(vtype, generator, split, compression) combination.
Generators are discovered dynamically from the dataset folders.
Videos are filtered against the split ID txt files.

Output layout:
  preprocessing/recons/fakes/{vtype}/{generator}/{split}/{compression}/1_fake/
    original/  {video_id}_{frame_idx}.jpg
    recons/    {video_id}_{frame_idx}.jpg

Usage (from MM-Det root):
  python preprocessing/02_preprocess_fakes.py
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
    SF_T2V_ROOT, SF_I2V_ROOT,
    fakes_recons_dir, load_split_ids, discover_generators,
    is_done, mark_done, all_combinations,
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
        cv2.imwrite(os.path.join(orig_dir, f"{video_id}_{frame_idx}.jpg"), frame)
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


def get_videos_for_split(vtype, generator, split, compression):
    """
    Return paths of fake videos matching the split IDs.

    Generator folder names don't always match the filename token
    (e.g. folder 'ltx2-3' → filename '000_ltx_2-3_t2v.mp4'), so we
    match by prefix: any file whose stem starts with '{split_id}_'.
    For i2v we additionally require the stem ends with '_i2v'.
    """
    root = SF_T2V_ROOT if vtype == "t2v" else SF_I2V_ROOT
    src_dir = os.path.join(root, compression, generator)
    if not os.path.isdir(src_dir):
        return []

    split_ids = set(load_split_ids(split))
    matched = []
    for fn in os.listdir(src_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() != ".mp4":
            continue
        if vtype == "i2v" and not stem.endswith("_i2v"):
            continue
        # stem starts with one of the split IDs followed by '_'
        for sid in split_ids:
            if stem.startswith(sid + "_"):
                matched.append(os.path.join(src_dir, fn))
                break
    return sorted(matched)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading VQVAE...")
    vqvae = VectorQuantizedVAE(3, 256, 512)
    vqvae.load_state_dict(torch.load(VQVAE_CKPT, map_location=device))
    vqvae.to(device).eval()

    done_count = skipped = 0

    for vtype, generator, split, compression in all_combinations():
        out_dir = fakes_recons_dir(vtype, generator, split, compression)
        if is_done(out_dir):
            skipped += 1
            continue

        videos = get_videos_for_split(vtype, generator, split, compression)
        if not videos:
            print(f"  [WARN] No videos: {vtype}/{generator}/{split}/{compression}")
            continue

        label = f"{vtype}/{generator}/{split}/{compression}"
        print(f"  [{label}] {len(videos)} videos")
        for vp in tqdm(videos, desc=label, leave=False):
            process_video(vp, out_dir, vqvae, device)

        mark_done(out_dir)
        done_count += 1

    print(f"\nFakes preprocessing complete. Done: {done_count}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
