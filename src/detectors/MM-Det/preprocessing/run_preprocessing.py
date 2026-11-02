"""
Full preprocessing pipeline (steps 1 → 2 → 3).

Runs in order:
  1. Preprocess real videos   (frames + VQVAE recons)
  2. Preprocess fake videos   (frames + VQVAE recons)
  3. Cache MMFR               (LLaVA features)

Each step skips already-completed work automatically.
The VQVAE is loaded once and shared between steps 1 and 2.
The LLaVA model is loaded once for step 3.

Usage (from MM-Det root):
  python preprocessing/run_preprocessing.py
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
    VQVAE_CKPT, LMM_CKPT,
    COMPRESSIONS, SPLITS,
    REAL_DATASET_TEMPLATES, SF_T2V_ROOT, SF_I2V_ROOT,
    reals_recons_dir, fakes_recons_dir,
    reals_mmfr_path, fakes_mmfr_path,
    load_split_ids, discover_generators,
    is_done, mark_done, all_combinations,
)
from models import VectorQuantizedVAE, MMEncoder

SUPPORTED_EXTS = {".mp4", ".avi", ".mpeg", ".wmv", ".mov", ".flv"}
RECONS_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
SAMPLE_INTERVAL = 200

LMM_CONFIG = {
    "lmm_ckpt":        LMM_CKPT,
    "lmm_base":        None,
    "load_8bit":       False,
    "load_4bit":       False,
    "conv_mode":       "llava_v1",
    "new_tokens":      64,
    "selected_layers": [-1],
    "interval":        SAMPLE_INTERVAL,
}


# ── VQVAE helpers ─────────────────────────────────────────────────────────────

def denormalize(img_t):
    return (img_t * 0.5 + 0.5).clamp(0, 1)


def process_video(video_path, out_dir, vqvae, device):
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
        img_t   = RECONS_TRANSFORM(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _ = vqvae(img_t)
            if recon.shape != img_t.shape:
                recon = F.interpolate(recon, img_t.shape[-2:], mode="nearest")
            recon_np = rearrange(denormalize(recon.squeeze(0)).cpu().numpy(), "c h w -> h w c")
            Image.fromarray(np.uint8(recon_np * 255)).save(
                os.path.join(recon_dir, f"{video_id}_{frame_idx}.jpg")
            )
        frame_idx += 1
    cap.release()


# ── Step 1: real videos ───────────────────────────────────────────────────────

def step1_reals(vqvae, device):
    print("\n══ Step 1: Preprocess real videos ══")
    done_count = skipped = 0
    for split in SPLITS:
        for compression in COMPRESSIONS:
            out_dir = reals_recons_dir(split, compression)
            if is_done(out_dir):
                skipped += 1
                continue
            src = REAL_DATASET_TEMPLATES[split].format(compression=compression)
            if not os.path.isdir(src):
                print(f"  [WARN] Not found: {src}")
                continue
            videos = sorted(f for f in os.listdir(src)
                            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS)
            print(f"  [{split}/{compression}] {len(videos)} videos")
            for v in tqdm(videos, desc=f"reals/{split}/{compression}", leave=False):
                process_video(os.path.join(src, v), out_dir, vqvae, device)
            mark_done(out_dir)
            done_count += 1
    print(f"  Reals done: {done_count}, skipped: {skipped}")


# ── Step 2: fake videos ───────────────────────────────────────────────────────

def get_fake_videos(vtype, generator, split, compression):
    root = SF_T2V_ROOT if vtype == "t2v" else SF_I2V_ROOT
    src  = os.path.join(root, compression, generator)
    if not os.path.isdir(src):
        return []
    split_ids = set(load_split_ids(split))
    matched = []
    for fn in os.listdir(src):
        stem, ext = os.path.splitext(fn)
        if ext.lower() != ".mp4":
            continue
        if vtype == "i2v" and not stem.endswith("_i2v"):
            continue
        for sid in split_ids:
            if stem.startswith(sid + "_"):
                matched.append(os.path.join(src, fn))
                break
    return sorted(matched)


def step2_fakes(vqvae, device):
    print("\n══ Step 2: Preprocess fake videos ══")
    done_count = skipped = 0
    for vtype, generator, split, compression in all_combinations():
        out_dir = fakes_recons_dir(vtype, generator, split, compression)
        if is_done(out_dir):
            skipped += 1
            continue
        videos = get_fake_videos(vtype, generator, split, compression)
        if not videos:
            print(f"  [WARN] No videos: {vtype}/{generator}/{split}/{compression}")
            continue
        label = f"{vtype}/{generator}/{split}/{compression}"
        print(f"  [{label}] {len(videos)} videos")
        for vp in tqdm(videos, desc=label, leave=False):
            process_video(vp, out_dir, vqvae, device)
        mark_done(out_dir)
        done_count += 1
    print(f"  Fakes done: {done_count}, skipped: {skipped}")


# ── Step 3: cache MMFR ───────────────────────────────────────────────────────

def sample_indices(count, interval=SAMPLE_INTERVAL):
    indices, i = [], 1
    while i <= count:
        indices.append(i)
        i += interval
    return indices


def cache_mmfr_dir(frames_dir, out_pth, model):
    orig_dir = os.path.join(frames_dir, "original")
    if not os.path.isdir(orig_dir):
        return
    meta = {}
    for fn in sorted(os.listdir(orig_dir)):
        if fn.endswith(".jpg"):
            prefix = fn.rsplit("_", maxsplit=1)[0]
            meta[prefix] = meta.get(prefix, 0) + 1
    mm_representations = {}
    for prefix, count in tqdm(meta.items(), desc=os.path.basename(frames_dir), leave=False):
        for idx in sample_indices(count):
            img_path = os.path.join(orig_dir, f"{prefix}_{idx}.jpg")
            if not os.path.exists(img_path):
                continue
            visual_feat, textual_feat = model(Image.open(img_path).convert("RGB"))
            mm_representations[f"{prefix}_{idx}.jpg"] = {
                "visual":  visual_feat.squeeze(0).cpu(),
                "textual": {str(l): textual_feat[i].cpu()
                            for i, l in enumerate(model.selected_layers)},
            }
    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    torch.save(mm_representations, out_pth)


def step3_mmfr(lmm):
    print("\n══ Step 3: Cache MMFR ══")
    done_count = skipped = 0

    print("  ── Reals ──")
    for split in SPLITS:
        for compression in COMPRESSIONS:
            out_pth = reals_mmfr_path(split, compression)
            if os.path.exists(out_pth):
                skipped += 1
                continue
            frames_dir = reals_recons_dir(split, compression)
            if not is_done(frames_dir):
                print(f"  [WARN] Not ready: reals/{split}/{compression}")
                continue
            print(f"  [MMFR] reals/{split}/{compression}")
            cache_mmfr_dir(frames_dir, out_pth, lmm)
            done_count += 1

    print("  ── Fakes ──")
    for vtype, generator, split, compression in all_combinations():
        out_pth = fakes_mmfr_path(vtype, generator, split, compression)
        if os.path.exists(out_pth):
            skipped += 1
            continue
        frames_dir = fakes_recons_dir(vtype, generator, split, compression)
        if not is_done(frames_dir):
            continue
        print(f"  [MMFR] fakes/{vtype}/{generator}/{split}/{compression}")
        cache_mmfr_dir(frames_dir, out_pth, lmm)
        done_count += 1

    print(f"  MMFR done: {done_count}, skipped: {skipped}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load VQVAE once for steps 1 & 2
    print("Loading VQVAE…")
    vqvae = VectorQuantizedVAE(3, 256, 512)
    vqvae.load_state_dict(torch.load(VQVAE_CKPT, map_location=device))
    vqvae.to(device).eval()

    step1_reals(vqvae, device)
    step2_fakes(vqvae, device)

    # Free VQVAE before loading LLaVA
    del vqvae
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load LLaVA once for step 3
    print("\nLoading MMEncoder (LLaVA)…  this may take a while.")
    lmm = MMEncoder(LMM_CONFIG)
    lmm.eval()

    step3_mmfr(lmm)

    print("\n✓ Full preprocessing complete.")


if __name__ == "__main__":
    main()
