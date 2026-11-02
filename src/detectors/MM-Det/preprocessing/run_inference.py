"""
Step 4 — Run inference and compute metrics for all combinations.

Loads MM-Det ONCE, then iterates over every (vtype, generator, split, compression).
For each combination it:
  1. Builds a combined dataset (real + fake) without copying data.
  2. Looks up cached MMFR representations.
  3. Runs the model in sliding-window mode (window_size=10).
  4. Computes AUROC, AP, EER, ACC and saves to a JSON file.

Results layout:
  preprocessing/results/{vtype}/{generator}/{split}/{compression}.json

Combinations whose JSON already exists are skipped.

Usage (from MM-Det root):
  python preprocessing/run_inference.py
"""
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.pipeline_config import (
    MMDET_CKPT, LMM_CKPT, COMPRESSIONS, SPLITS,
    reals_recons_dir, fakes_recons_dir,
    reals_mmfr_path, fakes_mmfr_path,
    results_path, is_done, all_combinations,
)
from models import MMDet
from dataset.vdataset import (
    VideoFolderDatasetForReconsWithFn,
    collate_video_folder_for_recons_dataset_with_fn,
)
from dataset.utils import get_default_transformation_cfg

WINDOW_SIZE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "model_name":      "MMDet",
    "lmm_ckpt":        LMM_CKPT,
    "lmm_base":        None,
    "st_pretrained":   False,
    "st_ckpt":         None,
    "window_size":     WINDOW_SIZE,
    "conv_mode":       "llava_v1",
    "new_tokens":      64,
    "selected_layers": [-1],
    "interval":        200,
    "load_8bit":       False,
    "load_4bit":       False,
    "cache_mm":        True,
}


# ── MMFR helpers ──────────────────────────────────────────────────────────────

def load_mmfr(real_pth, fake_pth):
    """Load and index MMFR .pth files for reals and fakes."""
    mmfr = {}
    for label, pth in [("0_real", real_pth), ("1_fake", fake_pth)]:
        if not os.path.exists(pth):
            continue
        data = torch.load(pth, map_location="cpu")
        index = {}
        for frame_id in data:
            prefix = os.path.splitext(frame_id)[0].rsplit("_", maxsplit=1)[0]
            idx    = int(os.path.splitext(frame_id)[0].rsplit("_", maxsplit=1)[1])
            index.setdefault(prefix, []).append(idx)
        mmfr[label] = {"data": data, "index": index}
    return mmfr


def nearest_cached(indices, target):
    """Return the largest cached index that is <= target."""
    candidates = [i for i in indices if i <= target]
    return max(candidates) if candidates else min(indices)


def get_mmfr_features(fn, frame_offset, mmfr, device):
    """
    Look up visual and textual features for a given fn and frame offset.
    fn format: {data_root}/{cls}/{video_prefix}__{start_frame}
    """
    parts       = fn.rsplit("/", maxsplit=2)
    label       = parts[-2]           # "0_real" or "1_fake"
    video_info  = parts[-1]           # "{prefix}__{start}"
    prefix, start_str = video_info.rsplit("__", maxsplit=1)
    target = int(start_str) + frame_offset

    if label not in mmfr or prefix not in mmfr[label]["index"]:
        return None, None

    nearest = nearest_cached(mmfr[label]["index"][prefix], target)
    key     = f"{prefix}_{nearest}.jpg"
    entry   = mmfr[label]["data"].get(key)
    if entry is None:
        return None, None

    visual   = entry["visual"].to(device)          # [1024]
    textual  = entry["textual"]["-1"].to(device)   # [64, 4096]
    return visual, textual


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_eer(gts, probas):
    fpr, tpr, thresholds = roc_curve(gts, probas)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(gts, probas):
    preds_binary = [1 if p >= 0.5 else 0 for p in probas]
    acc = float(sum(g == p for g, p in zip(gts, preds_binary)) / len(gts))
    return {
        "AUROC": float(roc_auc_score(gts, probas)),
        "AP":    float(average_precision_score(gts, probas)),
        "EER":   compute_eer(gts, probas),
        "ACC":   acc,
        "n_real": int(sum(g == 0 for g in gts)),
        "n_fake": int(sum(g == 1 for g in gts)),
    }


# ── Inference loop ────────────────────────────────────────────────────────────

def run_one(model, vtype, generator, split, compression):
    real_dir = reals_recons_dir(split, compression)
    fake_dir = fakes_recons_dir(vtype, generator, split, compression)

    if not is_done(real_dir):
        print(f"    [WARN] Reals not ready: {split}/{compression}")
        return None
    if not is_done(fake_dir):
        print(f"    [WARN] Fakes not ready: {vtype}/{generator}/{split}/{compression}")
        return None

    # Build dataset: two separate data_roots, concatenated
    transform_cfg = get_default_transformation_cfg(mode="test")
    real_dataset = VideoFolderDatasetForReconsWithFn(
        data_root=os.path.dirname(real_dir),   # .../reals/{split}/{compression}/
        selected_cls_labels=[("0_real", 0)],
        sample_method="entire",
        transform_cfg=transform_cfg,
    )
    fake_dataset = VideoFolderDatasetForReconsWithFn(
        data_root=os.path.dirname(fake_dir),   # .../fakes/{vtype}/{generator}/{split}/{compression}/
        selected_cls_labels=[("1_fake", 1)],
        sample_method="entire",
        transform_cfg=transform_cfg,
    )
    dataset = ConcatDataset([real_dataset, fake_dataset])
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_video_folder_for_recons_dataset_with_fn,
    )

    real_mmfr = reals_mmfr_path(split, compression)
    fake_mmfr = fakes_mmfr_path(vtype, generator, split, compression)
    for pth in (real_mmfr, fake_mmfr):
        if not os.path.exists(pth):
            print(f"    [WARN] MM representations not cached: {pth} (run 03_cache_mmfr.py)")
            return None
    mmfr = load_mmfr(real_mmfr, fake_mmfr)

    gts, probas = [], []
    n_clips = n_missing = 0
    model.eval()
    model.to(DEVICE)

    with torch.no_grad():
        for fns, (original, recons), labels in tqdm(loader, desc=f"{generator}/{split}/{compression}", leave=False):
            original = original.float().to(DEVICE)
            recons   = recons.float().to(DEVICE)
            B, L, C, H, W = original.shape

            video_proba_list = []
            for i in range(0, L - WINDOW_SIZE + 1, WINDOW_SIZE):
                clip = (
                    original[:, i: i + WINDOW_SIZE],
                    recons[:, i: i + WINDOW_SIZE],
                )
                visuals, textuals = [], []
                for fn in fns:
                    vis, txt = get_mmfr_features(fn, i, mmfr, DEVICE)
                    n_clips += 1
                    if vis is None:
                        n_missing += 1
                        vis = torch.zeros(1024, device=DEVICE)
                        txt = torch.zeros(64, 4096, device=DEVICE)
                    visuals.append(vis)
                    textuals.append(txt)

                pred = model(clip, cached_features={
                    "visual":   torch.stack(visuals, dim=0),   # [B, 1024]
                    "textual":  torch.stack(textuals, dim=0),  # [B, 64, 4096]
                })
                proba = F.softmax(pred, dim=-1)[:, 1]
                video_proba_list.append(proba.cpu())

            # Average probability over clips
            mean_proba = torch.stack(video_proba_list).mean(dim=0)
            for idx in range(len(fns)):
                gts.append(int(labels[idx].item()))
                probas.append(float(mean_proba[idx].item()))

    if n_missing:
        print(f"    [WARN] {n_missing}/{n_clips} clips without cached MM features: zero features used")
    return compute_metrics(gts, probas)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model():
    print("Loading MM-Det model…")
    model = MMDet(MODEL_CONFIG)
    state_dict = torch.load(MMDET_CKPT, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # Strip DDP prefix and rename legacy key
    clean = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("temporal_token", "fc_token")
        clean[k] = v
    model.load_state_dict(clean, strict=True)
    model.eval()
    return model


def main():
    model = load_model()
    done_count = skipped = errors = 0

    for vtype, generator, split, compression in all_combinations():
        out_path = results_path(vtype, generator, split, compression)
        if os.path.exists(out_path):
            skipped += 1
            continue

        label = f"{vtype}/{generator}/{split}/{compression}"
        print(f"  [{label}]")
        metrics = run_one(model, vtype, generator, split, compression)

        if metrics is None:
            errors += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "vtype": vtype, "generator": generator,
                "split": split, "compression": compression,
                "metrics": metrics,
            }, f, indent=2)
        done_count += 1
        print(f"    AUROC={metrics['AUROC']:.4f}  AP={metrics['AP']:.4f}  "
              f"EER={metrics['EER']:.4f}  ACC={metrics['ACC']:.4f}")

    print(f"\nInference complete. Done: {done_count}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
