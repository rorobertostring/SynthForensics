"""
Step 3 — Cache Multi-Modal Forgery Representations (MMFR).

Loads the fine-tuned LLaVA model ONCE and processes all frames
(real and fake) that have been preprocessed in steps 1 and 2.
One mm_representation.pth is written per (split/compression/label).

Output layout:
  preprocessing/mmfr/reals/{split}/{compression}/0_real/mm_representation.pth
  preprocessing/mmfr/fakes/{vtype}/{generator}/{split}/{compression}/1_fake/mm_representation.pth

Skips any entry whose .pth already exists.

Usage (from MM-Det root):
  python preprocessing/03_cache_mmfr.py
"""
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.pipeline_config import (
    LMM_CKPT, COMPRESSIONS, SPLITS,
    reals_recons_dir, fakes_recons_dir,
    reals_mmfr_path, fakes_mmfr_path,
    is_done, all_combinations,
)
from models import MMEncoder

SAMPLE_INTERVAL = 200  # cache one frame every N frames per video

LMM_CONFIG = {
    "lmm_ckpt":       LMM_CKPT,
    "lmm_base":       None,
    "load_8bit":      False,
    "load_4bit":      False,
    "conv_mode":      "llava_v1",
    "new_tokens":     64,
    "selected_layers": [-1],
    "interval":       SAMPLE_INTERVAL,
}


def sample_indices(frame_count, interval=SAMPLE_INTERVAL):
    indices, i = [], 1
    while i <= frame_count:
        indices.append(i)
        i += interval
    return indices


def cache_for_dir(frames_dir, out_pth, model):
    """
    Process all frames in frames_dir/original/ and save MMFR to out_pth.
    frames_dir is expected to contain original/ and recons/ sub-directories.
    """
    orig_dir = os.path.join(frames_dir, "original")
    if not os.path.isdir(orig_dir):
        print(f"    [WARN] original/ missing: {orig_dir}")
        return

    # Build {video_prefix: frame_count} map
    meta = {}
    for fn in sorted(os.listdir(orig_dir)):
        if not fn.endswith(".jpg"):
            continue
        prefix = fn.rsplit("_", maxsplit=1)[0]
        meta[prefix] = meta.get(prefix, 0) + 1

    mm_representations = {}
    for prefix, count in tqdm(meta.items(), desc=os.path.basename(frames_dir), leave=False):
        for idx in sample_indices(count):
            img_path = os.path.join(orig_dir, f"{prefix}_{idx}.jpg")
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            visual_feat, textual_feat = model(img)
            mm_layer_features = {
                str(layer): textual_feat[i].cpu()
                for i, layer in enumerate(model.selected_layers)
            }
            mm_representations[f"{prefix}_{idx}.jpg"] = {
                "visual":   visual_feat.squeeze(0).cpu(),
                "textual":  mm_layer_features,
            }

    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    torch.save(mm_representations, out_pth)


def main():
    print("Loading MMEncoder (LLaVA)…  this may take a while.")
    model = MMEncoder(LMM_CONFIG)
    model.eval()

    done_count = skipped = 0

    # ── Reals ────────────────────────────────────────────────────────────────
    print("\n── Caching MMFR for real videos ──")
    for split in SPLITS:
        for compression in COMPRESSIONS:
            out_pth = reals_mmfr_path(split, compression)
            if os.path.exists(out_pth):
                print(f"  [SKIP] reals/{split}/{compression}")
                skipped += 1
                continue
            frames_dir = reals_recons_dir(split, compression)
            if not is_done(frames_dir):
                print(f"  [WARN] Not preprocessed yet: reals/{split}/{compression}")
                continue
            print(f"  [MMFR] reals/{split}/{compression}")
            cache_for_dir(frames_dir, out_pth, model)
            done_count += 1

    # ── Fakes ────────────────────────────────────────────────────────────────
    print("\n── Caching MMFR for fake videos ──")
    for vtype, generator, split, compression in all_combinations():
        out_pth = fakes_mmfr_path(vtype, generator, split, compression)
        if os.path.exists(out_pth):
            skipped += 1
            continue
        frames_dir = fakes_recons_dir(vtype, generator, split, compression)
        if not is_done(frames_dir):
            print(f"  [WARN] Not preprocessed: {vtype}/{generator}/{split}/{compression}")
            continue
        print(f"  [MMFR] fakes/{vtype}/{generator}/{split}/{compression}")
        cache_for_dir(frames_dir, out_pth, model)
        done_count += 1

    print(f"\nMMFR caching complete. Done: {done_count}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
