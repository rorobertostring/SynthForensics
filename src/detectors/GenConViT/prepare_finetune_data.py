"""
Creates the train/valid/test structure for GenConViT finetuning.
Generators: all (T2V + I2V)

Output (symlinks, not copies):
  finetune_data/
    train/{fake,real}/
    valid/{fake,real}/
    test/{fake,real}/

Sources (DeepfakeBench face crops, see the DeepfakeBench README):
  - Fake: ../DeepfakeBench/preprocessing/fake/SF-FF_full/<t2v|i2v>/<GENERATOR>/frames/
  - Real: ../DeepfakeBench/preprocessing/real/FF++_full/frames/

Splits (FF++ IDs only): splits/{train,val,test}.json of this repository.

Usage:
    python prepare_finetune_data.py
"""

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DFB_DIR = ROOT.parent / 'DeepfakeBench' / 'preprocessing'
DEFAULT_SPLITS_DIR = ROOT.parents[2] / 'splits'

T2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'self-forcing', 'skyreels-v2', 'wan2-1']
I2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'skyreels-v2', 'wan2-1']

# output folder -> split file
SPLITS = {'train': 'train', 'valid': 'val', 'test': 'test'}


def video_dirs(frames_root):
    """Return {video ID: frame folder} for the videos in frames_root (ID = text before the first '_')."""
    if not frames_root.is_dir():
        print(f"  [WARN] Not found: {frames_root}")
        return {}
    return {d.name.split('_')[0]: d for d in frames_root.iterdir() if d.is_dir()}


def link_frames(src_dir, dst_dir, prefix):
    """Symlink every .png of src_dir into dst_dir as <prefix>_<frame>.png; return the number of frames."""
    frames = sorted(glob.glob(os.path.join(src_dir, '*.png')))
    for f in frames:
        dst = dst_dir / f"{prefix}_{os.path.basename(f)}"
        if not dst.is_symlink():
            dst.symlink_to(os.path.abspath(f))
    return len(frames)


def build_split(out_dir, split, ids, sources, real_videos):
    print(f"\n=== {split} ({len(ids)} IDs) ===")
    fake_dir, real_dir = out_dir / split / 'fake', out_dir / split / 'real'
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    total_fake = missing_fake = 0
    for (modality, gen), videos in sources.items():
        for vid_id in ids:
            n = link_frames(videos[vid_id], fake_dir, f"{modality}_{gen}_{vid_id}") if vid_id in videos else 0
            total_fake += n
            missing_fake += n == 0

    total_real = missing_real = 0
    for vid_id in ids:
        n = link_frames(real_videos[vid_id], real_dir, f"real_{vid_id}") if vid_id in real_videos else 0
        total_real += n
        missing_real += n == 0

    print(f"  fake frames: {total_fake}  (missing videos: {missing_fake})")
    print(f"  real frames: {total_real}  (missing videos: {missing_real})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dfb_dir', type=Path, default=DEFAULT_DFB_DIR,
                   help='DeepfakeBench preprocessing folder with fake/ and real/ (default: ../DeepfakeBench/preprocessing)')
    p.add_argument('--splits_dir', type=Path, default=DEFAULT_SPLITS_DIR,
                   help='Folder with train/val/test.json (default: splits/ of this repository)')
    p.add_argument('--output_dir', type=Path, default=ROOT / 'finetune_data')
    args = p.parse_args()

    fake_root = args.dfb_dir / 'fake' / 'SF-FF_full'
    sources = {('t2v', g): video_dirs(fake_root / 't2v' / g / 'frames') for g in T2V_GENERATORS}
    sources.update({('i2v', g): video_dirs(fake_root / 'i2v' / g / 'frames') for g in I2V_GENERATORS})
    real_videos = video_dirs(args.dfb_dir / 'real' / 'FF++_full' / 'frames')

    for split, split_file in SPLITS.items():
        with open(args.splits_dir / f"{split_file}.json") as f:
            ids = [i for i in json.load(f) if '__' not in i]  # FF++ IDs only
        build_split(args.output_dir, split, ids, sources, real_videos)

    print(f"\nDone. Structure at: {args.output_dir}")


if __name__ == '__main__':
    main()
