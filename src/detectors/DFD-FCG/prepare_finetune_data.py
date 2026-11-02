"""
Creates datasets/sfff_finetune/ with symlinks and CSVs for DFD-FCG finetuning.
Generators: all (T2V + I2V)

Output structure:
  datasets/sfff_finetune/
    REAL/videos/          (symlinks to real_ffpp/cropped/videos/<video>.avi)
    FAKE/videos/          (symlinks to fake_<t2v|i2v>_<GENERATOR>/cropped/videos/<video>.avi)
    csv_files/
      {train,val,test}_{real,fake}.csv

Sources: face-cropped videos produced by the DFD-FCG preprocessing (see README) in
  datasets/real_ffpp/  and  datasets/fake_<t2v|i2v>_<GENERATOR>/

Splits (FF++ IDs only): splits/{train,val,test}.json of the SynthForensics repository.

Usage (from the DFD-FCG folder):
    python prepare_finetune_data.py --splits_dir /path/to/SynthForensics/splits
"""

import argparse
import json
import os
from pathlib import Path

NAME = 'sfff_finetune'
T2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'self-forcing', 'skyreels-v2', 'wan2-1']
I2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'skyreels-v2', 'wan2-1']


def cropped_videos(root):
    """Return {video ID: cropped .avi} for root/cropped/videos (ID = text before the first '_')."""
    video_dir = root / 'cropped' / 'videos'
    if not video_dir.is_dir():
        print(f"  [WARN] Not found: {video_dir}")
        return {}
    return {Path(f).stem.split('_')[0]: video_dir / f for f in os.listdir(video_dir) if f.endswith('.avi')}


def link(src, dst_dir):
    dst = dst_dir / src.name
    if not dst.is_symlink():
        dst.symlink_to(src)
    return src.name


def build_split(out, split, ids, fake_sources, real_videos):
    print(f"\n=== {split} ({len(ids)} IDs) ===")
    real_dir, fake_dir, csv_dir = out / 'REAL' / 'videos', out / 'FAKE' / 'videos', out / 'csv_files'
    for d in (real_dir, fake_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    real = [link(real_videos[i], real_dir) for i in ids if i in real_videos]
    fake = [link(videos[i], fake_dir) for videos in fake_sources for i in ids if i in videos]

    with open(csv_dir / f"{split}_real.csv", 'w') as f:
        f.writelines(f"{name} 0\n" for name in real)
    with open(csv_dir / f"{split}_fake.csv", 'w') as f:
        f.writelines(f"{name} 1\n" for name in fake)

    print(f"  real: {len(real)} (missing: {len(ids) - len(real)})")
    print(f"  fake: {len(fake)} (missing: {len(ids) * len(fake_sources) - len(fake)})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--splits_dir', required=True, type=Path,
                   help='Folder with train/val/test.json (splits/ of the SynthForensics repository)')
    p.add_argument('--datasets_dir', type=Path, default=Path('datasets'),
                   help='Folder with the preprocessed real_ffpp/ and fake_*/ roots (default: datasets)')
    args = p.parse_args()

    datasets = args.datasets_dir.resolve()
    fake_sources = [cropped_videos(datasets / f"fake_t2v_{g}") for g in T2V_GENERATORS]
    fake_sources += [cropped_videos(datasets / f"fake_i2v_{g}") for g in I2V_GENERATORS]
    real_videos = cropped_videos(datasets / 'real_ffpp')

    for split in ('train', 'val', 'test'):
        with open(args.splits_dir / f"{split}.json") as f:
            ids = [i for i in json.load(f) if '__' not in i]  # FF++ IDs only
        build_split(datasets / NAME, split, ids, fake_sources, real_videos)

    print(f"\nDone. Dataset in: {datasets / NAME}")


if __name__ == '__main__':
    main()
