"""
Build the DeepfakeBench input layout from the published SynthForensics dataset
using symbolic links (no copies).

Output layout (default: <DeepfakeBench>/DFB_Data):

    <SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR>/<video>.mp4 -> <dataset_root>/<T2V|I2V>/videos/<comp>/<gen>/<video>.mp4

<SPLIT> is SF-FF or SF-DFD; <COMP> is empty for raw, otherwise
canonical, crf23 or crf40. Linked videos:
    SF-FF  = FF++ test IDs from test.json (numeric IDs, e.g. 000)
    SF-DFD = DFD test IDs from test.json (e.g. 01__kitchen_pan)

With --train_data, the raw videos of all FF++ IDs (train, val and test) used
for finetuning / training from scratch are also linked:

    SF-FF_full[_i2v]/<GENERATOR>/<video>.mp4

Real videos are not part of SynthForensics: see create_dfb_real_data.py.

Usage (from the DeepfakeBench root):
    python preprocessing/create_dfb_data.py --dataset_root /path/to/SynthForensics [--output_dir /path/to/DFB_Data] [--train_data]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

COMPRESSIONS = ['raw', 'canonical', 'crf23', 'crf40']
MODALITIES = ['T2V', 'I2V']

# Generator token used in published filenames: <ID>_<token>_<t2v|i2v>.mp4
GEN_FILE_TOKEN = {'ltx2-3': 'ltx_2-3'}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'DFB_Data'


def video_id(rel_path):
    """T2V/videos/raw/ltx2-3/01__kitchen_pan_ltx_2-3_t2v.mp4 -> 01__kitchen_pan"""
    parts = Path(rel_path).parts
    gen, name = parts[-2], parts[-1]
    suffix = f"_{GEN_FILE_TOKEN.get(gen, gen)}_{parts[0].lower()}.mp4"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected filename: {rel_path}")
    return name[:-len(suffix)]


def out_dir_name(split, modality, comp):
    if split == 'SF-FF_full':
        return 'SF-FF_full_i2v' if modality == 'I2V' else 'SF-FF_full'
    name = f"{split}_fake_test_set"
    if modality == 'I2V':
        name += '_i2v'
    if comp != 'raw':
        name += f"_{comp}"
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_root', required=True, type=Path,
                   help='Root of the published SynthForensics dataset (contains T2V/, I2V/, test.json)')
    p.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                   help='Output folder (default: <DeepfakeBench>/DFB_Data)')
    p.add_argument('--train_data', action='store_true',
                   help='Also link the raw videos of all FF++ IDs used for training (SF-FF_full)')
    p.add_argument('--dry_run', action='store_true', help='Print counts without creating anything')
    args = p.parse_args()

    root = args.dataset_root.resolve()
    out_root = args.output_dir.resolve()

    # SF-FF / SF-DFD come from test.json; SF-FF_full from all three split files
    jobs = []
    for split_file in ['train.json', 'val.json', 'test.json']:
        with open(root / split_file) as f:
            for rel in json.load(f):
                vid = video_id(rel)
                splits = []
                if split_file == 'test.json':
                    splits.append('SF-DFD' if '__' in vid else 'SF-FF')
                if args.train_data and '__' not in vid and Path(rel).parts[2] == 'raw':
                    splits.append('SF-FF_full')
                if splits:
                    jobs.append((rel, splits))

    counts = defaultdict(int)
    missing = []
    created = skipped = 0
    for rel, splits in jobs:
        modality, _, comp, gen, _ = Path(rel).parts
        if modality not in MODALITIES or comp not in COMPRESSIONS:
            raise ValueError(f"Unexpected path in split file: {rel}")
        src = root / rel
        if not src.is_file():
            missing.append(rel)
            continue
        for split in splits:
            dst_dir = out_root / out_dir_name(split, modality, comp) / gen
            dst = dst_dir / src.name
            counts[(dst_dir.parent.name, dst_dir.name)] += 1
            if args.dry_run:
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                if dst.is_symlink() and Path(os.readlink(dst)) == src:
                    skipped += 1
                    continue
                raise FileExistsError(f"{dst} exists and does not point to {src}")
            dst.symlink_to(src)
            created += 1

    for (folder, gen), n in sorted(counts.items()):
        print(f"{folder}/{gen}: {n}")
    print(f"\nOutput: {out_root}")
    print(f"Created: {created}  Already present: {skipped}  Missing sources: {len(missing)}")
    for rel in missing[:20]:
        print(f"  MISSING {rel}")


if __name__ == '__main__':
    main()
