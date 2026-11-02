"""
Generate the training dataset JSON for finetuning or training from scratch.

    sfff_finetune: fake videos of all generators (T2V + I2V) + real FF++ videos
    sfff_scratch:  fake videos of the in-domain generators (CogVideoX, Wan2.1,
                   LTX-2.3, MAGI-1; T2V + I2V) + real FF++ videos

Videos are assigned to train/val/test with the FF++ split in splits/*.json.

Expected face crops (custom_preprocess.py outputs, relative to --proc_dir):
    fake/SF-FF_full/<t2v|i2v>/<GENERATOR>/frames/<video>/*.png
    real/FF++_full/frames/<video>/*.png

Usage (from the DeepfakeBench root):
    python preprocessing/create_train_json.py --name sfff_finetune
    python preprocessing/create_train_json.py --name sfff_scratch
"""

import argparse
import glob
import json
import os
from pathlib import Path

DFB_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS_DIR = DFB_ROOT.parents[2] / 'splits'

FAKE_LABEL = 'SynthForensics_fake'
REAL_LABEL = 'SynthForensics_real'

T2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'self-forcing', 'skyreels-v2', 'wan2-1']
I2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'skyreels-v2', 'wan2-1']
IN_DOMAIN = ['cogvideox', 'ltx2-3', 'magi-1', 'wan2-1']

# DeepfakeBench selects the best checkpoint on the JSON 'test' phase:
# the official validation IDs go there, the official test IDs go to 'val'.
PHASE_SPLIT = {'train': 'train', 'val': 'test', 'test': 'val'}


def collect(frames_root, label, id_to_phase):
    """Return {phase: {video: entry}} for the videos in frames_root (skipping videos without frames)."""
    entries = {phase: {} for phase in PHASE_SPLIT}
    if not os.path.isdir(frames_root):
        print(f"  [WARN] Not found: {frames_root}")
        return entries
    skipped = 0
    for video_dir in sorted(os.scandir(frames_root), key=lambda d: d.name):
        if not video_dir.is_dir():
            continue
        phase = id_to_phase.get(video_dir.name.split('_')[0])
        if phase is None:
            continue
        frames = sorted(glob.glob(os.path.join(video_dir.path, '*.png')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not frames:
            skipped += 1
            continue
        entries[phase][video_dir.name] = {'label': label, 'frames': frames}
    counts = ', '.join(f"{phase}={len(v)}" for phase, v in entries.items())
    print(f"  {frames_root}: {counts} (no frames: {skipped})")
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True, choices=['sfff_finetune', 'sfff_scratch'])
    p.add_argument('--proc_dir', type=Path, default=DFB_ROOT / 'preprocessing',
                   help='Folder with the fake/ and real/ face crops (default: preprocessing)')
    p.add_argument('--splits_dir', type=Path, default=DEFAULT_SPLITS_DIR,
                   help='Folder with train/val/test.json (default: splits/ of this repository)')
    p.add_argument('--output_dir', type=Path, default=DFB_ROOT / 'preprocessing' / 'dataset_json')
    args = p.parse_args()

    id_to_phase = {}
    for phase, split in PHASE_SPLIT.items():
        with open(args.splits_dir / f"{split}.json") as f:
            id_to_phase.update({i: phase for i in json.load(f) if '__' not in i})  # FF++ IDs only

    sources = [('t2v', g) for g in T2V_GENERATORS] + [('i2v', g) for g in I2V_GENERATORS]
    if args.name == 'sfff_scratch':
        sources = [(m, g) for m, g in sources if g in IN_DOMAIN]

    fake = {phase: {} for phase in PHASE_SPLIT}
    for modality, gen in sources:
        frames_root = args.proc_dir / 'fake' / 'SF-FF_full' / modality / gen / 'frames'
        for phase, entries in collect(frames_root, FAKE_LABEL, id_to_phase).items():
            fake[phase].update(entries)
    real = collect(args.proc_dir / 'real' / 'FF++_full' / 'frames', REAL_LABEL, id_to_phase)

    print('\nTotal fake: ' + ', '.join(f"{phase}={len(v)}" for phase, v in fake.items()))
    print('Total real: ' + ', '.join(f"{phase}={len(v)}" for phase, v in real.items()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.name}.json"
    with open(out, 'w') as f:
        json.dump({args.name: {FAKE_LABEL: fake, REAL_LABEL: real}}, f)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
