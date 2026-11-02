"""
Run the full DeepfakeBench pipeline on SynthForensics:
face-crop extraction -> dataset JSON -> inference, for every
split / modality / compression / generator / detector combination.

Run from the DeepfakeBench root, after preprocessing/create_dfb_data.py and
preprocessing/create_dfb_real_data.py:

    python SF_scripts/sf_run_all.py

Steps already done are skipped (extracted folders, existing result CSVs),
so the script can be stopped and resumed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DFB_ROOT = Path(__file__).resolve().parent.parent

SPLITS = ['ff', 'dfd']
MODALITIES = ['t2v', 'i2v']
COMPRESSIONS = ['raw', 'canonical', 'crf23', 'crf40']
T2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'self-forcing', 'skyreels-v2', 'wan2-1']
I2V_GENERATORS = ['cogvideox', 'daVinci-MagiHuman', 'helios', 'ltx2-3', 'magi-1', 'skyreels-v2', 'wan2-1']

# detector -> (config YAML, weights file)
DETECTORS = {
    'recce':       ('recce.yaml',       'recce_best.pth'),
    'ucf':         ('ucf.yaml',         'ucf_best.pth'),
    'effort':      ('effort.yaml',      'effort_clip_L14_trainOn_FaceForensic.pth'),
    'altfreezing': ('altfreezing.yaml', 'altfreezing_ckpt.pth'),
    'ftcn':        ('ftcn.yaml',        'ftcn_tt.pth'),
    'prodet':      ('prodet.yaml',      'ProDet_best.pth'),
}

# SynthForensics split -> real dataset folder under --real_root
REAL_DATASET = {'ff': 'FF++', 'dfd': 'DFD'}

DONE_MARKER = '.done'


def fake_video_dir(data_dir, split, modality, comp, gen):
    name = f"SF-{split.upper()}_fake_test_set"
    if modality == 'i2v':
        name += '_i2v'
    if comp != 'raw':
        name += f"_{comp}"
    return data_dir / name / gen


def dataset_name(split, modality, comp, gen):
    mod = '_i2v' if modality == 'i2v' else ''
    return f"sf{split}{mod}_{comp}_{gen}"


def results_dir(split, modality, det):
    mod = '-i2v' if modality == 'i2v' else ''
    return DFB_ROOT / 'results' / f"sf-{split}{mod}" / det


def run(cmd, cwd):
    print('  $ ' + ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode == 0


def preprocess(video_dir, out_dir, workers):
    """Extract face crops once per folder; a marker file flags completed folders."""
    if (out_dir / DONE_MARKER).exists():
        return True
    cmd = [sys.executable, 'custom_preprocess.py',
           '--video_dir', video_dir, '--output_dir', out_dir,
           '--num_frames', 32, '--mode', 'fixed_num_frames']
    if workers:
        cmd += ['--workers', workers]
    if not run(cmd, cwd=DFB_ROOT / 'preprocessing'):
        return False
    (out_dir / DONE_MARKER).touch()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--real_root', type=Path, default=DFB_ROOT / 'DFB_Data' / 'real',
                   help='Real test set folders built by create_dfb_real_data.py (default: DFB_Data/real)')
    p.add_argument('--data_dir', type=Path, default=DFB_ROOT / 'DFB_Data',
                   help='Fake test set folders built by create_dfb_data.py (default: DFB_Data)')
    p.add_argument('--proc_dir', type=Path, default=DFB_ROOT / 'preprocessing',
                   help='Where extracted face crops are saved, in fake/ and real/ (default: preprocessing)')
    p.add_argument('--weights_dir', type=Path, default=DFB_ROOT / 'weights',
                   help='Folder with the detector checkpoints (default: weights)')
    p.add_argument('--gpu_id', default='0', help='CUDA device index')
    p.add_argument('--workers', type=int, default=None, help='Workers for face-crop extraction')
    p.add_argument('--splits', nargs='+', default=SPLITS, choices=SPLITS)
    p.add_argument('--modalities', nargs='+', default=MODALITIES, choices=MODALITIES)
    p.add_argument('--compressions', nargs='+', default=COMPRESSIONS, choices=COMPRESSIONS)
    p.add_argument('--generators', nargs='+', default=None,
                   help='Subset of generators (default: all generators of each modality)')
    p.add_argument('--detectors', nargs='+', default=list(DETECTORS), choices=list(DETECTORS))
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    data_dir, proc_dir = args.data_dir.resolve(), args.proc_dir.resolve()
    json_dir = DFB_ROOT / 'preprocessing' / 'dataset_json'
    failed = []

    combos = []
    for split in args.splits:
        for modality in args.modalities:
            gens = T2V_GENERATORS if modality == 't2v' else I2V_GENERATORS
            if args.generators:
                gens = [g for g in gens if g in args.generators]
            for comp in args.compressions:
                for gen in gens:
                    combos.append((split, modality, comp, gen))
    print(f"Combinations: {len(combos)}  Detectors: {', '.join(args.detectors)}\n")

    # Step 1 — face-crop extraction (real, then fake)
    print('=== Step 1: face-crop extraction ===')
    for split, comp in sorted({(s, c) for s, _, c, _ in combos}):
        real = REAL_DATASET[split]
        video_dir = args.real_root / real / f"real_{comp}_test_set"
        if not video_dir.is_dir():
            print(f"  [MISSING] {video_dir}")
            failed.append(f"real {real}/{comp}")
            continue
        print(f"-- real {real} / {comp}")
        if not preprocess(video_dir, proc_dir / 'real' / real / comp, args.workers):
            failed.append(f"real {real}/{comp}")

    for split, modality, comp, gen in combos:
        video_dir = fake_video_dir(data_dir, split, modality, comp, gen)
        if not video_dir.is_dir():
            print(f"  [MISSING] {video_dir}")
            failed.append(dataset_name(split, modality, comp, gen))
            continue
        print(f"-- fake {dataset_name(split, modality, comp, gen)}")
        out_dir = proc_dir / 'fake' / f"SF-{split.upper()}" / modality / comp / gen
        if not preprocess(video_dir, out_dir, args.workers):
            failed.append(dataset_name(split, modality, comp, gen))

    # Step 2 — dataset JSON (fake + matching real set)
    print('\n=== Step 2: dataset JSON ===')
    ready = []
    for split, modality, comp, gen in combos:
        name = dataset_name(split, modality, comp, gen)
        fake_frames = proc_dir / 'fake' / f"SF-{split.upper()}" / modality / comp / gen / 'frames'
        real_frames = proc_dir / 'real' / REAL_DATASET[split] / comp / 'frames'
        if not fake_frames.is_dir() or not real_frames.is_dir():
            continue
        if run([sys.executable, 'custom_rearrange.py', '--dataset_name', name,
                '--fake_frames_dir', fake_frames, '--real_frames_dir', real_frames,
                '--output_dir', json_dir], cwd=DFB_ROOT / 'preprocessing'):
            ready.append((split, modality, comp, gen))
        else:
            failed.append(name)

    # Step 3 — inference, one call per (split, modality, detector)
    print('\n=== Step 3: inference ===')
    for split in args.splits:
        for modality in args.modalities:
            for det in args.detectors:
                out_dir = results_dir(split, modality, det)
                todo = [dataset_name(*c) for c in ready
                        if c[0] == split and c[1] == modality
                        and not (out_dir / f"{dataset_name(*c)}.csv").exists()]
                if not todo:
                    continue
                print(f"-- {det} on sf-{split} {modality}: {len(todo)} datasets")
                yaml_file, weights = DETECTORS[det]
                if not run([sys.executable, 'training/test.py',
                            '--detector_path', f"training/config/detector/{yaml_file}",
                            '--test_dataset', *todo,
                            '--weights_path', args.weights_dir / weights,
                            '--output_dir', out_dir], cwd=DFB_ROOT):
                    failed.append(f"{det} sf-{split} {modality}")

    print('\nDone.' if not failed else f"\nDone with {len(failed)} failures:")
    for f in failed:
        print(f"  {f}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
