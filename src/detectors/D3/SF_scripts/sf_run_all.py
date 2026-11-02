"""
SynthForensics evaluation pipeline for D3 detector.

The combinations are selected from the command line instead of hard-coded
generator lists:

  - --real-datasets / --compressions / --modalities / --generators filter the
    (real_ds × compression × modality × generator) combinations
  - generators default to the folders found under
    preprocessing/frames/fake/{modality}/{compression}/
  - --split-dir sets where the SF-*_fake_ids_*.txt split files are

For each combination:
  - Builds real/fake CSV rows (format identical to folder2csv.py)
  - Filters fake videos using the split ID txt file of the real dataset
  - Runs D3 inference (same pipeline as eval.py)
  - Computes: AUC (paper metric), AP, EER, Accuracy, F1, Precision, Recall

Output layout:
    results/{encoder}_{loss}/
    ├── summary.csv           (rebuilt from all runs/*.json)
    └── runs/
        └── {real_ds}_{compression}_{modality}_{generator}.json

Combinations with missing frames are reported but no JSON is written, so they
are evaluated as soon as the frames are extracted. Existing JSONs are reused
unless --overwrite is given.

Prerequisites:
    Run sf_extract_frames.py first.

Usage:
    python SF_scripts/sf_run_all.py --gpu-id 0 --encoder XCLIP-16 --loss l2 \\
        [--real-datasets FF++] [--compressions raw] [--modalities t2v] \\
        [--generators cogvideox]
"""

import os
import sys
import json
import argparse
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

# Add D3 root to path so we can import data/ and models/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from data.datasets import read_video, set_preprocessing
from models import D3_model
from torch.utils.data import Dataset, DataLoader

# ─── Constants ────────────────────────────────────────────────────────────────

REAL_DATASETS = ['DFD', 'FF++']
COMPRESSIONS  = ['canonical', 'crf23', 'crf40', 'raw']
MODALITIES    = ['t2v', 'i2v']

SPLIT_FILES = {
    'FF++': 'SF-FF_fake_ids_140.txt',
    'DFD':  'SF-DFD_fake_ids_363.txt',
}

# ─── Dataset (same logic as datasets.py, accepts DataFrame directly) ──────────

class D3_dataset_from_df(Dataset):
    """
    Same as D3_dataset_AP but accepts pre-built DataFrames instead of CSV paths.
    Avoids writing temp files for each combination.
    """
    def __init__(self, df_real, df_fake):
        super().__init__()
        self.df    = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.trans = set_preprocessing(None, None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label      = int(self.df.loc[index]['label'])  # cast: '0'/'1' str → int
        frame_path = self.df.loc[index]['content_path']
        frames     = read_video(frame_path, trans=self.trans)
        return frames, label


# ─── Utilities ────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_ids(txt_path):
    """Read the fake video IDs from a split file, ignoring comment lines."""
    ids = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(line)
    return ids


def discover_generators(frames_root, modality, compression):
    """Generator folders found under frames/fake/{modality}/{compression}/."""
    gen_root = os.path.join(frames_root, 'fake', modality, compression)
    if not os.path.isdir(gen_root):
        return []
    return sorted(
        d for d in os.listdir(gen_root)
        if os.path.isdir(os.path.join(gen_root, d))
    )


def make_csv_row(frame_dir, is_real):
    """
    Build a single CSV row dict following the same format as folder2csv.py.
    frame_dir must exist and contain at least one jpg.
    """
    jpgs = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.split('.')[0])
    )
    if not jpgs:
        return None

    first_frame   = os.path.join(frame_dir, jpgs[0])
    frame_numbers = [int(j.split('.')[0]) for j in jpgs]

    return {
        'content_path': frame_dir,
        'image_path':   first_frame,
        'type_id':      'Real Video' if is_real else 'AI Video',
        'label':        str(0) if is_real else str(1),
        'frame_len':    len(jpgs),
        'frame_seq':    frame_numbers,
    }


def build_real_df(frames_root, real_ds, compression):
    """Build DataFrame for all real videos of (dataset, compression)."""
    real_dir = os.path.join(frames_root, 'real', real_ds, compression)
    if not os.path.isdir(real_dir):
        return pd.DataFrame()

    rows = []
    for video_stem in os.listdir(real_dir):
        frame_dir = os.path.join(real_dir, video_stem)
        if not os.path.isdir(frame_dir):
            continue
        row = make_csv_row(frame_dir, is_real=True)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def build_fake_df(frames_root, modality, compression, generator, split_ids):
    """
    Build DataFrame for fake videos filtered by split_ids.

    Matching rule (from filename convention):
      - Numeric ID '006'     → stem startswith '006_'
      - DFD-style '01__x'   → stem startswith '01__x_'
    """
    fake_dir = os.path.join(frames_root, 'fake', modality, compression, generator)
    if not os.path.isdir(fake_dir):
        return pd.DataFrame()

    # Build a lookup: id_prefix → frame_dir
    dir_map = {}
    for video_stem in os.listdir(fake_dir):
        frame_dir = os.path.join(fake_dir, video_stem)
        if not os.path.isdir(frame_dir):
            continue
        dir_map[video_stem] = frame_dir

    rows = []
    for vid_id in split_ids:
        prefix = vid_id + '_'
        matches = [s for s in dir_map if s.startswith(prefix)]
        if not matches:
            continue  # ID not present for this generator (e.g. daVinci partial)
        frame_dir = dir_map[matches[0]]
        row = make_csv_row(frame_dir, is_real=False)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    y_true : array, 0 = real, 1 = fake
    y_pred : array, dis_2nd_std from D3 (higher → more real, as in paper)

    Returns a dict with all metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Score for fake detection (invert, so higher → more fake)
    score_fake = -y_pred

    # AP for the real class
    ap_real = float(average_precision_score(1 - y_true, y_pred))

    # AUC and AP for fake detection
    auc    = float(roc_auc_score(y_true, score_fake))
    ap_fake = float(average_precision_score(y_true, score_fake))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, score_fake)
    fnr = 1 - tpr

    # EER: point where FPR ≈ FNR
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    # Optimal accuracy (maximise over all ROC thresholds)
    accs    = [(score_fake >= t).astype(int) for t in thresholds]
    acc_vals = [accuracy_score(y_true, p) for p in accs]
    best_idx = int(np.argmax(acc_vals))
    y_bin    = accs[best_idx]

    acc       = float(accuracy_score(y_true, y_bin))
    f1        = float(f1_score(y_true, y_bin, zero_division=0))
    precision = float(precision_score(y_true, y_bin, zero_division=0))
    recall    = float(recall_score(y_true, y_bin, zero_division=0))
    cm        = confusion_matrix(y_true, y_bin)
    tn, fp, fn, tp = cm.ravel()

    return {
        'ap_real':   ap_real,
        'ap_fake':   ap_fake,
        'auc':       auc,        # paper metric
        'eer':       eer,
        'accuracy':  acc,
        'f1':        f1,
        'precision': precision,
        'recall':    recall,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n_real':    int(np.sum(y_true == 0)),
        'n_fake':    int(np.sum(y_true == 1)),
    }


# ─── Inference (mirrors eval.py) ─────────────────────────────────────────────

def run_inference(model, df_real, df_fake):
    """
    Same inference loop as eval.py:
      DataLoader(batch_size=1, num_workers=1, pin_memory=True)
      model → (_, _, batch_dis_std)
    Returns y_true, y_pred lists.
    """
    if df_real.empty or df_fake.empty:
        return None, None

    dataset = D3_dataset_from_df(df_real, df_fake)
    if len(dataset) == 0:
        return None, None

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_frames, batch_label in loader:
            batch_inputs = batch_frames.cuda()
            _, _, batch_dis_std = model(batch_inputs)
            y_pred.extend(batch_dis_std.cpu().flatten().numpy())
            y_true.extend(batch_label.cpu().flatten().numpy())

    return y_true, y_pred


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run D3 evaluations on the SynthForensics dataset.'
    )
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--gpu-id',   type=str, default='0')
    parser.add_argument('--loss',     type=str, default='l2', choices=['l2', 'cos'])
    parser.add_argument('--encoder',  type=str, default='XCLIP-16',
                        choices=['CLIP-16', 'CLIP-32', 'XCLIP-16', 'XCLIP-32',
                                 'DINO-base', 'DINO-large', 'ResNet-18',
                                 'VGG-16', 'EfficientNet-b4', 'MobileNet-v3'])
    parser.add_argument('--base-dir', type=str, default=BASE_DIR,
                        help='D3 root: frames are read from preprocessing/frames, '
                             'results are written to results/ (default: D3 root)')
    parser.add_argument('--split-dir', type=str, default=None,
                        help='Folder with the SF-*_fake_ids_*.txt split files (default: --base-dir)')
    parser.add_argument('--real-datasets', nargs='+', default=REAL_DATASETS, choices=REAL_DATASETS)
    parser.add_argument('--compressions',  nargs='+', default=COMPRESSIONS,  choices=COMPRESSIONS)
    parser.add_argument('--modalities',    nargs='+', default=MODALITIES,    choices=MODALITIES)
    parser.add_argument('--generators',    nargs='+', default=None,
                        help='Generator folders to evaluate (default: all folders found in frames/fake/)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute combinations whose JSON result already exists')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    seed_everything(args.seed)

    frames_root = os.path.join(args.base_dir, 'preprocessing', 'frames')
    split_dir   = args.split_dir or args.base_dir

    run_label   = f'{args.encoder}_{args.loss}'
    results_dir = os.path.join(args.base_dir, 'results', run_label, 'runs')
    os.makedirs(results_dir, exist_ok=True)

    # ── Load split ID files ───────────────────────────────────────────────────
    split_ids = {}
    for ds in args.real_datasets:
        fpath = os.path.join(split_dir, SPLIT_FILES[ds])
        split_ids[ds] = load_split_ids(fpath)
        print(f'  Split {ds}: {len(split_ids[ds])} IDs')

    # ── Build combination list ─────────────────────────────────────────────────
    combinations = []
    for real_ds in args.real_datasets:
        for comp in args.compressions:
            for modality in args.modalities:
                generators = args.generators or discover_generators(frames_root, modality, comp)
                for gen in generators:
                    combinations.append((real_ds, comp, modality, gen))

    print(f'\nTotal combinations: {len(combinations)}\n')
    if not combinations:
        sys.exit('[ERROR] No combinations selected: check the filters and preprocessing/frames/')

    # ── Load model once ───────────────────────────────────────────────────────
    print(f'Loading model: encoder={args.encoder}, loss={args.loss}')
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).cuda()
    model.eval()

    # ── Run evaluations ───────────────────────────────────────────────────────
    n_ok, n_reused, n_missing, n_error = 0, 0, 0, 0

    for real_ds, comp, modality, generator in tqdm(combinations, desc='Evaluating'):
        run_id   = f'{real_ds}_{comp}_{modality}_{generator}'
        run_file = os.path.join(results_dir, f'{run_id}.json')

        # Skip if already computed
        if os.path.exists(run_file) and not args.overwrite:
            n_reused += 1
            continue

        df_real = build_real_df(frames_root, real_ds, comp)
        df_fake = build_fake_df(frames_root, modality, comp, generator, split_ids[real_ds])

        # Missing frames: report only, no JSON, so the run is not skipped later
        if df_real.empty or df_fake.empty:
            n_missing += 1
            tqdm.write(f'[skip] {run_id}: missing frames '
                       f'(real={len(df_real)}, fake={len(df_fake)})')
            continue

        try:
            y_true, y_pred = run_inference(model, df_real, df_fake)

            if y_true is None:
                raise ValueError('Empty dataset after frame filtering')

            metrics = compute_metrics(y_true, y_pred)
            row = {
                'run_id':      run_id,
                'real_ds':     real_ds,
                'compression': comp,
                'modality':    modality,
                'generator':   generator,
                'status':      'ok',
                **metrics,
            }
            n_ok += 1

        except Exception as e:
            row = {
                'run_id': run_id, 'real_ds': real_ds, 'compression': comp,
                'modality': modality, 'generator': generator,
                'status': f'error: {e}',
            }
            n_error += 1

        with open(run_file, 'w') as f:
            json.dump(row, f, indent=2)

    # ── Save summary (all runs, not only the ones selected now) ───────────────
    summary_rows = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json'):
            with open(os.path.join(results_dir, fname)) as f:
                summary_rows.append(json.load(f))
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.base_dir, 'results', run_label, 'summary.csv')
    df_summary.to_csv(summary_path, index=False)

    print(f'\n{"="*60}')
    print(f'This run: {n_ok} evaluated, {n_reused} reused, '
          f'{n_missing} missing frames, {n_error} errors')
    print(f'Results saved to results/{run_label}/')
    print(f'Summary: {summary_path} ({len(df_summary)} runs)')
    selected = {f'{r}_{c}_{m}_{g}' for r, c, m, g in combinations}
    if not df_summary.empty:
        sel = df_summary[df_summary['run_id'].isin(selected) & (df_summary['status'] == 'ok')]
        for _, r in sel.iterrows():
            print(f'  {r["run_id"]:40s} AUC={r["auc"]:.4f}  AP_real={r["ap_real"]:.4f}  '
                  f'n_real={r["n_real"]}  n_fake={r["n_fake"]}')
    print('='*60)
