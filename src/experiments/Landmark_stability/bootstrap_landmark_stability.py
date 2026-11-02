#!/usr/bin/env python3
"""Bootstrap 95% CI on per-benchmark landmark statistics (numpy-vectorized).

For each entity in the comparative pool, computes the mean of:
  - a-bar         (= mean_confidence column)
  - Comp_tau      (= completeness_th_0.X column for tau in 0.3..0.7)
on the face-positive subset (faces_detected > 0), and reports a 95%
percentile bootstrap CI with B=1000 resamples (sampling videos with
replacement). Uses chunked numpy bootstrap to keep memory bounded.
"""
import csv
from pathlib import Path
import numpy as np

ROOT = Path('/path/to/results_32')
SEED = 42
B = 1000
CHUNK = 100  # bootstrap resamples per batch

BENCHMARKS = {
    'GenVideo':    'genvideo_wo_duplicates.csv',
    'AIGVDBench':  'aigvdbench_wo_duplicates.csv',
    'GenVidBench': 'genvidbench.csv',
    'GVD':         'gvd_wo_duplicates.csv',
    'DVF':         'dvf_wo_duplicates.csv',
    'DeepAction':  'deepaction.csv',
    'GVF':         'decof_wo_duplicates.csv',
    'LOKI':        'loki_wo_duplicates.csv',
    'AEGIS':       'aegis.csv',
}

REALS = {'FF++ (real)': 'ff++_real.csv', 'DFD (real)': 'dfd_real.csv'}

SF_FILES = [
    'sf_cogvideox_t2v_raw.csv', 'sf_cogvideox_i2v_raw.csv',
    'sf_magi-1_t2v_raw.csv',    'sf_magi-1_i2v_raw.csv',
    'sf_wan2-1_t2v_raw.csv',    'sf-wan21-i2v.csv',
    'sf_skyreels-v2_t2v_raw.csv', 'sf_skyreels_i2v_raw.csv',
    'sf_self-forcing_t2v_raw.csv',
    'sf_ltx2-3_t2v_raw.csv',    'sf_ltx_i2v_raw.csv',
    'sf-helios-t2v.csv',        'sf-helios-i2v.csv',
    'sf-davinci-t2v.csv',       'sf-davinci-i2v.csv',
]

METRICS = [
    ('a-bar',     'mean_confidence'),
    ('Comp_0.3',  'completeness_th_0.3'),
    ('Comp_0.4',  'completeness_th_0.4'),
    ('Comp_0.5',  'completeness_th_0.5'),
    ('Comp_0.6',  'completeness_th_0.6'),
    ('Comp_0.7',  'completeness_th_0.7'),
]


def load_face_positive(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fd = int(row['faces_detected'])
            except (KeyError, ValueError):
                continue
            if fd > 0:
                rows.append(row)
    return rows


def cols_to_array(rows):
    """Return (n, K) float array stacking all metric columns in METRICS order."""
    out = np.empty((len(rows), len(METRICS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, (_, col) in enumerate(METRICS):
            try:
                out[i, j] = float(r[col])
            except (KeyError, ValueError):
                out[i, j] = np.nan
    return out


def bootstrap_means_ci(arr, B=1000, chunk=CHUNK, alpha=0.05, seed=SEED):
    """Vectorized chunked bootstrap. arr is (n, K). Returns (means, lo, hi) each (K,)."""
    rng = np.random.default_rng(seed)
    n, K = arr.shape
    if n == 0:
        return None, None, None
    boot_means = np.empty((B, K), dtype=np.float64)
    written = 0
    while written < B:
        b = min(chunk, B - written)
        idx = rng.integers(0, n, size=(b, n))
        # arr[idx] is (b, n, K); mean over axis=1 -> (b, K)
        boot_means[written:written + b] = arr[idx].mean(axis=1)
        written += b
    means = arr.mean(axis=0)
    lo = np.percentile(boot_means, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return means, lo, hi


def collect_sf():
    rows = []
    for f in SF_FILES:
        p = ROOT / f
        if p.exists():
            rows.extend(load_face_positive(p))
        else:
            print(f'WARN: missing {f}')
    return rows


def main():
    pool = []
    for name, fname in BENCHMARKS.items():
        pool.append((name, load_face_positive(ROOT / fname)))
    pool.append(('SynthForensics', collect_sf()))
    for name, fname in REALS.items():
        pool.append((name, load_face_positive(ROOT / fname)))

    # Header
    print(f'{"Entity":<18} {"n":>7}', end='')
    for label, _ in METRICS:
        print(f'  {label:>22}', end='')
    print()
    print('-' * (18 + 7 + 24 * len(METRICS)))

    for name, rows in pool:
        arr = cols_to_array(rows)
        n = arr.shape[0]
        print(f'{name:<18} {n:>7}', end='')
        m, lo, hi = bootstrap_means_ci(arr, B=B, seed=SEED)
        for j, (label, _) in enumerate(METRICS):
            print(f'  {m[j]:.3f}[{lo[j]:.3f},{hi[j]:.3f}]', end='')
        print()


if __name__ == '__main__':
    main()
