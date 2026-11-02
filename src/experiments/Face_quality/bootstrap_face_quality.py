#!/usr/bin/env python3
"""Bootstrap 95% CI on per-benchmark mean face quality Q (numpy-vectorized).

For each entity in the comparative pool, computes the mean of TOPIQ-NR-Face
score Q on the face-positive subset (faces_detected > 0), and reports a 95%
percentile bootstrap CI with B=1000 resamples (sampling videos with
replacement).
"""
import csv
from pathlib import Path
import numpy as np

ROOT = Path('/path/to/results')
SEED = 42
B = 1000
CHUNK = 100

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
    'sf_wan2-1_t2v_raw.csv',
    'sf_skyreels-v2_t2v_raw.csv', 'sf_skyreels_i2v_raw.csv',
    'sf_self-forcing_t2v_raw.csv',
    'sf_ltx2-3_t2v_raw.csv',    'sf_ltx_i2v_raw.csv',
    'sf-helios-t2v.csv',        'sf-helios-i2v.csv',
    'sf-davinci-t2v.csv',       'sf-davinci-i2v.csv',
    'sf-wan21-i2v.csv',
]


def load_q(path):
    out = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fd = int(row['faces_detected'])
                q = float(row['mean_score'])
            except (KeyError, ValueError):
                continue
            if fd > 0 and q >= 0:
                out.append(q)
    return np.asarray(out, dtype=np.float64)


def bootstrap_ci(values, B=1000, chunk=CHUNK, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return None, None, None
    boot_means = np.empty(B, dtype=np.float64)
    written = 0
    while written < B:
        b = min(chunk, B - written)
        idx = rng.integers(0, n, size=(b, n))
        boot_means[written:written + b] = values[idx].mean(axis=1)
        written += b
    return float(values.mean()), float(np.percentile(boot_means, 100 * alpha / 2)), \
           float(np.percentile(boot_means, 100 * (1 - alpha / 2)))


def collect_sf():
    parts = []
    files = list(SF_FILES)
    for f in files:
        p = ROOT / f
        if p.exists():
            parts.append(load_q(p))
        else:
            print(f'WARN: missing {f}')
    return np.concatenate(parts)


def main():
    pool = []
    for name, fname in BENCHMARKS.items():
        pool.append((name, load_q(ROOT / fname)))
    pool.append(('SynthForensics', collect_sf()))
    for name, fname in REALS.items():
        pool.append((name, load_q(ROOT / fname)))

    print(f'{"Entity":<18} {"n":>8}  {"Q":>6}  {"CI_lo":>6}  {"CI_hi":>6}')
    print('-' * 56)
    for name, vals in pool:
        m, lo, hi = bootstrap_ci(vals, B=B, seed=SEED)
        print(f'{name:<18} {len(vals):>8}  {m:>6.3f}  {lo:>6.3f}  {hi:>6.3f}')


if __name__ == '__main__':
    main()
