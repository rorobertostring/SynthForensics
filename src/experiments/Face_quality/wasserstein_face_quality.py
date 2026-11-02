#!/usr/bin/env python3
"""Compute Wasserstein distances between per-video Q distributions.

Loads per-video TOPIQ-NR-Face mean scores from the face_quality.py results directory (ROOT),
restricts to the face-positive subset (faces_detected > 0), and reports
1D Wasserstein distance between every benchmark and the two real-video
baselines (FF++, DFD), plus pairwise SF<->FF++<->DFD.
"""
import csv
from pathlib import Path
from scipy.stats import wasserstein_distance

ROOT = Path('/path/to/results')

# Canonical filename (prefer _wo_duplicates when available).
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

REALS = {'FF++': 'ff++_real.csv', 'DFD': 'dfd_real.csv'}


def load_q(path):
    """Return per-video mean_score for face-positive videos."""
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
    return out


def main():
    # Load real-video distributions
    reals = {k: load_q(ROOT / v) for k, v in REALS.items()}

    # Load SF as union of all generator files
    sf = []
    for f in SF_FILES:
        p = ROOT / f
        if p.exists():
            sf.extend(load_q(p))
        else:
            print(f'WARNING: missing {f}')
    sf_dist = sf

    # Load benchmark distributions
    bench = {k: load_q(ROOT / v) for k, v in BENCHMARKS.items()}

    # Print summary
    print(f'{"Distribution":<15} {"n":>8} {"mean":>8} {"std":>8}')
    print('-' * 45)
    import statistics
    def stats(d):
        return f'{len(d):>8} {statistics.mean(d):>8.3f} {statistics.stdev(d):>8.3f}'
    print(f'{"FF++":<15} {stats(reals["FF++"])}')
    print(f'{"DFD":<15} {stats(reals["DFD"])}')
    print(f'{"SynthForensics":<15} {stats(sf_dist)}')
    for k, d in bench.items():
        print(f'{k:<15} {stats(d)}')

    print()
    print(f'Wasserstein-1D distances:')
    print(f'{"":<15} {"vs FF++":>10} {"vs DFD":>10}')
    print('-' * 38)

    def w(a, b):
        return wasserstein_distance(a, b)

    print(f'{"DFD":<15} {w(reals["DFD"], reals["FF++"]):>10.4f} {0.0:>10.4f}')
    print(f'{"FF++":<15} {0.0:>10.4f} {w(reals["FF++"], reals["DFD"]):>10.4f}')
    print(f'{"SynthForensics":<15} {w(sf_dist, reals["FF++"]):>10.4f} '
          f'{w(sf_dist, reals["DFD"]):>10.4f}')
    for k, d in bench.items():
        print(f'{k:<15} {w(d, reals["FF++"]):>10.4f} {w(d, reals["DFD"]):>10.4f}')


if __name__ == '__main__':
    main()
