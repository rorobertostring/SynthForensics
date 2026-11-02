#!/usr/bin/env python3
"""
02_identity_analysis.py — Identity Audit: analysis over the ArcFace embeddings.

Consumes the outputs of 01_extract_embeddings.py (results/emb/{set}/*.npy and
results/{set}.csv) and produces the Identity Audit statistics of
Appendix B.7.3.

Calibration:
  - different-person pairs: FF++ x DFD reference frames (YouTube subjects vs
    hired actors, disjoint populations by construction). Same-corpus pairs are
    not used, since FF++ contains subjects recurring across videos and DFD
    scenes are recorded with pairs of actors.
  - tau: 99th percentile of the different-person pairs (false-match rate 1%),
    the threshold above which two faces are declared the same person.
  - same-person reference: pristine source video vs its own reference frame.
  - same-person floor: 1st percentile of the same-person reference; 99% of
    the same-person comparisons score above it.

Analyses:
  A. Calibration (above).
  B. I2V identity preservation vs own reference frame, per generator: mean
     similarity and share of videos above tau.
  C. T2V similarity to the subject of their own source video, per generator.
  D. T2V videos generated from the same caption (same source, different
     generators): mean similarity.
  E. Identity count: FF++ subjects by connected components of the FF++
     reference frames at the same-person floor, documented DFD actors from
     the file names, and T2V videos (one synthetic individual each).

Run (after all extraction jobs are complete):
    python 02_identity_analysis.py

Outputs in results/analysis/: calibration.csv, i2v_preservation.csv,
t2v_source_similarity.csv, summary.txt
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / 'results'
EMB = RESULTS / 'emb'
OUT = RESULTS / 'analysis'

T2V_GENERATORS = ['cogvideox', 'daVinci', 'helios', 'ltx2-3', 'magi-1',
                  'self-forcing', 'skyreels-v2', 'wan2-1']
I2V_GENERATORS = ['cogvideox', 'daVinci', 'helios', 'ltx2-3', 'magi-1',
                  'skyreels-v2', 'wan2-1']
FFPP_RE = re.compile(r'^\d{3}$')


def load_set(name: str) -> dict[str, np.ndarray]:
    d = {}
    set_dir = EMB / name
    if not set_dir.exists():
        raise FileNotFoundError(f'missing embeddings dir: {set_dir}')
    for f in sorted(set_dir.glob('*.npy')):
        d[f.stem] = np.load(f)
    return d


def source_stem_of(video_stem: str, known_stems: list[str]) -> str | None:
    best = None
    for s in known_stems:
        if video_stem.startswith(s) and (best is None or len(s) > len(best)):
            best = s
    return best


def cos_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = []

    def log(msg=''):
        print(msg)
        report.append(msg)

    # ---------- Load ----------
    refs = load_set('refframes')
    sources = {**load_set('sources_ffpp'), **load_set('sources_dfd')}
    source_stems = sorted(refs.keys())
    log(f'refframes: {len(refs)}, sources: {len(sources)}')

    t2v, i2v = {}, {}
    for g in T2V_GENERATORS:
        for stem, e in load_set(f't2v_{g}').items():
            t2v[(g, stem)] = e
    for g in I2V_GENERATORS:
        for stem, e in load_set(f'i2v_{g}').items():
            i2v[(g, stem)] = e
    log(f'T2V embeddings: {len(t2v)}, I2V embeddings: {len(i2v)}')

    def map_to_source(d):
        out = {}
        for (g, stem) in d:
            src = source_stem_of(stem, source_stems)
            if src is not None:
                out[(g, stem)] = src
        return out

    t2v_src = map_to_source(t2v)
    i2v_src = map_to_source(i2v)

    # ---------- A. Calibration ----------
    is_dfd = np.array([not FFPP_RE.match(s) for s in source_stems])

    same_pairs = [(sources[s], refs[s]) for s in source_stems
                  if s in sources and s in refs]
    same = cos_pairs(np.stack([a for a, _ in same_pairs]),
                     np.stack([b for _, b in same_pairs]))
    floor = float(np.percentile(same, 1))

    R = np.stack([refs[s] for s in source_stems])
    null = (R[~is_dfd] @ R[is_dfd].T).ravel()
    tau = float(np.percentile(null, 99))

    pd.DataFrame([
        {'distribution': 'same person (source video vs own reference frame)',
         'n': len(same), 'mean': float(np.mean(same)), 'p1_same_person_floor': floor},
        {'distribution': 'different people (FF++ x DFD reference frames)',
         'n': len(null), 'mean': float(np.mean(null)), 'p99_tau': tau},
    ]).to_csv(OUT / 'calibration.csv', index=False)
    log(f'\n=== A. CALIBRATION ===')
    log(f'different people (FF++ x DFD): n={len(null):,}  mean {np.mean(null):.4f}  '
        f'tau (p99, FMR 1%) = {tau:.3f}')
    log(f'same person (source vs own reference frame): n={len(same):,}  '
        f'mean {np.mean(same):.3f}  same-person floor (p1) = {floor:.3f}')

    # ---------- B. I2V identity preservation ----------
    rows, all_sims = [], []
    log(f'\n=== B. I2V IDENTITY PRESERVATION (vs own reference frame) ===')
    for g in I2V_GENERATORS:
        pairs = [(i2v[k], refs[i2v_src[k]]) for k in i2v if k[0] == g and k in i2v_src]
        sims = cos_pairs(np.stack([a for a, _ in pairs]),
                         np.stack([b for _, b in pairs]))
        all_sims.append(sims)
        rows.append({'generator': g, 'n': len(sims), 'mean_similarity': float(np.mean(sims)),
                     'identity_preserved_pct': float(np.mean(sims >= tau)) * 100})
    allv = np.concatenate(all_sims)
    rows.append({'generator': 'all videos', 'n': len(allv), 'mean_similarity': float(np.mean(allv)),
                 'identity_preserved_pct': float(np.mean(allv >= tau)) * 100})
    pd.DataFrame(rows).to_csv(OUT / 'i2v_preservation.csv', index=False)
    for r in rows:
        log(f"{r['generator']:12s} mean {r['mean_similarity']:.3f}  "
            f"identity preserved {r['identity_preserved_pct']:.1f}%  (n={r['n']})")

    # ---------- C. T2V similarity to own source subject ----------
    rows = []
    log(f'\n=== C. T2V SIMILARITY TO OWN SOURCE SUBJECT ===')
    t2v_keys = sorted(t2v_src)
    T = np.stack([t2v[k] for k in t2v_keys])
    own = np.array([float(t2v[k] @ refs[t2v_src[k]]) for k in t2v_keys])
    gens = np.array([g for (g, _) in t2v_keys])
    for g in T2V_GENERATORS:
        x = own[gens == g]
        rows.append({'generator': g, 'n': len(x), 'mean_similarity': float(np.mean(x))})
    rows.append({'generator': 'all videos', 'n': len(own), 'mean_similarity': float(np.mean(own))})
    pd.DataFrame(rows).to_csv(OUT / 't2v_source_similarity.csv', index=False)
    for r in rows:
        log(f"{r['generator']:12s} mean {r['mean_similarity']:.3f}  (n={r['n']})")

    # ---------- D. T2V videos from the same caption ----------
    srcs = np.array([t2v_src[k] for k in t2v_keys])
    sib = []
    for s in source_stems:
        idx = np.nonzero(srcs == s)[0]
        if len(idx) >= 2:
            E = T[idx]
            sib.append((E @ E.T)[np.triu_indices(len(idx), k=1)])
    sib = np.concatenate(sib)
    log(f'\n=== D. T2V VIDEOS FROM THE SAME CAPTION ===')
    log(f'mean similarity {np.mean(sib):.3f}  (n={len(sib):,} pairs)')

    # ---------- E. Identity count ----------
    ffpp_idx = np.nonzero(~is_dfd)[0]
    F = R[ffpp_idx]
    S = F @ F.T
    uf = UnionFind(len(F))
    for i, j in zip(*np.nonzero(np.triu(S >= floor, k=1))):
        uf.union(int(i), int(j))
    n_ffpp_subjects = len({uf.find(i) for i in range(len(F))})
    n_dfd_actors = len({s.split('__')[0] for s, d in zip(source_stems, is_dfd) if d})
    n_real = n_ffpp_subjects + n_dfd_actors
    n_t2v_videos = sum(len(pd.read_csv(RESULTS / f't2v_{g}.csv')) for g in T2V_GENERATORS)
    log(f'\n=== E. IDENTITY COUNT ===')
    log(f'FF++ subjects (components of {len(F)} reference frames at floor {floor:.3f}): {n_ffpp_subjects}')
    log(f'DFD documented actors (from file names): {n_dfd_actors}')
    log(f'real people animated by I2V: {n_real}')
    log(f'T2V videos (one synthetic individual each): {n_t2v_videos:,}')
    log(f'individuals depicted: {n_t2v_videos + n_real:,}')

    (OUT / 'summary.txt').write_text('\n'.join(report) + '\n')
    print(f'\nSaved all outputs to {OUT}')


if __name__ == '__main__':
    main()
