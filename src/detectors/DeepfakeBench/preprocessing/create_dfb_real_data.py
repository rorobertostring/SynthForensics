"""
Build the real test set folders used by DeepfakeBench from the original
FaceForensics++ and DeepFakeDetection videos.

Output layout (default: <DeepfakeBench>/DFB_Data/real):

    <DATASET>/real_<COMP>_test_set/<video>.mp4

<DATASET> is FF++ or DFD; <COMP> is raw, canonical, crf23 or crf40.
raw videos are symbolic links to the originals; the other versions are
re-encoded with the same ffmpeg settings used for SynthForensics.
Test videos:
    FF++ = FF++ test IDs from splits/test.json (140, numeric IDs)
    DFD  = DFD test IDs from splits/test.json (363)

Usage (from the DeepfakeBench root):
    python preprocessing/create_dfb_real_data.py \\
        --ffpp_dir /path/to/FF++/original_sequences/youtube/raw/videos \\
        --dfd_dir  /path/to/DFD/original_sequences/actors/c23/videos
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

DFB_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_JSON = DFB_ROOT.parents[2] / 'splits' / 'test.json'
DEFAULT_OUTPUT_DIR = DFB_ROOT / 'DFB_Data' / 'real'

# Compression -> CRF (raw is linked, not re-encoded)
CRF = {'canonical': 0, 'crf23': 23, 'crf40': 40}


def encode(src, dst, crf):
    """Re-encode src into dst; written to a temporary file first so interrupted runs leave nothing partial."""
    tmp = dst.with_name(dst.stem + '.tmp' + dst.suffix)
    subprocess.run([
        'ffmpeg', '-y', '-i', str(src),
        '-c:v', 'libx264', '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-c:a', 'copy', str(tmp),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.rename(dst)


def test_videos(args):
    """Return {dataset: [source video paths]} for the real test sets."""
    with open(args.test_json) as f:
        ids = json.load(f)
    return {
        'FF++': [args.ffpp_dir / f"{i}.mp4" for i in sorted(ids) if '__' not in i],
        'DFD':  [args.dfd_dir / f"{i}.mp4" for i in sorted(ids) if '__' in i],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ffpp_dir', required=True, type=Path,
                   help='FF++ original videos (original_sequences/youtube/raw/videos)')
    p.add_argument('--dfd_dir', required=True, type=Path,
                   help='DFD original videos (original_sequences/actors/c23/videos)')
    p.add_argument('--output_dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                   help='Output folder (default: <DeepfakeBench>/DFB_Data/real)')
    p.add_argument('--test_json', type=Path, default=DEFAULT_TEST_JSON,
                   help='SynthForensics test IDs (default: splits/test.json of this repository)')
    p.add_argument('--workers', type=int, default=8, help='Parallel ffmpeg processes')
    p.add_argument('--dry_run', action='store_true', help='Print counts without creating anything')
    args = p.parse_args()

    videos = test_videos(args)
    missing = [v for vs in videos.values() for v in vs if not v.is_file()]
    for ds, vs in videos.items():
        print(f"{ds}: {len(vs)} test videos")
    if missing:
        print(f"\nMissing sources: {len(missing)}")
        for v in missing[:20]:
            print(f"  MISSING {v}")
        raise SystemExit(1)
    if args.dry_run:
        return

    out_root = args.output_dir.resolve()
    jobs = []
    for ds, vs in videos.items():
        raw_dir = out_root / ds / 'real_raw_test_set'
        raw_dir.mkdir(parents=True, exist_ok=True)
        for src in vs:
            dst = raw_dir / src.name
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())
        for comp, crf in CRF.items():
            comp_dir = out_root / ds / f"real_{comp}_test_set"
            comp_dir.mkdir(parents=True, exist_ok=True)
            for tmp in comp_dir.glob('*.tmp.mp4'):
                tmp.unlink()  # leftovers of an interrupted run
            jobs += [(src, comp_dir / src.name, crf) for src in vs if not (comp_dir / src.name).exists()]

    print(f"\nVideos to encode: {len(jobs)}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(lambda j: encode(*j), jobs), total=len(jobs)))
    print(f"\nOutput: {out_root}")


if __name__ == '__main__':
    main()
