"""
Build the Canonical, CRF23 and CRF40 versions of a folder of Raw videos, as in SynthForensics.

- Canonical: each Raw video re-encoded with H.264 (libx264) at CRF 0, YUV420p, BT.709 color tags,
  at its original resolution and frame rate; audio, if any, is copied.
- CRF23 and CRF40: each Canonical video re-encoded with the same parameters at CRF 23 and CRF 40.

Output layout: <output_dir>/{canonical,crf23,crf40}/<same relative path as in input_dir>

The released versions were produced with this command and x264 core 164 r3095, using FFmpeg 6.0.1 for
all I2V videos and for the T2V videos of LTX-2.3, Helios and daVinci-MagiHuman, and FFmpeg 8.0 for the
T2V videos of CogVideoX, MAGI-1, Self-Forcing, SkyReels-V2 and Wan2.1. With the same versions this script
reproduces the released files byte for byte.

The FFmpeg version matters because the Raw videos have no color tags: when BT.709 tags are requested,
FFmpeg 8.0 converts the colors, while FFmpeg 6.0.1 keeps the pixel values. A different FFmpeg or x264
version, or a different number of x264 threads (chosen automatically from the CPU and the video size),
may produce files that are not byte-identical.

Usage:
    python src/generators/utils/make_compression_versions.py \\
        --input_dir /path/to/raw_videos --output_dir /path/to/output
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CANONICAL_CRF = 0
COMPRESSED = {'crf23': 23, 'crf40': 40}  # encoded from the Canonical version


def encode(ffmpeg, src, dst, crf):
    """Re-encode src into dst; written to a temporary file first so interrupted runs leave nothing partial."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + '.tmp' + dst.suffix)
    r = subprocess.run([
        ffmpeg, '-y', '-i', str(src),
        '-c:v', 'libx264', '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-c:a', 'copy', str(tmp),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src}:\n{r.stderr[-2000:]}")
    tmp.replace(dst)


def process(ffmpeg, raw, rel, output_dir):
    """Raw -> Canonical -> CRF23 / CRF40 for one video; versions that already exist are skipped."""
    canonical = output_dir / 'canonical' / rel
    if not canonical.exists():
        encode(ffmpeg, raw, canonical, CANONICAL_CRF)
    for name, crf in COMPRESSED.items():
        dst = output_dir / name / rel
        if not dst.exists():
            encode(ffmpeg, canonical, dst, crf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_dir', required=True, type=Path, help='Folder of Raw .mp4 videos (searched recursively)')
    p.add_argument('--output_dir', required=True, type=Path, help='Output folder for canonical/, crf23/ and crf40/')
    p.add_argument('--ffmpeg', default='ffmpeg', help='FFmpeg executable (default: ffmpeg)')
    p.add_argument('--workers', type=int, default=4, help='Videos processed in parallel (default: 4)')
    args = p.parse_args()

    version = subprocess.run([args.ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    print(version)
    raws = sorted(v for v in args.input_dir.rglob('*.mp4') if not v.name.endswith('.tmp.mp4'))
    print(f"{len(raws)} Raw videos in {args.input_dir}")
    for tmp in args.output_dir.rglob('*.tmp.mp4'):
        tmp.unlink()  # leftovers of an interrupted run

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, args.ffmpeg, raw, raw.relative_to(args.input_dir), args.output_dir): raw
                   for raw in raws}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
            except RuntimeError as e:
                failed.append(futures[f])
                print(e)
            print(f"\r{i}/{len(raws)} done", end='', flush=True)
    print(f"\nOutput: {args.output_dir}")
    if failed:
        raise SystemExit(f"{len(failed)} videos failed")


if __name__ == '__main__':
    main()
