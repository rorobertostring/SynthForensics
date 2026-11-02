#!/usr/bin/env python3
"""
01_extract_embeddings.py — ArcFace identity embeddings for the Identity Audit.

Extracts one ArcFace identity embedding per video (or per image, for the I2V
reference frames) with the insightface 'buffalo_l' model pack, of which only
the face detection and recognition models are loaded.

Per video: 32 equidistant frames (same sampling as the face-level analyses of
Section 4), face detection per frame, largest-bbox face kept, per-frame normed
embeddings averaged and re-normalized into a single per-video identity
embedding. Videos with no detected face on any sampled frame are recorded with
frames_with_face = 0 and no embedding.

Outputs (append/resume safe; already-done stems are skipped):
    {output-dir}/emb/{set-name}/{stem}.npy    — 512-d float32 embedding
    {output-dir}/{set-name}.csv               — per-video frame counts

Usage:
    python 01_extract_embeddings.py --video-dir /path/to/videos \
        --set-name t2v_cogvideox --gpu-id 0
    python 01_extract_embeddings.py --images-dir /path/to/frames \
        --set-name refframes --gpu-id 0
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

NUM_FRAMES = 32
DET_SIZE = 640
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

CSV_FIELDS = ['stem', 'frames_analyzed', 'frames_with_face']


def parse_args():
    p = argparse.ArgumentParser(description='Identity Audit: ArcFace embedding extraction')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video-dir', type=str)
    src.add_argument('--images-dir', type=str)
    p.add_argument('--set-name', type=str, required=True)
    p.add_argument('--output-dir', type=str, default=None,
                   help='Default: results/ next to this script')
    p.add_argument('--num-frames', type=int, default=NUM_FRAMES)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--start-index', type=int, default=None)
    p.add_argument('--end-index', type=int, default=None)
    return p.parse_args()


def get_frame_indices(total_frames, num_frames):
    if total_frames <= num_frames:
        return list(range(total_frames))
    step = total_frames / num_frames
    return [int(round(i * step)) for i in range(num_frames)]


def largest_face(faces):
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


def process_frames(app, frames_bgr):
    """Run detection+embedding on a list of BGR frames; aggregate per video."""
    embs = []
    for frame in frames_bgr:
        faces = app.get(frame)
        if not faces:
            continue
        embs.append(largest_face(faces).normed_embedding.astype(np.float32))
    if not embs:
        return None
    emb = np.mean(np.stack(embs), axis=0)
    emb = emb / np.linalg.norm(emb)
    return {'embedding': emb.astype(np.float32), 'frames_with_face': len(embs)}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'results'
    emb_dir = out_dir / 'emb' / args.set_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{args.set_name}.csv'

    # Resume: skip stems already present in the CSV.
    done = set()
    if csv_path.exists():
        with open(csv_path) as fh:
            done = {row['stem'] for row in csv.DictReader(fh)}
        logger.info(f'Resume: {len(done)} stems already done in {csv_path.name}')

    if args.video_dir:
        root = Path(args.video_dir)
        files = sorted(f for f in root.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
        mode = 'video'
    else:
        root = Path(args.images_dir)
        files = sorted(f for f in root.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        mode = 'image'
    if args.start_index is not None or args.end_index is not None:
        files = files[args.start_index:args.end_index]
    todo = [f for f in files if f.stem not in done]
    logger.info(f'{args.set_name}: {len(files)} files, {len(todo)} to process ({mode} mode)')
    if not todo:
        return

    import cv2
    from insightface.app import FaceAnalysis
    if mode == 'video':
        from decord import VideoReader, cpu as decord_cpu

    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                       providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=args.gpu_id, det_size=(DET_SIZE, DET_SIZE))

    write_header = not csv_path.exists()
    t_start = time.time()
    n_done, n_nofaces, n_errors = 0, 0, 0

    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, path in enumerate(todo):
            try:
                if mode == 'video':
                    vr = VideoReader(str(path), ctx=decord_cpu(0))
                    idxs = get_frame_indices(len(vr), args.num_frames)
                    batch = vr.get_batch(idxs).asnumpy()          # RGB
                    frames = [cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in batch]
                else:
                    img = cv2.imread(str(path))
                    frames = [img] if img is not None else []

                result = process_frames(app, frames)
                row = {'stem': path.stem, 'frames_analyzed': len(frames)}
                if result is None:
                    n_nofaces += 1
                    row['frames_with_face'] = 0
                else:
                    np.save(emb_dir / f'{path.stem}.npy', result['embedding'])
                    row['frames_with_face'] = result['frames_with_face']
                writer.writerow(row)
                fh.flush()
                n_done += 1
            except Exception as exc:                              # noqa: BLE001
                n_errors += 1
                logger.error(f'{path.name}: {exc}')

            if (i + 1) % 100 == 0 or (i + 1) == len(todo):
                rate = (i + 1) / (time.time() - t_start)
                eta_min = (len(todo) - i - 1) / rate / 60 if rate > 0 else 0
                logger.info(f'{args.set_name}: {i + 1}/{len(todo)} '
                            f'({rate:.2f} vid/s, ETA {eta_min:.0f} min, '
                            f'no-face {n_nofaces}, errors {n_errors})')

    logger.info(f'DONE {args.set_name}: {n_done} processed, '
                f'{n_nofaces} without faces, {n_errors} errors')
    if n_errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
