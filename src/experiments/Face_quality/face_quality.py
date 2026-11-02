#!/usr/bin/env python3
"""
Face Quality Assessment

Measures face image quality using TOPIQ-NR-Face (CGFIQA-trained).
Extracts 32 equidistant frames per video, runs face quality assessment,
and outputs per-video scores.

Usage:
    # Single worker on GPU (fastest per-video, model loaded once)
    python face_quality.py --video-dir /path/to/videos --dataset-name loki --device cuda --gpu-id 0

    # Multi-worker CPU (for large datasets)
    python face_quality.py --video-dir /path/to/videos --dataset-name aigvdbench --device cpu --workers 16 --recursive
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ─── Defaults ────────────────────────────────────────────────────────────────
NUM_FRAMES = 32
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Face Quality Assessment')
    parser.add_argument('--video-dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name for output files')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--num-frames', type=int, default=NUM_FRAMES, help='Frames to sample per video')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device for TOPIQ model')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--decode-device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Decord decode device')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers (CPU only, each loads own model)')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories')
    return parser.parse_args()


def get_frame_indices(total_frames, num_frames):
    """Get equidistant frame indices."""
    if total_frames <= num_frames:
        return list(range(total_frames))
    step = total_frames / num_frames
    return [int(round(i * step)) for i in range(num_frames)]


def decode_frames(video_path, num_frames, decode_device='cpu', gpu_id=0):
    """Decode equidistant frames from video using decord.

    Returns: (frames_rgb numpy array (N,H,W,3), total_frames) or (None, 0) on failure.
    """
    from decord import VideoReader, cpu as decord_cpu, gpu as decord_gpu

    ctx = decord_gpu(gpu_id) if decode_device == 'gpu' else decord_cpu(0)
    try:
        vr = VideoReader(str(video_path), ctx=ctx)
        total_frames = len(vr)
    except Exception as e:
        return None, 0

    if total_frames < 1:
        return None, 0

    indices = get_frame_indices(total_frames, num_frames)
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception:
        return None, total_frames

    return frames, total_frames


def score_frames(frames_rgb, metric, device):
    """Score each frame with TOPIQ-NR-Face.

    Args:
        frames_rgb: numpy (N, H, W, 3) uint8
        metric: loaded pyiqa metric
        device: torch device

    Returns: list of scores (floats), only for frames where face was detected.
    """
    scores = []
    for i in range(len(frames_rgb)):
        frame_tensor = torch.from_numpy(frames_rgb[i]).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)
        try:
            with torch.no_grad():
                score = metric(frame_tensor).item()
            scores.append(score)
        except Exception:
            continue  # face not detected
    return scores


def build_result(video_id, scores, frames_analyzed, total_frames):
    """Build result dict from scores."""
    if not scores:
        return {
            'video_id': video_id,
            'mean_score': -1.0,
            'std_score': -1.0,
            'min_score': -1.0,
            'max_score': -1.0,
            'faces_detected': 0,
            'frames_analyzed': frames_analyzed,
            'total_frames': total_frames,
        }
    return {
        'video_id': video_id,
        'mean_score': round(float(np.mean(scores)), 6),
        'std_score': round(float(np.std(scores)), 6),
        'min_score': round(float(np.min(scores)), 6),
        'max_score': round(float(np.max(scores)), 6),
        'faces_detected': len(scores),
        'frames_analyzed': frames_analyzed,
        'total_frames': total_frames,
    }


# ─── Single-worker GPU path (model loaded once) ──────────────────────────────

def run_single_worker(videos, args, output_csv):
    """Process all videos sequentially with one model instance."""
    import pyiqa

    device = torch.device(f'cuda:{args.gpu_id}' if args.device == 'cuda' else 'cpu')
    logger.info(f'Loading TOPIQ-NR-Face on {device}...')
    metric = pyiqa.create_metric('topiq_nr-face', device=device)
    logger.info('Model loaded.')

    results = []
    fieldnames = ['video_id', 'mean_score', 'std_score', 'min_score', 'max_score', 'faces_detected', 'frames_analyzed', 'total_frames']

    # Resume: load already processed video IDs
    already_done = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline='') as ef:
            for row in csv.DictReader(ef):
                already_done.add(row['video_id'])
        if already_done:
            logger.info(f'Resuming: {len(already_done)} videos already processed, skipping them')
            csv_file = open(output_csv, 'a', newline='')
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        else:
            csv_file = open(output_csv, 'w', newline='')
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    else:
        csv_file = open(output_csv, 'w', newline='')
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    csv_file.flush()

    t0 = time.time()
    for i, video_path in enumerate(videos, 1):
        video_id = str(video_path)
        if video_id in already_done:
            continue
        frames, total_frames = decode_frames(video_path, args.num_frames, args.decode_device, args.gpu_id)

        if frames is None:
            logger.warning(f'{video_id}: decode failed')
            continue

        scores = score_frames(frames, metric, device)
        result = build_result(video_id, scores, len(get_frame_indices(total_frames, args.num_frames)), total_frames)
        results.append(result)
        writer.writerow(result)
        csv_file.flush()

        if i % 10 == 0 or i == len(videos):
            elapsed = time.time() - t0
            logger.info(f'Progress: {i}/{len(videos)} ({len(results)} OK, {elapsed/i:.2f}s/video)')

    csv_file.close()
    elapsed = time.time() - t0
    return results, elapsed


# ─── Multi-worker CPU path (each worker loads own model) ─────────────────────

def _worker_process_batch(batch_args):
    """Worker function: loads model once, processes a batch of videos."""
    video_paths, num_frames, decode_device, gpu_id = batch_args
    import pyiqa

    device = torch.device('cpu')
    metric = pyiqa.create_metric('topiq_nr-face', device=device)

    batch_results = []
    for video_path in video_paths:
        video_id = str(video_path)
        frames, total_frames = decode_frames(video_path, num_frames, decode_device, gpu_id)
        if frames is None:
            continue
        scores = score_frames(frames, metric, device)
        indices = get_frame_indices(total_frames, num_frames)
        result = build_result(video_id, scores, len(indices), total_frames)
        batch_results.append(result)

    del metric
    return batch_results


def run_multi_worker(videos, args, output_csv):
    """Process videos in parallel, each worker handles a batch."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Split videos into batches (one per worker, evenly distributed)
    num_workers = args.workers
    video_paths = [str(v) for v in videos]
    batch_size = max(1, len(video_paths) // num_workers)
    batches = []
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i + batch_size]
        batches.append((batch, args.num_frames, args.decode_device, args.gpu_id))

    logger.info(f'Launching {len(batches)} worker batches ({batch_size} videos each)...')

    results = []
    fieldnames = ['video_id', 'mean_score', 'std_score', 'min_score', 'max_score', 'faces_detected', 'frames_analyzed', 'total_frames']

    csv_file = open(output_csv, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_process_batch, b): b for b in batches}
        done_count = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for r in batch_results:
                results.append(r)
                writer.writerow(r)
            csv_file.flush()
            done_count += len(batch_results)
            elapsed = time.time() - t0
            logger.info(f'Progress: ~{done_count}/{len(videos)} ({elapsed:.0f}s elapsed)')

    csv_file.close()
    elapsed = time.time() - t0
    return results, elapsed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f'{args.dataset_name}.csv'

    # Find videos
    if args.recursive:
        videos = sorted([f for f in video_dir.rglob('*') if f.suffix.lower() in VIDEO_EXTENSIONS])
    else:
        videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS])

    if not videos:
        logger.error(f'No videos found in {video_dir}')
        sys.exit(1)

    logger.info(f'Found {len(videos)} videos in {video_dir}')
    logger.info(f'Config: {args.num_frames} frames/video, device={args.device}, workers={args.workers}')

    # Run
    if args.workers > 1 and args.device == 'cpu':
        results, elapsed = run_multi_worker(videos, args, output_csv)
    else:
        results, elapsed = run_single_worker(videos, args, output_csv)

    logger.info(f'Done: {len(results)}/{len(videos)} videos in {elapsed:.1f}s ({elapsed/max(len(videos),1):.2f}s/video)')
    logger.info(f'CSV saved to {output_csv}')

    # Summary
    if results:
        valid_scores = [r['mean_score'] for r in results if r['mean_score'] >= 0]
        no_face = sum(1 for r in results if r['faces_detected'] == 0)

        summary = {
            'dataset': args.dataset_name,
            'video_dir': str(video_dir),
            'total_videos': len(videos),
            'processed_videos': len(results),
            'failed_videos': len(videos) - len(results),
            'config': {
                'num_frames': args.num_frames,
                'device': args.device,
                'workers': args.workers,
                'metric': 'topiq_nr-face',
                'metric_paper': 'TOPIQ (Chen et al., IEEE TIP 2024, arXiv:2308.03060)',
                'training_dataset': 'CGFIQA-40k (real faces only)',
            },
            'metrics': {
                'face_quality_score': {
                    'mean': round(float(np.mean(valid_scores)), 6) if valid_scores else None,
                    'std': round(float(np.std(valid_scores)), 6) if valid_scores else None,
                    'median': round(float(np.median(valid_scores)), 6) if valid_scores else None,
                    'min': round(float(np.min(valid_scores)), 6) if valid_scores else None,
                    'max': round(float(np.max(valid_scores)), 6) if valid_scores else None,
                    'num_valid': len(valid_scores),
                },
                'face_detection': {
                    'videos_with_face': len(valid_scores),
                    'videos_no_face': no_face,
                    'no_face_rate': round(no_face / len(results), 4) if results else None,
                },
            },
            'elapsed_seconds': round(elapsed, 1),
            'seconds_per_video': round(elapsed / max(len(videos), 1), 2),
        }

        summary_path = output_csv.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f'Summary saved to {summary_path}')
        logger.info(f'  Mean face quality: {summary["metrics"]["face_quality_score"]["mean"]}')
        logger.info(f'  No face rate: {summary["metrics"]["face_detection"]["no_face_rate"]}')


if __name__ == '__main__':
    main()
