#!/usr/bin/env python3
"""
Landmark Stability Assessment

Extracts 68 facial landmarks from 32 equidistant frames using FAN (2DFAN4),
computes per-landmark confidence scores (heatmap peak values), and reports
completeness at multiple thresholds.

Usage:
    python landmark_quality.py --video-dir /path/to/videos --dataset-name loki --gpu-id 0
    python landmark_quality.py --video-dir /path/to/videos --dataset-name aigvdbench --gpu-id 2 --recursive
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

NUM_FRAMES = 32
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Landmark Stability Assessment (FAN)')
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: results_{num_frames})')
    parser.add_argument('--num-frames', type=int, default=NUM_FRAMES)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--start-index', type=int, default=None, help='Start index in video list (inclusive)')
    parser.add_argument('--end-index', type=int, default=None, help='End index in video list (exclusive)')
    return parser.parse_args()


def get_frame_indices(total_frames, num_frames):
    if total_frames <= num_frames:
        return list(range(total_frames))
    step = total_frames / num_frames
    return [int(round(i * step)) for i in range(num_frames)]


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(f'results_{args.num_frames}')
    output_dir.mkdir(parents=True, exist_ok=True)

    import face_alignment
    from decord import VideoReader, cpu as decord_cpu

    # Find videos
    if args.recursive:
        videos = sorted([f for f in video_dir.rglob('*') if f.suffix.lower() in VIDEO_EXTENSIONS])
    else:
        videos = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS])

    if not videos:
        logger.error(f'No videos found in {video_dir}')
        sys.exit(1)

    # Slice if start/end specified
    total_found = len(videos)
    start = args.start_index if args.start_index is not None else 0
    end = args.end_index if args.end_index is not None else len(videos)
    videos = videos[start:end]

    if args.start_index is not None or args.end_index is not None:
        output_csv = output_dir / f'{args.dataset_name}_{start}_{end}.csv'
    else:
        output_csv = output_dir / f'{args.dataset_name}.csv'

    logger.info(f'Found {total_found} videos, processing {len(videos)} (index {start}-{end})')
    logger.info(f'Config: {args.num_frames} frames, gpu_id={args.gpu_id}')

    # Load FAN once
    device = f'cuda:{args.gpu_id}'
    logger.info(f'Loading FAN on {device}...')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    logger.info('FAN loaded.')

    # CSV setup
    fieldnames = [
        'video_id', 'faces_detected', 'frames_analyzed', 'total_frames',
        'mean_confidence', 'min_confidence', 'max_confidence',
    ] + [f'completeness_th_{th}' for th in THRESHOLDS]

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

    # Accumulators for dataset summary
    all_rows = []
    t0 = time.time()

    for i, vp in enumerate(videos, 1):
        if str(vp) in already_done:
            continue
        # Decode
        try:
            vr = VideoReader(str(vp), ctx=decord_cpu(0))
            total_frames = len(vr)
        except Exception:
            continue

        if total_frames < 1:
            continue

        indices = get_frame_indices(total_frames, args.num_frames)

        try:
            frames_batch = vr.get_batch(indices).asnumpy()
        except Exception:
            continue

        # FAN per-frame inference
        all_conf = []  # list of (68,) arrays
        for frame in frames_batch:
            try:
                results = fa.get_landmarks_from_image(
                    frame, return_landmark_score=True, return_bboxes=True
                )
                if results[0] is not None:
                    all_conf.append(results[1][0])  # first face, (68,)
            except Exception:
                continue

        faces_detected = len(all_conf)
        frames_analyzed = len(indices)

        # Compute metrics
        row = {
            'video_id': str(vp),
            'faces_detected': faces_detected,
            'frames_analyzed': frames_analyzed,
            'total_frames': total_frames,
        }

        if faces_detected > 0:
            conf_array = np.array(all_conf)  # (N, 68)
            row['mean_confidence'] = round(float(np.mean(conf_array)), 6)
            row['min_confidence'] = round(float(np.min(conf_array)), 6)
            row['max_confidence'] = round(float(np.max(conf_array)), 6)
            for th in THRESHOLDS:
                complete = np.all(conf_array >= th, axis=1)  # (N,) bool
                row[f'completeness_th_{th}'] = round(float(np.mean(complete)), 4)
        else:
            row['mean_confidence'] = -1.0
            row['min_confidence'] = -1.0
            row['max_confidence'] = -1.0
            for th in THRESHOLDS:
                row[f'completeness_th_{th}'] = 0.0

        all_rows.append(row)
        writer.writerow(row)
        csv_file.flush()

        if i % 10 == 0 or i == len(videos):
            elapsed = time.time() - t0
            logger.info(f'Progress: {i}/{len(videos)} ({len(all_rows)} OK, {elapsed/i:.2f}s/video)')

    csv_file.close()
    elapsed = time.time() - t0
    logger.info(f'Done: {len(all_rows)}/{len(videos)} in {elapsed:.1f}s ({elapsed/max(len(videos),1):.2f}s/video)')
    logger.info(f'CSV saved to {output_csv}')

    # Summary
    if all_rows:
        valid_conf = [r['mean_confidence'] for r in all_rows if r['mean_confidence'] >= 0]
        no_face = sum(1 for r in all_rows if r['faces_detected'] == 0)

        summary = {
            'dataset': args.dataset_name,
            'video_dir': str(video_dir),
            'total_videos': len(videos),
            'processed_videos': len(all_rows),
            'config': {
                'num_frames': args.num_frames,
                'gpu_id': args.gpu_id,
                'method': 'FAN 2DFAN4 (Bulat & Tzimiropoulos, ICCV 2017)',
                'library': 'face-alignment (1adrianb/face-alignment)',
                'landmarks': 68,
                'confidence_source': 'Heatmap peak values per landmark (0-1)',
                'thresholds': THRESHOLDS,
            },
            'metrics': {
                'confidence': {
                    'mean': round(float(np.mean(valid_conf)), 6) if valid_conf else None,
                    'std': round(float(np.std(valid_conf)), 6) if valid_conf else None,
                    'median': round(float(np.median(valid_conf)), 6) if valid_conf else None,
                    'num_valid': len(valid_conf),
                },
                'no_face_videos': no_face,
                'no_face_rate': round(no_face / len(all_rows), 4) if all_rows else None,
                'completeness_at_threshold': {},
            },
            'elapsed_seconds': round(elapsed, 1),
            'seconds_per_video': round(elapsed / max(len(videos), 1), 2),
        }

        # Completeness summary
        print(f'\n{"="*60}')
        print(f'Dataset: {args.dataset_name} ({len(all_rows)} videos, {no_face} no-face)')
        print(f'{"="*60}')
        print(f'Confidence: mean={summary["metrics"]["confidence"]["mean"]}, median={summary["metrics"]["confidence"]["median"]}')
        print(f'\nCompleteness (% frames where ALL 68 landmarks >= threshold):')
        print(f'{"Threshold":>10} {"Mean":>10} {"Median":>10}')
        print('-' * 35)

        for th in THRESHOLDS:
            col = f'completeness_th_{th}'
            vals = [r[col] for r in all_rows if r['mean_confidence'] >= 0]
            if vals:
                m = round(float(np.mean(vals)), 4)
                med = round(float(np.median(vals)), 4)
            else:
                m = med = None
            summary['metrics']['completeness_at_threshold'][str(th)] = {'mean': m, 'median': med}
            print(f'{th:>10.1f} {m:>10} {med:>10}')

        summary_path = output_csv.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()
