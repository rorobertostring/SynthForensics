"""
SynthForensics frame extraction script for D3 detector.

Same extraction as utils/video2frame.py (8fps, 3s clip, identical ffmpeg
command line), with the clip start times read from start_times.csv for
reproducibility.

Input layout:
    {real_root}/{DFD,FF++}/real_{compression}_test_set/*.mp4
    {fake_root}/{T2V,I2V}/videos/{compression}/{generator}/*.mp4

Output layout:
    {output_dir}/
    ├── frames/
    │   ├── real/{DFD,FF++}/{compression}/{video_stem}/
    │   └── fake/{t2v,i2v}/{compression}/{generator}/{video_stem}/
    ├── metadata/
    │   └── extraction_log.csv   (one row per video, incl. start_time/start_source)
    └── tmp/                     (in-progress extractions, removed at the end)

By default only the fake videos in the test split (SF-*_fake_ids_*.txt) are
extracted, i.e. the ones used by sf_run_all.py. Use --all-videos to extract
every fake video.

Usage:
    python SF_scripts/sf_extract_frames.py \\
        --real-root /path/to/data \\
        --fake-root /path/to/data/SynthForensics \\
        [--compressions raw] [--modalities t2v] [--generators cogvideox] \\
        [--workers 32] [--face-detection]
"""

import os
import sys
import math
import random
import shutil
import argparse
import subprocess
import multiprocessing
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

# ─── Constants ────────────────────────────────────────────────────────────────
REAL_DATASETS = ['DFD', 'FF++']
COMPRESSIONS  = ['canonical', 'crf23', 'crf40', 'raw']
MODALITIES    = ['t2v', 'i2v']
SPLIT_FILES   = ['SF-FF_fake_ids_140.txt', 'SF-DFD_fake_ids_363.txt']
VIDEO_EXTS    = ('.mp4', '.avi', '.mov', '.mkv', '.gif')

FRAME_RATE     = 8
DURATION       = 3
FFMPEG_VERSION = '8.0.1'  # version used for the original SynthForensics frames

# ─── Video utilities (same as sf_extract_frames.py) ───────────────────────────

def get_video_length(file_path):
    """Return video duration in seconds using cv2 (no moviepy dependency needed)."""
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


def get_video_info(file_path):
    cap = cv2.VideoCapture(file_path)
    info = {
        'fps':    cap.get(cv2.CAP_PROP_FPS),
        'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    info['duration'] = round(n / info['fps'], 3) if info['fps'] > 0 else 0.0
    return info


def check_face(frame_dir, n_check=4):
    """Return True if a face is detected in any of the first n_check frames."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    frames = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.split('.')[0])
    )[:n_check]
    for fname in frames:
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            continue
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) > 0:
            return True
    return False


def check_ffmpeg(ffmpeg):
    """Return the ffmpeg version line; warn if it differs from FFMPEG_VERSION."""
    try:
        out = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit(f'[ERROR] ffmpeg not found: {ffmpeg}')
    first = out.splitlines()[0] if out else 'unknown ffmpeg version'
    if not first.startswith(f'ffmpeg version {FFMPEG_VERSION} '):
        print(f'[WARN] {first}')
        print(f'[WARN] The original frames were extracted with ffmpeg {FFMPEG_VERSION}: '
              f'start times will match, but JPEG bytes may differ.')
    return first

# ─── Start times ──────────────────────────────────────────────────────────────

def load_start_times(csv_path):
    """Return {table_key: start_time} from start_times.csv."""
    df = pd.read_csv(csv_path)
    return dict(zip(df['video'], df['start_time'].astype(int)))


def table_key(video_path, root):
    """Key used in start_times.csv: video path relative to its root, '/'-separated."""
    return os.path.relpath(video_path, root).replace(os.sep, '/')


def fallback_start(key, video_length, seed):
    """
    Start time for videos missing from the table. Same formula as
    video2frame.py, but the random generator is seeded per video, so the
    value does not depend on which worker processes the video.
    """
    if video_length <= DURATION:
        return 0
    rng = random.Random(f'{seed}/{key}')
    return math.floor(rng.uniform(0, video_length - DURATION))

# ─── Split filtering ──────────────────────────────────────────────────────────

def load_split_ids(split_dir):
    """Union of the fake video IDs of all SF split files."""
    ids = set()
    for fname in SPLIT_FILES:
        with open(os.path.join(split_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ids.add(line)
    return ids


def video_id(stem, split_ids):
    """
    Return the split ID of a fake video stem, or None if it is not in the split.
    Tries increasingly long underscore-joined prefixes, e.g.
      '000_cogvideox_t2v'                → '000'
      '01__exit_phone_room_magi-1_i2v'   → '01__exit_phone_room'
    """
    parts = stem.split('_')
    for i in range(1, len(parts) + 1):
        candidate = '_'.join(parts[:i])
        if candidate in split_ids:
            return candidate
    return None

# ─── Frame extraction ─────────────────────────────────────────────────────────

def process_video(job):
    """
    Extract frames from a single video:
      - frame_rate = 8, duration = 3 s, same ffmpeg arguments as video2frame.py
      - start_time from start_times.csv, or fallback_start() if missing
      - frames are written to tmp_dir (outside frames/) and moved to image_path
        when complete, so an interrupted run never leaves a partially
        extracted directory where sf_run_all.py would read it
      - skip if the output directory already contains frames
    """
    video_path, image_path, tmp_dir, key, start_time, seed, face_detection, ffmpeg = job

    row = {
        'video_path':         video_path,
        'frames_dir':         image_path,
        'table_key':          key,
        'start_time':         start_time,
        'start_source':       'table' if start_time is not None else 'fallback',
        'status':             None,
        'error':              None,
        'duration':           None,
        'width':              None,
        'height':             None,
        'source_fps':         None,
        'n_frames_extracted': None,
        'has_face':           None,
    }

    final_dir = image_path.rstrip('/')

    if os.path.isdir(final_dir):
        existing = [f for f in os.listdir(final_dir) if f.endswith('.jpg')]
        if existing:
            row['status']             = 'skipped'
            row['n_frames_extracted'] = len(existing)
            if face_detection:
                row['has_face'] = check_face(final_dir)
            return row

    try:
        if start_time is None:
            start_time        = fallback_start(key, get_video_length(video_path), seed)
            row['start_time'] = start_time

        shutil.rmtree(tmp_dir, ignore_errors=True)  # leftover of an interrupted run
        os.makedirs(tmp_dir)

        # Same arguments, in the same order, as video2frame.py: -ss before -i
        ret = subprocess.run([
            ffmpeg, '-loglevel', 'quiet',
            '-ss', str(start_time), '-t', str(DURATION),
            '-i', video_path,
            '-vf', f'fps={FRAME_RATE}',
            os.path.join(tmp_dir, '%d.jpg'),
        ]).returncode

        extracted = [f for f in os.listdir(tmp_dir) if f.endswith('.jpg')]
        row['n_frames_extracted'] = len(extracted)

        info = get_video_info(video_path)
        row.update({
            'duration':   info['duration'],
            'width':      info['width'],
            'height':     info['height'],
            'source_fps': round(info['fps'], 2),
        })

        if not extracted:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            row['status'] = 'error_no_frames'
            return row

        if os.path.isdir(final_dir):
            shutil.rmtree(final_dir)  # empty directory left by a previous run
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)
        os.rename(tmp_dir, final_dir)

        if len(extracted) < 8:
            row['status'] = 'warning_few_frames'
        elif ret != 0:
            row['status'] = 'ffmpeg_error'
        else:
            row['status'] = 'ok'

        if face_detection:
            row['has_face'] = check_face(final_dir)

    except Exception as e:
        row['status'] = 'error'
        row['error']  = str(e)

    return row

# ─── Video collection ─────────────────────────────────────────────────────────

def list_videos(video_dir):
    """Sorted list of video paths in video_dir (sorted: order-independent runs)."""
    return [
        os.path.join(video_dir, vf)
        for vf in sorted(os.listdir(video_dir))
        if vf.lower().endswith(VIDEO_EXTS)
    ]


def collect_jobs(args, start_times, split_ids):
    """
    Build the list of job tuples for process_video().
    Returns (jobs, missing_keys) where missing_keys are videos not in the table.
    """
    frames_root = os.path.join(args.output_dir, 'frames')
    tmp_root    = os.path.join(args.output_dir, 'tmp')
    jobs, missing = [], []

    def add_job(video_path, out_dir, root):
        key     = table_key(video_path, root)
        start   = start_times.get(key)
        tmp_dir = os.path.join(tmp_root, os.path.relpath(out_dir, frames_root))
        if start is None:
            missing.append(key)
        jobs.append((video_path, out_dir, tmp_dir, key, start, args.seed,
                     args.face_detection, args.ffmpeg))

    # Real videos
    if not args.skip_real:
        for ds in args.real_datasets:
            for comp in args.compressions:
                video_dir = os.path.join(args.real_root, ds, f'real_{comp}_test_set')
                if not os.path.isdir(video_dir):
                    print(f'[WARN] missing: {video_dir}')
                    continue
                for vp in list_videos(video_dir):
                    out_dir = os.path.join(frames_root, 'real', ds, comp, Path(vp).stem) + '/'
                    add_job(vp, out_dir, args.real_root)

    # Fake videos (T2V / I2V)
    if not args.skip_fake:
        for mod in args.modalities:
            for comp in args.compressions:
                comp_dir = os.path.join(args.fake_root, mod.upper(), 'videos', comp)
                if not os.path.isdir(comp_dir):
                    print(f'[WARN] missing: {comp_dir}')
                    continue
                generators = args.generators or sorted(
                    d for d in os.listdir(comp_dir)
                    if os.path.isdir(os.path.join(comp_dir, d))
                )
                for gen in generators:
                    video_dir = os.path.join(comp_dir, gen)
                    if not os.path.isdir(video_dir):
                        print(f'[WARN] missing: {video_dir}')
                        continue
                    for vp in list_videos(video_dir):
                        stem = Path(vp).stem
                        if not args.all_videos and video_id(stem, split_ids) is None:
                            continue
                        out_dir = os.path.join(frames_root, 'fake', mod, comp, gen, stem) + '/'
                        add_job(vp, out_dir, args.fake_root)

    return jobs, missing

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract frames from SynthForensics dataset for D3 detector '
                    '(start times from start_times.csv).'
    )
    parser.add_argument('--real-root', type=str, required=True,
                        help='Folder containing DFD/, FF++/ with real_{compression}_test_set/')
    parser.add_argument('--fake-root', type=str, required=True,
                        help='SynthForensics root containing T2V/ and I2V/')
    parser.add_argument('--start-times', type=str,
                        default=os.path.join(SCRIPT_DIR, 'start_times.csv'),
                        help='CSV with the original clip start times (default: SF_scripts/start_times.csv)')
    parser.add_argument('--split-dir', type=str, default=BASE_DIR,
                        help='Folder with the SF-*_fake_ids_*.txt split files (default: D3 root)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(BASE_DIR, 'preprocessing'),
                        help='Output root, frames/ and metadata/ are created inside (default: D3/preprocessing)')
    parser.add_argument('--real-datasets', nargs='+', default=REAL_DATASETS, choices=REAL_DATASETS)
    parser.add_argument('--compressions',  nargs='+', default=COMPRESSIONS,  choices=COMPRESSIONS)
    parser.add_argument('--modalities',    nargs='+', default=MODALITIES,    choices=MODALITIES)
    parser.add_argument('--generators',    nargs='+', default=None,
                        help='Generator folders to process (default: all folders found)')
    parser.add_argument('--skip-real', action='store_true', help='Do not extract real videos')
    parser.add_argument('--skip-fake', action='store_true', help='Do not extract fake videos')
    parser.add_argument('--all-videos', action='store_true',
                        help='Extract all fake videos, not only the test split ones')
    parser.add_argument('--strict', action='store_true',
                        help='Fail if a selected video has no entry in start_times.csv')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the per-video fallback start time (default: 42)')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg',
                        help='ffmpeg executable (default: ffmpeg from PATH)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers (default: 32, same as video2frame.py)')
    parser.add_argument('--face-detection', action='store_true',
                        help='Run face detection on extracted frames and save result to metadata')
    args = parser.parse_args()

    ffmpeg_version = check_ffmpeg(args.ffmpeg)
    # Clip start times of the reported results, for reproducibility
    start_times    = load_start_times(args.start_times)
    split_ids      = set() if (args.skip_fake or args.all_videos) else load_split_ids(args.split_dir)

    print(f'Real root  : {args.real_root}')
    print(f'Fake root  : {args.fake_root}')
    print(f'Output dir : {args.output_dir}')
    print(f'ffmpeg     : {ffmpeg_version}')
    print(f'Start times: {len(start_times)} entries ({args.start_times})')
    if split_ids:
        print(f'Split IDs  : {len(split_ids)} (union of {", ".join(SPLIT_FILES)})')
    print(f'Workers    : {args.workers}')
    print(f'Face detect: {args.face_detection}')

    jobs, missing = collect_jobs(args, start_times, split_ids)
    print(f'\nFound {len(jobs)} videos to process '
          f'({len(jobs) - len(missing)} with table start time, {len(missing)} fallback).')
    if missing:
        for key in missing[:10]:
            print(f'  [no table entry] {key}')
        if len(missing) > 10:
            print(f'  ... and {len(missing) - 10} more')
        if args.strict:
            sys.exit('[ERROR] --strict: some videos are missing from start_times.csv')
    print()

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_video, jobs),
            total=len(jobs),
            desc='Extracting frames'
        ))

    # All workers are done: whatever is left in tmp/ belongs to failed videos
    shutil.rmtree(os.path.join(args.output_dir, 'tmp'), ignore_errors=True)

    # Save metadata
    meta_dir = os.path.join(args.output_dir, 'metadata')
    os.makedirs(meta_dir, exist_ok=True)
    log_path = os.path.join(meta_dir, 'extraction_log.csv')
    df_log = pd.DataFrame(results)
    df_log.to_csv(log_path, index=False)
    print(f'\nMetadata saved to {log_path}')

    # Summary
    print('\n--- Extraction summary ---')
    print(df_log['status'].value_counts().to_string())
    print('\n--- Start time source ---')
    print(df_log['start_source'].value_counts().to_string())
    if args.face_detection:
        no_face = df_log[df_log['has_face'] == False]
        nf_path = os.path.join(meta_dir, 'no_face_videos.csv')
        no_face[['video_path', 'frames_dir']].to_csv(nf_path, index=False)
        print(f'\nVideos without face: {len(no_face)} → saved to {nf_path}')
