"""
Custom preprocessing script for any flat folder of MP4 videos.
Extracts aligned face crops using dlib (same logic as preprocess.py).

Run from: DeepfakeBench/preprocessing/

Usage:
    python custom_preprocess.py \
        --video_dir /path/to/videos \
        --output_dir /path/to/output \
        --num_frames 32 \
        --mode fixed_num_frames

Output structure:
    output_dir/
    ├── frames/
    │   ├── video_stem/
    │   │   ├── 000.png
    │   │   └── ...
    └── landmarks/
        └── video_stem/
            ├── 000.npy
            └── ...
"""

import os
import cv2
import dlib
import glob
import shutil
import argparse
import numpy as np
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans

PREDICTOR_PATH = './dlib_tools/shape_predictor_81_face_landmarks.dat'
DONE_MARKER = '.done'  # written in a video's frames folder once it is fully processed


def get_keypts(image, face, predictor):
    shape = predictor(image, face)
    leye   = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye   = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose   = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def img_align_crop(img, landmark, outsize):
    target_size = [112, 112]
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    dst[:, 0] += 8.0
    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]
    target_size = outsize
    scale = 1.3
    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.
    y_margin = target_size[1] * margin_rate / 2.
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)
    src = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))
    img = cv2.resize(img, (outsize[1], outsize[0]))
    return img


def extract_aligned_face(face_detector, predictor, image, res=256):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if not faces:
        return None, None
    face = max(faces, key=lambda r: r.width() * r.height())
    landmarks = get_keypts(rgb, face, predictor)
    cropped = img_align_crop(rgb, landmarks, outsize=(res, res))
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    # Re-detect landmarks on the aligned face
    face_align = face_detector(cropped, 1)
    if not face_align:
        return None, None
    landmark_81 = predictor(cropped, face_align[0])
    landmark_81 = face_utils.shape_to_np(landmark_81)
    return cropped, landmark_81


def process_video(movie_path, output_dir, mode, num_frames, stride):
    movie_path = Path(movie_path)
    output_dir = Path(output_dir)
    frame_save_dir = output_dir / 'frames' / movie_path.stem
    land_save_dir  = output_dir / 'landmarks' / movie_path.stem

    if (frame_save_dir / DONE_MARKER).exists():
        return  # already processed, skip
    # Partial output from an interrupted run: start this video from scratch
    for d in (frame_save_dir, land_save_dir):
        if d.exists():
            shutil.rmtree(d)

    face_detector = dlib.get_frontal_face_detector()
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(f"Predictor not found: {PREDICTOR_PATH}")
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(str(movie_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open: {movie_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if mode == 'fixed_num_frames':
        frame_idxs = set(np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int))
    else:
        frame_idxs = set(np.arange(0, frame_count, stride, dtype=int))
    frame_save_dir.mkdir(parents=True, exist_ok=True)
    land_save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for cnt_frame in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt_frame not in frame_idxs:
            continue
        cropped, landmarks = extract_aligned_face(face_detector, face_predictor, frame)
        if cropped is None:
            print(f"  [WARN] No face in frame {cnt_frame} of {movie_path.name}")
            continue
        img_path = frame_save_dir / f"{cnt_frame:03d}.png"
        if not img_path.exists():
            cv2.imwrite(str(img_path), cropped)
        npy_path = land_save_dir / f"{cnt_frame:03d}.npy"
        np.save(str(npy_path), landmarks)
        saved += 1

    cap.release()
    if saved == 0:
        print(f"  [WARN] No faces saved for {movie_path.name}")
    (frame_save_dir / DONE_MARKER).touch()


def main():
    parser = argparse.ArgumentParser(description='Face crop extractor for custom video folders')
    parser.add_argument('--video_dir',  required=True, help='Folder containing .mp4 files')
    parser.add_argument('--output_dir', required=True, help='Where to save frames/ and landmarks/')
    parser.add_argument('--num_frames', type=int, default=32, help='Frames per video (fixed_num_frames mode)')
    parser.add_argument('--stride',     type=int, default=10, help='Frame stride (fixed_stride mode)')
    parser.add_argument('--mode', choices=['fixed_num_frames', 'fixed_stride'], default='fixed_num_frames')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.video_dir, '*.mp4')))
    if not videos:
        print(f"No .mp4 files found in {args.video_dir}")
        return

    num_workers = args.workers or min(os.cpu_count(), 8)
    print(f"Found {len(videos)} videos. Processing with {num_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_video, v, args.output_dir, args.mode, args.num_frames, args.stride): v
            for v in videos
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(videos)):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] {futures[future]}: {e}")

    print("Preprocessing complete.")


if __name__ == '__main__':
    main()
