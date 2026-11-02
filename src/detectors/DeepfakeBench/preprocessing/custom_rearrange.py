"""
Custom JSON generator for preprocessed frames.
Generates the JSON format expected by DeepfakeBench's abstract_dataset.py.

For meaningful metrics (AUC, ACC, EER) the JSON must contain BOTH fake and real
videos. Use --real_frames_dir to include real videos in the same JSON.

Run from: DeepfakeBench/preprocessing/

Usage (fake only — metrics will be meaningless):
    python custom_rearrange.py \
        --dataset_name magi-1 \
        --fake_frames_dir /path/to/processed/magi-1/frames \
        --output_dir ./dataset_json

Usage (fake + real — recommended):
    python custom_rearrange.py \
        --dataset_name magi-1 \
        --fake_frames_dir /path/to/processed/magi-1/frames \
        --real_frames_dir /path/to/processed/ffreal/frames \
        --output_dir ./dataset_json

Labels used:
    SynthForensics_fake: 1  (fake)
    SynthForensics_real: 0  (real)
Both must be present in training/config/test_config.yaml under label_dict.
"""

import os
import json
import glob
import argparse

FAKE_LABEL = 'SynthForensics_fake'
REAL_LABEL = 'SynthForensics_real'


def collect_videos(frames_dir, label):
    """Scan frames_dir/ subdirectories and return a dict of video entries."""
    frames_dir = os.path.abspath(frames_dir)
    if not os.path.isdir(frames_dir):
        raise ValueError(f"Directory not found: {frames_dir}")

    entries = {}
    video_dirs = sorted([d for d in os.scandir(frames_dir) if d.is_dir()], key=lambda d: d.name)
    skipped = 0
    for video_dir in video_dirs:
        frame_paths = sorted(
            glob.glob(os.path.join(video_dir.path, '*.png')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        if not frame_paths:
            print(f"  [WARN] No frames in {video_dir.path}, skipping.")
            skipped += 1
            continue
        entries[video_dir.name] = {'label': label, 'frames': frame_paths}

    print(f"  {label}: {len(entries)} videos loaded (skipped: {skipped})")
    return entries


def generate_json(dataset_name, fake_frames_dir, real_frames_dir, output_dir):
    dataset_dict = {dataset_name: {}}

    # Fake videos (required)
    fake_entries = collect_videos(fake_frames_dir, FAKE_LABEL)
    dataset_dict[dataset_name][FAKE_LABEL] = {
        'train': {},
        'val':   {k: v for k, v in fake_entries.items()},
        'test':  {k: v for k, v in fake_entries.items()},
    }

    # Real videos (optional but strongly recommended for valid metrics)
    if real_frames_dir:
        real_entries = collect_videos(real_frames_dir, REAL_LABEL)
        dataset_dict[dataset_name][REAL_LABEL] = {
            'train': {},
            'val':   {k: v for k, v in real_entries.items()},
            'test':  {k: v for k, v in real_entries.items()},
        }
    else:
        print("  [WARN] No real videos provided. AUC/ACC/EER will be meaningless.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, dataset_name + '.json')
    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=2)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='JSON generator for custom preprocessed datasets')
    parser.add_argument('--dataset_name', required=True,
                        help='Dataset name (JSON key and output filename, e.g. magi-1)')
    parser.add_argument('--fake_frames_dir', required=True,
                        help='frames/ directory for fake (synthetic) videos')
    parser.add_argument('--real_frames_dir', default=None,
                        help='frames/ directory for real videos (FF++ or DFD real). Strongly recommended.')
    parser.add_argument('--output_dir', default='./dataset_json',
                        help='Where to save the JSON file')
    args = parser.parse_args()

    print(f"Generating JSON for dataset: {args.dataset_name}")
    generate_json(args.dataset_name, args.fake_frames_dir, args.real_frames_dir, args.output_dir)


if __name__ == '__main__':
    main()
