"""
SynthForensics video dataset for NSG-VD.

Loads videos from a flat directory with an optional IDs filter file,
decoupled from the GenVideo directory structure.
"""

import os
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
from loguru import logger
from data.video_frames_dataset import process_video2frames


class SynthForensicsVideoDataset(Dataset):
    """
    Dataset that loads videos from a flat directory.

    For fake videos:
        - ids_file contains bare IDs (one per line, lines starting with # are ignored)
        - filename_suffix is appended to each bare ID to form the actual filename stem
          e.g. bare_id="000", suffix="_cogvideox" → file "000_cogvideox.mp4"

    For real videos:
        - ids_file=None → all .mp4 files in the directory are loaded
        - filename_suffix="" (files already named "{id}.mp4")

    Frames are extracted once and cached to frame_cache_dir/{video_id}/.
    """

    def __init__(
        self,
        video_dir: str,
        frame_cache_dir: str,
        num_frames: int = 8,
        label: str = "fake",
        ids_file: str = None,
        filename_suffix: str = "",
        load_len: int = None,
    ):
        assert os.path.isdir(video_dir), f"Video dir not found: {video_dir}"

        self.video_dir = video_dir
        self.frame_cache_dir = frame_cache_dir
        self.num_frames = num_frames
        self.label = label

        if ids_file is not None:
            bare_ids = []
            with open(ids_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        bare_ids.append(line)
            video_ids = [f"{bid}{filename_suffix}" for bid in bare_ids]
            # Keep only IDs whose video file actually exists
            video_ids = [
                vid for vid in video_ids
                if os.path.exists(os.path.join(video_dir, f"{vid}.mp4"))
            ]
        else:
            files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
            video_ids = [os.path.splitext(f)[0] for f in files]

        if load_len is not None:
            video_ids = video_ids[:load_len]

        self._extract_frames(video_ids)

        # Only keep videos for which frame extraction succeeded
        self.video_ids = [
            vid for vid in video_ids
            if os.path.isdir(os.path.join(frame_cache_dir, vid))
            and len(os.listdir(os.path.join(frame_cache_dir, vid))) == num_frames
        ]

        logger.info(
            f"SynthForensicsVideoDataset [{label}]: {len(self.video_ids)} videos "
            f"from {video_dir}"
        )
        assert len(self.video_ids) > 0, (
            f"No videos successfully processed in {video_dir}"
        )

        self.fakes = (
            np.ones(len(self.video_ids))
            if label == "fake"
            else np.zeros(len(self.video_ids))
        )

    def _extract_frames(self, video_ids):
        missing = [
            vid for vid in video_ids
            if not os.path.isdir(os.path.join(self.frame_cache_dir, vid))
            or len(os.listdir(os.path.join(self.frame_cache_dir, vid))) != self.num_frames
        ]
        if missing:
            logger.info(f"Extracting frames for {len(missing)} videos...")
            process_video2frames(
                self.video_dir, missing, self.num_frames, self.frame_cache_dir
            )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        frames = sorted(
            glob(os.path.join(self.frame_cache_dir, self.video_ids[idx], "*"))
        )
        return {
            "frames": frames,
            "fake": int(self.fakes[idx]),
            "num_frames": len(frames),
        }

    def get_video_frames(self, video_id):
        frames = sorted(
            glob(os.path.join(self.frame_cache_dir, video_id, "*"))
        )
        return {
            "frames": frames,
            "fake": int(self.label == "fake"),
            "num_frames": len(frames),
        }
