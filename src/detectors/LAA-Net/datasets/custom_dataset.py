# -*- coding: utf-8 -*-
"""
CustomDataset: loads real/fake face frames directly from two folder trees.

Expected structure:
    REAL_ROOT/
        video_001/
            frame_000.png
            frame_001.png
            ...
        video_002/
            ...

    FAKE_ROOT/
        video_001/
            frame_000.png
            ...

Each subfolder = one video. Returns (image_tensor, label, video_id).
Frames are already cropped/preprocessed — only resize + normalize is applied.
"""
import os
from glob import glob

import cv2
import numpy as np

from .builder import DATASETS
from .common import CommonDataset

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


@DATASETS.register_module()
class CustomDataset(CommonDataset):
    """
    Dataset for evaluating LAA-Net on a custom set of real/fake frames.

    Required config fields (under DATASET.DATA.TEST):
        REAL_ROOT : absolute path to folder containing real video subfolders
        FAKE_ROOT : absolute path to folder containing fake video subfolders
        FROM_FILE : False
    """

    def __init__(self, config, split, **kwargs):
        self.split = split
        if kwargs is not None:
            for k, v in kwargs.items():
                if v is None:
                    raise ValueError(f"{k}:{v} retrieved a None value!")
                self.__setattr__(k, v)
        super().__init__(config, **kwargs)

        self.image_size = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])  # (H, W)
        self.image_paths, self.labels, self.video_ids = self._load_data(split)

    # ------------------------------------------------------------------
    # Required abstract implementations
    # ------------------------------------------------------------------

    def _load_from_path(self, split):
        data_cfg = self._cfg.DATA[split.upper()]
        real_root = data_cfg.REAL_ROOT
        fake_root = data_cfg.FAKE_ROOT

        assert os.path.isdir(real_root), f"REAL_ROOT not found: {real_root}"
        assert os.path.isdir(fake_root), f"FAKE_ROOT not found: {fake_root}"

        image_paths, labels, video_ids = [], [], []

        for root, label in [(real_root, 0), (fake_root, 1)]:
            for vid_name in sorted(os.listdir(root)):
                vid_dir = os.path.join(root, vid_name)
                if not os.path.isdir(vid_dir):
                    continue
                frames = self._get_frames(vid_dir)
                if not frames:
                    print(f"[CustomDataset] WARNING: no frames found in {vid_dir}")
                    continue
                image_paths.extend(frames)
                labels.extend([label] * len(frames))
                video_ids.extend([vid_name] * len(frames))

        return image_paths, labels, video_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]),
                         interpolation=cv2.INTER_LINEAR)

        label = np.expand_dims(self.labels[idx], axis=-1)
        vid_id = self.video_ids[idx]

        img_tensor = self.final_transforms(img / 255)
        return img_tensor, label, vid_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_frames(self, folder):
        frames = []
        for ext in IMAGE_EXTENSIONS:
            frames += glob(os.path.join(folder, f"*{ext}"))
            frames += glob(os.path.join(folder, f"*{ext.upper()}"))
        return sorted(set(frames))

    def _load_data(self, split):
        image_paths, labels, video_ids = self._load_from_path(split)

        assert len(image_paths) > 0, (
            "No images loaded. Check REAL_ROOT and FAKE_ROOT in the config."
        )
        n_fake = sum(labels)
        n_real = len(labels) - n_fake
        print(f"[CustomDataset] Loaded {len(image_paths)} frames "
              f"({n_real} real, {n_fake} fake) "
              f"from {len(set(video_ids))} videos")
        return image_paths, np.array(labels), video_ids
