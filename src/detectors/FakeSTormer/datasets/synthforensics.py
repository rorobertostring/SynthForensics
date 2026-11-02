#-*- coding: utf-8 -*-
import os

from glob import glob
import numpy as np

from .builder import DATASETS
from .common import CommonDataset


@DATASETS.register_module()
class SynthForensics(CommonDataset):
    """Dataset loader for the SynthForensics benchmark.

    Expected on-disk layout (produced by preprocess_synthforensics.py):

        {ROOT}/test/videos/{faketype}/{video_id}/*.png

    where ``faketype`` is ``real`` for genuine videos and the generator name
    (e.g. ``cogvideox``, ``magi-1``, …) for synthetic ones.

    Labels:  0 → real,  1 → fake (any generator listed).

    FAKETYPE in the config controls which directories are loaded:
      - Explicit list, e.g. ``[real, cogvideox]``.
      - Use ``"*"`` as a wildcard to auto-discover every fake directory:
        ``[real, "*"]``  →  loads real + all generator sub-folders.

    Adding a new generator only requires dropping its extracted frames under
    ``{ROOT}/test/videos/{new_generator}/`` — no code changes needed.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    # ------------------------------------------------------------------
    def _discover_generators(self, videos_dir: str) -> list:
        """Return sorted list of all subdirectory names that are not 'real'."""
        if not os.path.isdir(videos_dir):
            return []
        return sorted(
            d for d in os.listdir(videos_dir)
            if os.path.isdir(os.path.join(videos_dir, d)) and d != "real"
        )

    # ------------------------------------------------------------------
    def _load_from_path(self, split):
        root = self._cfg.DATA[self.split.upper()].ROOT
        assert os.path.exists(root), \
            f"[SynthForensics] Root path does not exist: {root}"

        data_type = self._cfg["DATA"].TYPE           # typically "videos"
        fake_types_cfg = list(self._cfg.DATA[split.upper()]["FAKETYPE"])

        # Expand wildcard "*" → all generator directories
        videos_dir = os.path.join(root, self.split, data_type)
        if "*" in fake_types_cfg:
            generators = self._discover_generators(videos_dir)
            fake_types = [ft for ft in fake_types_cfg if ft != "*"] + generators
            print(f"[SynthForensics] auto-discovered generators: {generators}")
        else:
            fake_types = fake_types_cfg

        img_paths, labels, mask_paths, ot_props = [], [], [], []

        for ft in fake_types:
            label = 0 if ft == "real" else 1
            data_dir = os.path.join(videos_dir, ft)

            if not os.path.exists(data_dir):
                print(f"[SynthForensics] directory not found, skipping: {data_dir}")
                continue

            for vid_id in sorted(os.listdir(data_dir)):
                vid_path = os.path.join(data_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                frames = glob(f"{vid_path}/*.{self._cfg.IMAGE_SUFFIX}")
                if not frames:
                    continue
                img_paths.extend(frames)
                labels.extend([label] * len(frames))

        n_real = sum(1 for lb in labels if lb == 0)
        n_fake = sum(1 for lb in labels if lb == 1)
        print(f"[SynthForensics] {len(img_paths)} frames loaded "
              f"({n_real} real, {n_fake} fake) — faketypes: {fake_types}")
        return img_paths, labels, mask_paths, ot_props
