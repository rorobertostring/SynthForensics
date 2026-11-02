"""
SynthForensics preprocessing for NSG-VD.

For every combination of (video_type, generator, compression, real_dataset):
  1. Extracts video frames with ffmpeg (cached to frames_cache_dir)
  2. Computes NSG features via the diffusion model (cached to nsg_cache_dir)

Run this once before test_synthforensics.py. Subsequent runs are instant
because both caches are checked before re-extracting.

Usage:
    conda activate nsg-vd
    cd /path/to/NSG-VD
    python SF_scripts/preprocess_synthforensics.py
    # subset (Hydra overrides):
    python SF_scripts/preprocess_synthforensics.py 'synthforensics.compressions=[raw]' \
        'synthforensics.t2v_generators=[cogvideox]' 'synthforensics.i2v_generators=[]' \
        'synthforensics.real_datasets=[FF]'
"""

import os
import sys
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from tqdm import tqdm

# NSG-VD root (data/, utils/, models/, libs/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthforensics_utils import build_score_dataset, iter_combinations


@hydra.main(config_path="../configs/synthforensics", config_name="test.yaml", version_base=None)
def main(cfg: DictConfig):
    os.chdir(get_original_cwd())
    logger.info(OmegaConf.to_yaml(cfg))

    combos = list(iter_combinations(cfg))
    logger.info(f"Total combinations to preprocess: {len(combos)}")

    # Track which real (real_dataset, compression) pairs we've already processed
    # so we don't re-extract real features multiple times
    processed_real = set()

    for combo in tqdm(combos, desc="Preprocessing combinations"):
        (
            video_type, generator, suffix, compression, real_dataset,
            fake_video_dir, real_video_dir, split_file,
            fake_frame_cache, fake_nsg_cache,
            real_frame_cache, real_nsg_cache,
        ) = combo

        # ------------------------------------------------------------------ #
        #  Real videos — extract once per (real_dataset, compression)
        # ------------------------------------------------------------------ #
        real_key = (real_dataset, compression)
        if real_key not in processed_real:
            logger.info(f"Preprocessing real [{real_dataset}/{compression}]...")
            try:
                build_score_dataset(
                    video_dir=real_video_dir,
                    ids_file=None,
                    filename_suffix="",
                    frame_cache_dir=real_frame_cache,
                    nsg_cache_dir=real_nsg_cache,
                    label="real",
                    cfg=cfg,
                )
                processed_real.add(real_key)
            except Exception as e:
                logger.error(f"Failed real preprocessing [{real_dataset}/{compression}]: {e}")
                continue

        # ------------------------------------------------------------------ #
        #  Fake videos
        # ------------------------------------------------------------------ #
        combo_name = f"{video_type}/{generator}/{compression}/{real_dataset}"
        logger.info(f"Preprocessing fake [{combo_name}]...")
        try:
            build_score_dataset(
                video_dir=fake_video_dir,
                ids_file=split_file,
                filename_suffix=suffix,
                frame_cache_dir=fake_frame_cache,
                nsg_cache_dir=fake_nsg_cache,
                label="fake",
                cfg=cfg,
            )
        except Exception as e:
            logger.error(f"Failed fake preprocessing [{combo_name}]: {e}")

    logger.success("Preprocessing complete.")


if __name__ == "__main__":
    main()
