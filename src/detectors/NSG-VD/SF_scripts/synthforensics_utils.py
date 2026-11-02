"""
Shared utilities for SynthForensics preprocessing and evaluation scripts.
"""

import os
from omegaconf import DictConfig
from data.synthforensics_score_dataset import SynthForensicsScoreFeaturesDataset

# ---------------------------------------------------------------------------
# Generator → filename suffix maps
# ---------------------------------------------------------------------------
T2V_SUFFIX_MAP = {
    "cogvideox":         "_cogvideox_t2v",
    "ltx2-3":            "_ltx_2-3_t2v",
    "magi-1":            "_magi-1_t2v",
    "self-forcing":      "_self-forcing_t2v",
    "skyreels-v2":       "_skyreels-v2_t2v",
    "wan2-1":            "_wan2-1_t2v",
    "daVinci-MagiHuman": "_daVinci-MagiHuman_t2v",
    "helios":            "_helios_t2v",
}

I2V_SUFFIX_MAP = {
    "cogvideox":         "_cogvideox_i2v",
    "ltx2-3":            "_ltx_2-3_i2v",
    "magi-1":            "_magi-1_i2v",
    "skyreels-v2":       "_skyreels-v2_i2v",
    "wan2-1":            "_wan2-1_i2v",
    "daVinci-MagiHuman": "_daVinci-MagiHuman_i2v",
    "helios":            "_helios_i2v",
}

# Real dataset name → folder under real_dir
REAL_DIR_MAP = {
    "FF":  "FF++",
    "DFD": "DFD",
}

REAL_SUBDIR_TMPL = "real_{compression}_test_set"

SPLIT_FILE_MAP = {
    "FF":  "SF-FF_fake_ids_140.txt",
    "DFD": "SF-DFD_fake_ids_363.txt",
}


def build_score_dataset(
    video_dir: str,
    ids_file: str,
    filename_suffix: str,
    frame_cache_dir: str,
    nsg_cache_dir: str,
    label: str,
    cfg: DictConfig,
    load_len: int = None,
):
    """Instantiate a SynthForensicsScoreFeaturesDataset with standard NSG-VD params."""
    return SynthForensicsScoreFeaturesDataset(
        video_dir=video_dir,
        ids_file=ids_file,
        filename_suffix=filename_suffix,
        frame_cache_dir=frame_cache_dir,
        nsg_cache_dir=nsg_cache_dir,
        label=label,
        score_config_path="./libs/eps_ad/imagnet.yml",
        score_args_path="./libs/eps_ad/args.yml",
        num_frames=cfg.data.num_frames,
        input_shape=tuple(cfg.data.input_shape),
        diffuse_steps=cfg.data.diffuse_steps,
        feature_type=cfg.data.feature_type,
        process_batch_size=3,
        device="cuda",
        mode="test",
        load_len=load_len,
        filter_nsg=False,
        resolution_size=cfg.data.resolution_size,
    )


def iter_combinations(cfg: DictConfig):
    """
    Yield (video_type, generator, suffix, compression, real_dataset,
           fake_video_dir, real_video_dir, split_file,
           fake_frame_cache, fake_nsg_cache, real_frame_cache, real_nsg_cache)
    for every valid combination in the config.
    Skips empty or missing directories silently (logs a warning).
    """
    from loguru import logger

    sf = cfg.synthforensics

    generator_maps = [
        ("T2V", list(sf.t2v_generators), T2V_SUFFIX_MAP, sf.t2v_videos_dir),
        ("I2V", list(sf.i2v_generators), I2V_SUFFIX_MAP, sf.i2v_videos_dir),
    ]

    for video_type, generators, suffix_map, videos_root in generator_maps:
        for real_dataset in sf.real_datasets:
            split_file = sf.get("split_file_override") or os.path.join(
                sf.splits_dir, SPLIT_FILE_MAP[real_dataset]
            )
            for compression in sf.compressions:
                real_video_dir = os.path.join(
                    sf.real_dir, REAL_DIR_MAP[real_dataset],
                    REAL_SUBDIR_TMPL.format(compression=compression),
                )
                if not os.path.isdir(real_video_dir):
                    logger.warning(f"Skipping missing real dir: {real_video_dir}")
                    continue

                real_frame_cache = os.path.join(
                    sf.frames_cache_dir, "real", real_dataset, compression
                )
                real_nsg_cache = os.path.join(
                    sf.nsg_cache_dir, "real", real_dataset, compression
                )

                for generator in generators:
                    fake_video_dir = os.path.join(videos_root, compression, generator)
                    if not os.path.isdir(fake_video_dir) or not any(
                        f.endswith(".mp4") for f in os.listdir(fake_video_dir)
                    ):
                        logger.warning(f"Skipping empty generator dir: {fake_video_dir}")
                        continue

                    suffix = suffix_map.get(generator, f"_{generator}")
                    fake_frame_cache = os.path.join(
                        sf.frames_cache_dir, video_type, generator, compression
                    )
                    fake_nsg_cache = os.path.join(
                        sf.nsg_cache_dir, video_type, generator, compression, real_dataset
                    )

                    yield (
                        video_type, generator, suffix, compression, real_dataset,
                        fake_video_dir, real_video_dir, split_file,
                        fake_frame_cache, fake_nsg_cache,
                        real_frame_cache, real_nsg_cache,
                    )
