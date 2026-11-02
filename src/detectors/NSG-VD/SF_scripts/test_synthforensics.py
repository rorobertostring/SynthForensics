"""
NSG-VD evaluation on the SynthForensics dataset.

Assumes preprocessing (frame extraction + NSG feature caching) has already
been run via preprocess_synthforensics.py.

Iterates over all combinations of:
  video_type  : T2V | I2V
  generator   : cogvideox, ltx2-3, …
  compression : raw, canonical, crf23, crf40
  real_dataset: FF, DFD

Results are saved to one CSV per (video_type, real_dataset).

Usage:
    conda activate nsg-vd
    cd /path/to/NSG-VD
    python SF_scripts/test_synthforensics.py
    # subset (Hydra overrides):
    python SF_scripts/test_synthforensics.py 'synthforensics.compressions=[raw]' \
        'synthforensics.t2v_generators=[cogvideox]' 'synthforensics.i2v_generators=[]' \
        'synthforensics.real_datasets=[FF]'
    # different checkpoint:
    python SF_scripts/test_synthforensics.py ckpt_path=./ckpts/standard-SEINE-mp.pth
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from loguru import logger
from tabulate import tabulate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import hydra
from hydra.utils import get_original_cwd

# NSG-VD root (data/, utils/, models/, libs/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.experiment_utils import set_seed
from utils.train_utils import get_ref_features
from utils.data_utils import get_data_loaders_for_mmd
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from test_dMMD import test_dMMD
from synthforensics_utils import build_score_dataset, iter_combinations


@hydra.main(config_path="../configs/synthforensics", config_name="test.yaml", version_base=None)
def main(cfg: DictConfig):
    os.chdir(get_original_cwd())
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # ------------------------------------------------------------------ #
    #  Load pre-trained MMD model
    # ------------------------------------------------------------------ #
    logger.info(f"Loading checkpoint: {cfg.ckpt_path}")
    discriminator = SingleSwinBlockDiscriminator(num_features=cfg.model.feature_dim)
    model = deep_MMD(
        discriminator=discriminator,
        sigma=cfg.model.sigma,
        sigma0=cfg.model.sigma0,
        epsilon=cfg.model.epsilon,
        img_size=cfg.model.img_size,
        is_yy_zero=cfg.model.is_yy_zero,
        is_smooth=cfg.model.is_smooth,
    )
    model.load_state_dict(torch.load(cfg.ckpt_path, weights_only=True))
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        model.net = nn.DataParallel(
            model.net, device_ids=cfg.trainer.device_ids[: cfg.trainer.num_gpus]
        )
    model = model.to(device)
    model.eval()

    sf = cfg.synthforensics
    all_results = {}  # (video_type, real_dataset) → list of rows

    # Cache ref features per (real_dataset, compression) to avoid recomputing
    ref_cache = {}

    for combo in iter_combinations(cfg):
        (
            video_type, generator, suffix, compression, real_dataset,
            fake_video_dir, real_video_dir, split_file,
            fake_frame_cache, fake_nsg_cache,
            real_frame_cache, real_nsg_cache,
        ) = combo

        table_key = (video_type, real_dataset)
        all_results.setdefault(table_key, [])

        # ------------------------------------------------------------------ #
        #  Real reference features — computed once per (real_dataset, compression)
        # ------------------------------------------------------------------ #
        real_key = (real_dataset, compression)
        if real_key not in ref_cache:
            logger.info(f"Loading real reference [{real_dataset}/{compression}]...")
            try:
                real_ref_dataset = build_score_dataset(
                    video_dir=real_video_dir,
                    ids_file=None,
                    filename_suffix="",
                    frame_cache_dir=real_frame_cache,
                    nsg_cache_dir=real_nsg_cache,
                    label="real",
                    cfg=cfg,
                    load_len=cfg.data.ref_load_len,
                )
            except Exception as e:
                logger.error(f"Failed to load real reference [{real_dataset}/{compression}]: {e}")
                continue
            ref_loader = DataLoader(
                real_ref_dataset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
            )
            feature_ref, ref_data = get_ref_features(model, ref_loader, cfg.data.ref_load_len)
            ref_cache[real_key] = (feature_ref.cuda(), ref_data)

        feature_ref, ref_data = ref_cache[real_key]

        # ------------------------------------------------------------------ #
        #  Fake + real test datasets
        # ------------------------------------------------------------------ #
        combo_name = f"{video_type}/{generator}/{compression}/{real_dataset}"
        logger.info(f"Testing: {combo_name}")

        try:
            fake_dataset = build_score_dataset(
                video_dir=fake_video_dir,
                ids_file=split_file,
                filename_suffix=suffix,
                frame_cache_dir=fake_frame_cache,
                nsg_cache_dir=fake_nsg_cache,
                label="fake",
                cfg=cfg,
            )
        except Exception as e:
            logger.error(f"Failed to load fake dataset [{combo_name}]: {e}")
            continue

        real_test_dataset = build_score_dataset(
            video_dir=real_video_dir,
            ids_file=None,
            filename_suffix="",
            frame_cache_dir=real_frame_cache,
            nsg_cache_dir=real_nsg_cache,
            label="real",
            cfg=cfg,
            load_len=None,
        )

        test_loader = get_data_loaders_for_mmd(
            cfg.data,
            {"fake": fake_dataset, "real": real_test_dataset},
            batch_size=cfg.data.val_batch_size,
        )

        with torch.no_grad():
            metrics = test_dMMD(model, test_loader, feature_ref, ref_data, regression_model=None)

        row = [
            combo_name,
            metrics["recall"],
            metrics["accuracy"],
            metrics["f1"],
            metrics["auroc"],
            metrics["precision"],
        ]
        all_results[table_key].append(row)
        logger.info(
            f"{combo_name} | Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | AUROC={metrics['auroc']:.4f}"
        )

    # ------------------------------------------------------------------ #
    #  Save results — one CSV per (video_type, real_dataset)
    # ------------------------------------------------------------------ #
    headers = ["Combination", "Recall", "Accuracy", "F1", "AUROC", "Precision"]
    os.makedirs(cfg.log_path, exist_ok=True)

    for (video_type, real_dataset), rows in all_results.items():
        if not rows:
            continue
        mean_row = ["Avg."] + [
            sum(r[i] for r in rows) / len(rows) for i in range(1, len(headers))
        ]
        rows_with_avg = rows + [mean_row]
        logger.info(
            f"\n=== {video_type} vs {real_dataset} ===\n"
            + tabulate(rows_with_avg, headers=headers, tablefmt="grid")
        )
        csv_name = f"{cfg.save_csv_file.replace('.csv', '')}_{video_type}_{real_dataset}.csv"
        csv_path = os.path.join(cfg.log_path, csv_name)
        df = pd.DataFrame(rows_with_avg, columns=headers)
        df = df.map(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
        df.to_csv(csv_path, index=False)
        logger.success(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
