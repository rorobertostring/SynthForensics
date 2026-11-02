"""
Experiment configs for GenD training/fine-tuning on the SynthForensics dataset.

Data layout (produced by SF_scripts/extract_frames_training.py):
  preprocessed_gend/training/
    real_ffpp/real/{id}/frame_XXXXXX.png   ← label 0
    real_dfd/real/{id}/frame_XXXXXX.png    ← label 0
    t2v/{gen}/{id}/frame_XXXXXX.png        ← label 1
    i2v/{gen}/{id}/frame_XXXXXX.png        ← label 1

Split txt files (one per source and split, listing the PNG paths of the videos
of that split in splits/*.json; the file name ends with the generator name as
written in _IN_DOMAIN / _OUT_DOMAIN, e.g. t2v_cogvideox.txt, i2v_daVinci.txt):
  config/datasets/gend/training/{train,val,test}/*.txt

Generator pools (sf-scratch):
  In-Domain  (train/val): cogvideox, wan2-1, ltx2-3, magi-1  — both t2v and i2v
  Out-of-Domain (test):   skyreels-v2, self-forcing, helios, daVinci — applicable modalities

Experiments:
  sf-finetune  — fine-tune LayerNorm layers on top of GenD_CLIP_L_14 pretrained checkpoint
  sf-scratch   — train CLIP-L/14 from scratch, in-domain generators only,
                 tested on out-of-domain generators
"""

import glob
import os

from .. import config as C
from ..config import Config

_SPLIT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "datasets", "gend", "training",
)

_LAYERNORM_CLIP = ["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"]

_IN_DOMAIN  = {"cogvideox", "wan2-1", "ltx2-3", "magi-1"}
_OUT_DOMAIN = {"skyreels-v2", "self-forcing", "helios", "daVinci"}


def _txts(split):
    """Glob all .txt files for a given split (train / val / test)."""
    return sorted(glob.glob(os.path.join(_SPLIT_DIR, split, "*.txt")))


def _txts_in_domain(split):
    """Real sources + in-domain fake generators for train/val."""
    return [f for f in _txts(split)
            if not any(f"_{g}." in os.path.basename(f) for g in _OUT_DOMAIN)]


def _txts_out_domain(split):
    """Real sources + out-of-domain fake generators for test."""
    return [f for f in _txts(split)
            if not any(f"_{g}." in os.path.basename(f) for g in _IN_DOMAIN)]


experiments = {
    # ------------------------------------------------------------------
    # Fine-tuning: start from GenD_CLIP_L_14 pretrained weights,
    # keep backbone frozen except LayerNorm layers (~0.03% trainable params).
    # Protocol: LR 1e-5, WD 1e-4, 25 epochs, early stop patience 5 on val AUC.
    # ------------------------------------------------------------------
    "sf-finetune": [
        Config(
            run_name="sf-finetune",
            run_dir="runs/sf-finetune",
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            checkpoint="weights/GenD_CLIP_L_14.ckpt",
            freeze_feature_extractor=True,
            unfreeze_layers=_LAYERNORM_CLIP,
            trn_files=_txts("train"),
            val_files=_txts("val"),
            tst_files=_txts("test"),
            loss=C.Loss(ce_labels=1.0),
            augmentations=C.Augmentations(
                random_horizontal_flip=0.5,
                color_jitter_brightness=0.1,
                color_jitter_contrast=0.1,
                jpeg_quality=[40, 100],
            ),
            optimizer="AdamW",
            batch_size=512,
            mini_batch_size=64,
            lr=1e-5,
            min_lr=1e-6,
            weight_decay=1e-4,
            lr_scheduler="cosine",
            max_epochs=25,
            num_workers=8,
            devices=[0],
            precision="bf16-mixed",
            early_stopping_patience=5,
            monitor_metric="val/auroc_video",
            monitor_metric_mode="max",
            checkpoint_name="best_auroc",
            throw_exception_if_run_exists=False,
            remove_if_run_exists=True,
            wandb=False,
        )
    ],

    # ------------------------------------------------------------------
    # Training from scratch: no pretrained GenD weights, backbone init
    # from CLIP-L/14 pre-trained weights, head initialized randomly.
    # Protocol: WD 1e-4, 50 epochs, early stop patience 10 on val AUC.
    # ------------------------------------------------------------------
    "sf-scratch": [
        Config(
            run_name="sf-scratch",
            run_dir="runs/sf-scratch",
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            checkpoint=None,
            freeze_feature_extractor=False,
            trn_files=_txts_in_domain("train"),
            val_files=_txts_in_domain("val"),
            tst_files=_txts_out_domain("test"),
            loss=C.Loss(ce_labels=1.0, uniformity=0.5, alignment_labels=0.1),
            augmentations=C.Augmentations(
                random_horizontal_flip=0.5,
                color_jitter_brightness=0.1,
                color_jitter_contrast=0.1,
                jpeg_quality=[40, 100],
            ),
            optimizer="AdamW",
            batch_size=128,
            mini_batch_size=128,
            lr=3e-4,
            min_lr=1e-5,
            weight_decay=1e-4,
            lr_scheduler="cyclic",
            num_epochs_in_cycle=10,
            warmup_epochs=1,
            max_epochs=50,
            num_workers=8,
            devices=[0],
            precision="bf16-mixed",
            early_stopping_patience=10,
            monitor_metric="val/auroc_video",
            monitor_metric_mode="max",
            checkpoint_name="best_auroc",
            throw_exception_if_run_exists=False,
            wandb=False,
        )
    ],
}
