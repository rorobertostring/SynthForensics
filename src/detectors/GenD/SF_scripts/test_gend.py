#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run GenD inference on pre-built txt file pairs (real + fake).

The txt files are generated automatically by extract_frames_gend.py.
Labels are derived from the path structure automatically:
  - path[-3] contains "real" → label 0
  - otherwise              → label 1

Usage:
  python SF_scripts/test_gend.py \\
      --real_txt config/datasets/gend/SF-FF/canonical/real.txt \\
      --fake_txt config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt \\
      --run_name SF-FF_canonical_cogvideox

  # Multiple fake sources at once
  python SF_scripts/test_gend.py \\
      --real_txt config/datasets/gend/SF-FF/canonical/real.txt \\
      --fake_txt config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt \\
                 config/datasets/gend/SF-FF/canonical/fake_wan2-1.txt \\
      --run_name SF-FF_canonical_multi

  # Different checkpoint
  python SF_scripts/test_gend.py \\
      --real_txt config/datasets/gend/SF-FF/canonical/real.txt \\
      --fake_txt config/datasets/gend/SF-FF/canonical/fake_cogvideox.txt \\
      --run_name SF-FF_canonical_cogvideox_PE \\
      --checkpoint yermandy/GenD_PE_L
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run import main
from src import config as C
from src.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="GenD inference on real/fake txt file pairs"
    )
    parser.add_argument(
        "--real_txt", required=True, nargs="+",
        help="One or more .txt files listing real frame paths",
    )
    parser.add_argument(
        "--fake_txt", required=True, nargs="+",
        help="One or more .txt files listing fake frame paths",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Name of the run (used as output folder under --run_dir)",
    )
    parser.add_argument(
        "--checkpoint", default="yermandy/GenD_CLIP_L_14",
        help="HF repo id or local .ckpt path (default: yermandy/GenD_CLIP_L_14)",
    )
    parser.add_argument(
        "--run_dir", default="runs/synthforensics",
        help="Root directory for run outputs (default: runs/synthforensics)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
    )
    parser.add_argument(
        "--devices", type=int, nargs="+", default=[0],
        help="GPU device ids (default: [0])",
    )
    parser.add_argument(
        "--num_workers", type=int, default=12,
    )
    parser.add_argument(
        "--precision", default="bf16-mixed",
    )
    parser.add_argument(
        "--backbone", default=None,
        help="Backbone override for local .ckpt checkpoints (e.g. CLIP_L_14). "
             "Ignored when using a HuggingFace checkpoint.",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()

    tst_files = args.real_txt + args.fake_txt

    for f in tst_files:
        if not os.path.exists(f):
            print(f"[ERROR] File not found: {f}")
            sys.exit(1)

    backbone_map = {
        "CLIP_L_14": C.Backbone.CLIP_L_14,
        "CLIP_B_32": C.Backbone.CLIP_B_32,
    }
    backbone = backbone_map[args.backbone] if args.backbone else None

    config_kwargs = dict(
        run_name=args.run_name,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        tst_files=tst_files,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        devices=args.devices,
        num_workers=args.num_workers,
        precision=args.precision,
        binary_labels=True,
        wandb=False,
        throw_exception_if_run_exists=False,
        remove_if_run_exists=True,
    )
    if backbone is not None:
        config_kwargs["backbone"] = backbone

    config = Config(**config_kwargs)

    main(config, train=False)


if __name__ == "__main__":
    main_cli()
