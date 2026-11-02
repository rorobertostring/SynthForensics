#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline script: for each (dataset × compression × generator) combination,
runs preprocess_synthforensics.py → eval_synthforensics.py in sequence.

Supports the same filters as the two child scripts to limit execution to a
subset. Without filters the behaviour is identical to running both scripts
separately on everything.

The log is saved to {output_root}/pipeline_{mode}.log (append).

Usage — full run:
    CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \\
        --output_root preprocessed \\
        --data_root /path/to/Data

Usage — single combination:
    CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \\
        --dataset SF-FF --compression crf23 --generator cogvideox \\
        --output_root preprocessed \\
        --data_root /path/to/Data

Usage — i2v mode:
    CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \\
        --mode i2v \\
        --fake_videos_subdir SynthForensics/I2V/videos \\
        --real_preprocessed_root preprocessed \\
        --output_root preprocessed_i2v \\
        --data_root /path/to/Data

Use --preprocess_only / --eval_only to run only one of the two phases.
"""

import os
import sys
import argparse
import subprocess
import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DATASETS     = ["SF-FF", "SF-DFD"]
ALL_COMPRESSIONS = ["crf23", "crf40", "raw", "canonical"]

MODE_DEFAULTS = {
    "t2v": {
        "fake_videos_subdir":    "SynthForensics/T2V/videos",
        "preprocessed_root_default": "preprocessed",
        "results_root_default":      "results",
    },
    "i2v": {
        "fake_videos_subdir":    "SynthForensics/I2V/videos",
        "preprocessed_root_default": "preprocessed_i2v",
        "results_root_default":      "results_i2v",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_generators(fake_base: str) -> list:
    """Return sorted list of generator directories in fake_base."""
    if not os.path.isdir(fake_base):
        return []
    return sorted(
        d for d in os.listdir(fake_base)
        if os.path.isdir(os.path.join(fake_base, d))
    )


def log(msg: str, log_path: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def run_subprocess(cmd: list, log_path: str, label: str) -> int:
    """Run cmd, stream output to stdout and log file, return exit code."""
    log(f"START {label}", log_path)
    log(f"  CMD: {' '.join(cmd)}", log_path)
    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    log(f"END   {label} → {status}", log_path)
    return proc.returncode


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("FakeSTormer full pipeline: preprocess + eval")

    # ── mode ─────────────────────────────────────────────────────────────────
    p.add_argument("--mode", default="t2v", choices=["t2v", "i2v"],
                   help="Generation mode: t2v (default) or i2v")

    # ── filters (all optional; omit = run everything) ─────────────────────────
    p.add_argument("--dataset",     "-d", default=None,
                   choices=ALL_DATASETS,
                   help="Process only this dataset (default: all)")
    p.add_argument("--compression", "-c", default=None,
                   choices=ALL_COMPRESSIONS,
                   help="Process only this compression (default: all)")
    p.add_argument("--generator",   "-g", default=None,
                   help="Process only this generator (default: auto-discover all)")

    # ── preprocess args ───────────────────────────────────────────────────────
    p.add_argument("--output_root", "-o", default=None,
                   help="Root for extracted frames "
                        "(default: preprocessed for t2v, preprocessed_i2v for i2v)")
    p.add_argument("--data_root",
                   default="/path/to/SynthForensics/Data",
                   help="Root of the SynthForensics Data directory")
    p.add_argument("--fake_videos_subdir", default=None,
                   help="Subdirectory under data_root for fake videos "
                        "(default: SynthForensics/T2V/videos for t2v, "
                        "SynthForensics/I2V/videos for i2v)")
    p.add_argument("--real_preprocessed_root", default=None,
                   help="(i2v) Path to t2v real frames to symlink instead of re-extracting")
    p.add_argument("--num_frames", default=32, type=int,
                   help="Frames to extract per video (default: 32)")
    p.add_argument("--gpu_preprocess", default=0, type=int,
                   help="GPU index for RetinaFace (default: 0)")

    # ── eval args ─────────────────────────────────────────────────────────────
    p.add_argument("--results_root", "-r", default=None,
                   help="Root for result JSONs "
                        "(default: results for t2v, results_i2v for i2v)")
    p.add_argument("--base_cfg",
                   default="configs/temporal/FakeSFormer_synthforensics_eval.yaml",
                   help="Base YAML config for eval")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip eval combos whose JSON result already exists")
    p.add_argument("--batch_size", default=8, type=int,
                   help="Batch size for eval (default: 8)")
    p.add_argument("--num_workers", default=8, type=int,
                   help="DataLoader workers for eval (default: 8)")

    # ── pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--preprocess_only", action="store_true",
                   help="Run only preprocessing, skip eval")
    p.add_argument("--eval_only", action="store_true",
                   help="Run only eval, skip preprocessing")

    args = p.parse_args()

    # Fill mode-based defaults
    md = MODE_DEFAULTS[args.mode]
    if args.fake_videos_subdir is None:
        args.fake_videos_subdir = md["fake_videos_subdir"]
    if args.output_root is None:
        args.output_root = md["preprocessed_root_default"]
    if args.results_root is None:
        args.results_root = md["results_root_default"]

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    datasets     = [args.dataset]     if args.dataset     else ALL_DATASETS
    compressions = [args.compression] if args.compression else ALL_COMPRESSIONS

    # Log file in the output root
    os.makedirs(args.output_root, exist_ok=True)
    log_path = os.path.join(args.output_root, f"pipeline_{args.mode}.log")

    log(f"Pipeline start — mode={args.mode}  datasets={datasets}  "
        f"compressions={compressions}  generator={args.generator}", log_path)

    python = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_ok = 0
    run_fail = 0

    for sf_dataset in datasets:
        for compression in compressions:

            # Discover generators for this (dataset, compression) combo
            fake_base = os.path.join(
                args.data_root, args.fake_videos_subdir, compression
            )
            if args.generator:
                generators = [args.generator]
            else:
                generators = discover_generators(fake_base)
                if not generators:
                    log(f"[skip] no generators found in {fake_base}", log_path)
                    continue

            log(f"\n{'='*60}", log_path)
            log(f"{sf_dataset} / {compression} — generators: {generators}", log_path)

            for generator in generators:
                label = f"{sf_dataset}/{compression}/{generator}"

                # ── PREPROCESS ────────────────────────────────────────────
                if not args.eval_only:
                    preprocess_cmd = [
                        python, "-u", os.path.join(script_dir, "preprocess_synthforensics.py"),
                        "--mode",              args.mode,
                        "--dataset",           sf_dataset,
                        "--compression",       compression,
                        "--generator",         generator,
                        "--output_root",       args.output_root,
                        "--data_root",         args.data_root,
                        "--fake_videos_subdir", args.fake_videos_subdir,
                        "--num_frames",        str(args.num_frames),
                        "--gpu",               str(args.gpu_preprocess),
                    ]
                    if args.real_preprocessed_root:
                        preprocess_cmd += ["--real_preprocessed_root", args.real_preprocessed_root]

                    rc = run_subprocess(preprocess_cmd, log_path, f"preprocess:{label}")
                    if rc != 0:
                        run_fail += 1
                        log(f"[warn] preprocess failed for {label}, continuing...", log_path)
                        continue

                # ── EVAL ──────────────────────────────────────────────────
                if not args.preprocess_only:
                    eval_cmd = [
                        python, "-u", os.path.join(script_dir, "eval_synthforensics.py"),
                        "--mode",              args.mode,
                        "--datasets",          sf_dataset,
                        "--compressions",      compression,
                        "--generators",        generator,
                        "--preprocessed_root", args.output_root,
                        "--results_root",      args.results_root,
                        "--base_cfg",          args.base_cfg,
                        "--batch_size",        str(args.batch_size),
                        "--num_workers",       str(args.num_workers),
                    ]
                    if args.skip_existing:
                        eval_cmd.append("--skip_existing")

                    rc = run_subprocess(eval_cmd, log_path, f"eval:{label}")
                    if rc != 0:
                        run_fail += 1
                        log(f"[warn] eval failed for {label}", log_path)
                        continue

                run_ok += 1

    log(f"\nPipeline done — ok={run_ok}  failed={run_fail}", log_path)
    log(f"Log: {os.path.abspath(log_path)}", log_path)


if __name__ == "__main__":
    main()
