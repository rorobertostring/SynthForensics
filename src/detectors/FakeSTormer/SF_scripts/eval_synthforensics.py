#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation of FakeSTormer on the SynthForensics dataset.

For each (sf_dataset × compression × generator) combination it:
  1. Patches the base YAML config with the correct ROOT and FAKETYPE.
  2. Runs scripts/test.py via subprocess.
  3. Saves per-combination JSON under results_root/:

      {results_root}/{sf_dataset}/{compression}/{generator}.json

  identical to the structure used by other detectors in this repo.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_synthforensics.py \\
        --preprocessed_root preprocessed \\
        --results_root results \\
        --datasets SF-FF SF-DFD \\
        --compressions crf23 crf40 raw canonical \\
        --gpu 0

The script auto-discovers generators from each preprocessed directory, so
no code change is needed when a new generator is added.
"""

import os
import sys
import re
import json
import copy
import argparse
import subprocess
import tempfile

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover_generators(videos_dir: str) -> list:
    """Return sorted list of generator directories (everything except 'real')."""
    if not os.path.isdir(videos_dir):
        return []
    return sorted(
        d for d in os.listdir(videos_dir)
        if os.path.isdir(os.path.join(videos_dir, d)) and d != "real"
    )


def parse_metrics_from_output(text: str) -> dict:
    """
    Extract metrics from test.py log line.
    APR=True  → ACC, AUC, AP, AR, mF1, best_thr
    APR=False → ACC, BACC, AUC, P, R, S, F1, EER, best_thr
    Returns a dict, or empty dict if nothing matched.
    """
    # APR=True branch
    m = re.search(
        r"Current ACC, AUC, AP, AR, mF1, THR.*?---.*?"
        r"(\d+\.\d+).*?--\s*(\d+\.\d+).*?--\s*(\d+\.\d+)"
        r".*?--\s*(\d+\.\d+).*?--\s*(\d+\.\d+).*?--\s*(\S+)",
        text, re.DOTALL
    )
    if m:
        return dict(
            acc=float(m.group(1)), auc=float(m.group(2)),
            ap=float(m.group(3)),  ar=float(m.group(4)),
            mf1=float(m.group(5)), best_thr=float(m.group(6))
        )
    # APR=False branch
    m2 = re.search(
        r"Current ACC, BACC, AUC, P, R, S, F1, EER, THR.*?---.*?"
        r"(\d+\.\d+).*?--\s*(\d+\.\d+).*?--\s*(\d+\.\d+)"
        r".*?--\s*(\d+\.\d+).*?--\s*(\d+\.\d+).*?--\s*(\d+\.\d+)"
        r".*?--\s*(\d+\.\d+).*?--\s*(\d+\.\d+).*?--\s*(\S+)",
        text, re.DOTALL
    )
    if m2:
        return dict(
            acc=float(m2.group(1)),  bacc=float(m2.group(2)),
            auc=float(m2.group(3)),  p=float(m2.group(4)),
            r=float(m2.group(5)),    s=float(m2.group(6)),
            f1=float(m2.group(7)),   eer=float(m2.group(8)),
            best_thr=float(m2.group(9))
        )
    return {}


def save_result(result: dict, results_root: str, sf_dataset: str,
                compression: str, generator: str):
    """Save metrics to {results_root}/{sf_dataset}/{compression}/{generator}.json"""
    out_dir = os.path.join(results_root, sf_dataset, compression)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{generator}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("FakeSTormer batch eval on SynthForensics")

    # ── mode ─────────────────────────────────────────────────────────────────
    p.add_argument("--mode", default="t2v", choices=["t2v", "i2v"],
                   help="Generation mode: t2v (default) or i2v. "
                        "Sets default preprocessed_root and results_root if not overridden.")

    # ── paths (None = filled by mode defaults after parsing) ─────────────────
    p.add_argument("--preprocessed_root", "-p", default=None,
                   help="Root of preprocessed face crops "
                        "(default: preprocessed for t2v, preprocessed_i2v for i2v)")
    p.add_argument("--results_root", "-r", default=None,
                   help="Root folder for per-combination JSON results "
                        "(default: results for t2v, results_i2v for i2v)")
    p.add_argument("--base_cfg", "-c",
                   default="configs/temporal/FakeSFormer_synthforensics_eval.yaml",
                   help="Base YAML config to patch for each run")

    # ── filters ──────────────────────────────────────────────────────────────
    p.add_argument("--datasets", nargs="+",
                   default=["SF-FF", "SF-DFD"],
                   help="SF dataset(s) to evaluate")
    p.add_argument("--compressions", nargs="+",
                   default=["crf23", "crf40", "raw", "canonical"],
                   help="Compression(s) to evaluate")
    p.add_argument("--generators", nargs="*", default=None,
                   help="Generators to evaluate (default: auto-discover all)")
    p.add_argument("--combined", action="store_true",
                   help="Also run real-vs-all-generators and save as 'ALL.json'")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a combo if its JSON result already exists")

    # ── execution ────────────────────────────────────────────────────────────
    p.add_argument("--gpu", default=0, type=int,
                   help="[DEPRECATED] ignored — set CUDA_VISIBLE_DEVICES externally")
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--num_workers", default=8, type=int)

    args = p.parse_args()

    # Apply mode-based defaults for paths not explicitly set
    mode_defaults = {
        "t2v": {"preprocessed_root": "preprocessed",     "results_root": "results"},
        "i2v": {"preprocessed_root": "preprocessed_i2v", "results_root": "results_i2v"},
    }
    for key, default in mode_defaults[args.mode].items():
        if getattr(args, key) is None:
            setattr(args, key, default)

    return args


def run_eval(base_cfg: dict, root: str, faketype: list,
             compression: str, batch_size: int,
             num_workers: int) -> tuple:
    """Patch config, run scripts/test.py, stream output, return (metrics_dict, raw_output)."""
    cfg = copy.deepcopy(base_cfg)

    cfg["DATASET"]["COMPRESSION"] = compression
    cfg["DATASET"]["NUM_WORKERS"] = num_workers
    for split in ("TRAIN", "VAL", "TEST"):
        cfg["DATASET"]["DATA"][split]["ROOT"] = root
        cfg["DATASET"]["DATA"][split]["NAME"] = "SynthForensics"
    cfg["DATASET"]["DATA"]["TEST"]["FAKETYPE"] = faketype
    cfg["DATASET"]["DATA"]["TEST"]["FROM_FILE"] = False
    cfg["TRAIN"]["batch_size"] = batch_size

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="sf_eval_"
    ) as tf:
        yaml.dump(cfg, tf, default_flow_style=False, allow_unicode=True)
        tmp_cfg_path = tf.name

    # FakeSTormer root is the parent of SF_scripts/
    fakestormer_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-u", os.path.join(fakestormer_root, "scripts", "test.py"),
           "--cfg", tmp_cfg_path]
    print(f"  CMD: python scripts/test.py --cfg <tmp>  "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")

    lines = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            cwd=fakestormer_root,
        )
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            sys.stdout.flush()
            lines.append(line)
        proc.wait()
        output = "".join(lines)
        metrics = parse_metrics_from_output(output)
        if proc.returncode != 0:
            print(f"  [WARN] test.py exited with code {proc.returncode}")
    except Exception as e:
        output = str(e)
        metrics = {}
        print(f"  [ERROR] subprocess failed: {e}")
    finally:
        os.unlink(tmp_cfg_path)

    return metrics, output


def main():
    args = parse_args()
    base_cfg = load_yaml(args.base_cfg)

    run_count = 0
    fail_count = 0
    skip_count = 0

    for sf_dataset in args.datasets:
        for compression in args.compressions:
            videos_dir = os.path.join(
                args.preprocessed_root, sf_dataset, compression, "test", "videos"
            )
            if not os.path.isdir(videos_dir):
                print(f"[skip] not found: {videos_dir}")
                continue

            root = os.path.join(args.preprocessed_root, sf_dataset, compression)
            generators = (
                args.generators if args.generators is not None
                else discover_generators(videos_dir)
            )

            print(f"\n{'='*64}")
            print(f"{sf_dataset} / {compression}  |  generators: {generators}")

            eval_targets = [(gen, ["real", gen]) for gen in generators
                            if os.path.isdir(os.path.join(videos_dir, gen))]
            if args.combined and generators:
                eval_targets.append(("ALL", ["real"] + generators))

            for gen_name, faketype in eval_targets:
                out_path = os.path.join(
                    args.results_root, sf_dataset, compression, f"{gen_name}.json"
                )

                if args.skip_existing and os.path.exists(out_path):
                    print(f"  [skip existing] {sf_dataset}/{compression}/{gen_name}")
                    skip_count += 1
                    continue

                print(f"\n  → {sf_dataset}/{compression}/{gen_name}")

                metrics, _ = run_eval(
                    base_cfg, root, faketype, compression,
                    args.batch_size, args.num_workers
                )
                run_count += 1

                if not metrics:
                    fail_count += 1
                    print(f"  [WARN] no metrics parsed")
                    result_to_save = {"error": "metrics not parsed"}
                else:
                    auc = metrics.get("auc", float("nan"))
                    ap  = metrics.get("ap",  float("nan"))
                    mf1 = metrics.get("mf1", float("nan"))
                    print(f"  AUC={auc:.2f}  AP={ap:.2f}  mF1={mf1:.2f}")
                    result_to_save = metrics

                saved = save_result(
                    result_to_save, args.results_root,
                    sf_dataset, compression, gen_name
                )
                print(f"  Saved → {saved}")

    print(f"\n{'='*64}")
    print(f"Done.  runs={run_count}  failures={fail_count}  skipped={skip_count}")
    print(f"Results tree: {os.path.abspath(args.results_root)}/")

    # Summary table
    print(f"\n{'Combination':<50} {'AUC':>7} {'AP':>7} {'mF1':>7}")
    print("-" * 70)
    for sf_dataset in args.datasets:
        for compression in args.compressions:
            result_dir = os.path.join(args.results_root, sf_dataset, compression)
            if not os.path.isdir(result_dir):
                continue
            for fname in sorted(os.listdir(result_dir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(result_dir, fname)) as f:
                    m = json.load(f)
                key = f"{sf_dataset}/{compression}/{fname[:-5]}"
                auc = f"{m['auc']:.2f}" if "auc" in m else "n/a"
                ap  = f"{m['ap']:.2f}"  if "ap"  in m else "n/a"
                mf1 = f"{m['mf1']:.2f}" if "mf1" in m else "n/a"
                print(f"{key:<50} {auc:>7} {ap:>7} {mf1:>7}")


if __name__ == "__main__":
    main()
