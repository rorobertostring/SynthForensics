#!/usr/bin/env python3
"""
01_prepare_data.py — Prepare survey export for downstream analysis.

Reads the raw survey export, validates structure, derives the benchmark
name / SF modality / SF generator columns from video paths, and writes a
parquet file consumed by subsequent scripts.

Run:
    python 01_prepare_data.py

Input:
    survey_export_2026-05-01.csv   (13 columns, 13575 rows)

Output:
    prepared.parquet               (17 columns, 13575 rows)

Notes:
    Pristine vs benchmark pair is derived from `video_b_category` and
    exposed as `is_pristine_pair`.

    The Q3 (`could_be_fake`) response category 'Neither' is encoded as
    NaN in `chosen_video` and `chosen_category` by the survey UI. It
    represents a valid 4th response option, not missing data, and is
    decoded into (is_A_fake=0, is_B_fake=0) by 04_q3_detection.py. All
    Q1/Q2 NaNs are missing responses from partial sessions.
"""

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
INPUT_CSV = HERE / "survey_export_2026-05-01.csv"
OUTPUT_PARQUET = HERE / "prepared.parquet"

EXPECTED_BENCHMARKS = {
    "GenVideo", "GenVidBench", "AIGVDBench", "DVF", "GVF",
    "GVD", "LOKI", "AEGIS", "DeepAction",
}
EXPECTED_REAL = {"FF++", "DFD"}
EXPECTED_MODALITIES = {"t2v", "i2v"}
EXPECTED_SF_GENERATORS = {
    "Wan2.1", "CogVideoX", "SkyReels-V2", "Self-Forcing",
    "MAGI-1", "LTX-2", "Helios", "daVinci-MagiHuman",
}

# Maps from raw path tokens to canonical paper names.
BENCHMARK_NAME_MAP = {
    "GenVideo": "GenVideo",
    "GenVidBench": "GenVidBench",
    "AIGVDBench": "AIGVDBench",
    "DVF": "DVF",
    "DeCoF": "GVF",
    "GVD": "GVD",
    "LOKI": "LOKI",
    "AEGIS": "AEGIS",
    "deepaction_v1": "DeepAction",
    "ff_plusplus": "FF++",
    "dfd": "DFD",
}

SF_GENERATOR_NAME_MAP = {
    "wan2-1": "Wan2.1",
    "cogvideox": "CogVideoX",
    "skyreels-v2": "SkyReels-V2",
    "self-forcing": "Self-Forcing",
    "magi-1": "MAGI-1",
    "ltx2-3": "LTX-2",
    "helios": "Helios",
    "daVinci": "daVinci-MagiHuman",
}

# Regex for path parsing. Tolerant to leading directory prefix and case.
RE_BENCHMARK = re.compile(r"category_b/([^/]+)/")
RE_REAL = re.compile(r"category_c/([^/]+)/")
RE_SF_MODALITY = re.compile(r"synthforensics_(t2v|i2v)", re.IGNORECASE)
RE_SF_GENERATOR = re.compile(r"synthforensics_[ti]2v/raw/([^/]+)/", re.IGNORECASE)


def derive_benchmark_name(row: pd.Series) -> str:
    """Extract benchmark identifier from video_b_path and canonicalise.

    Competitor paths: `category_b/<RAW>/...`.
    Pristine paths: `category_c/<RAW>/...` with RAW in {ff_plusplus, dfd}.
    """
    path = row["video_b_path"]
    regex = RE_REAL if row["is_pristine_pair"] else RE_BENCHMARK
    m = regex.search(path)
    if not m:
        kind = "pristine" if row["is_pristine_pair"] else "competitor"
        raise ValueError(f"Cannot parse {kind} benchmark from path: {path}")
    raw = m.group(1)
    if raw not in BENCHMARK_NAME_MAP:
        raise ValueError(f"Unknown benchmark token '{raw}' in path: {path}")
    return BENCHMARK_NAME_MAP[raw]


def derive_sf_modality(path: str) -> str:
    m = RE_SF_MODALITY.search(path)
    if not m:
        raise ValueError(f"Cannot find SF modality in path: {path}")
    return m.group(1).lower()


def derive_sf_generator(path: str) -> str:
    m = RE_SF_GENERATOR.search(path)
    if not m:
        raise ValueError(f"Cannot find SF generator in path: {path}")
    raw = m.group(1)
    if raw not in SF_GENERATOR_NAME_MAP:
        raise ValueError(f"Unknown SF generator token '{raw}' in path: {path}")
    return SF_GENERATOR_NAME_MAP[raw]


def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {INPUT_CSV.name}")

    # Validation 1: video_a_category is always 'Synthforensics'.
    bad_a = df.loc[df["video_a_category"] != "Synthforensics"]
    if len(bad_a):
        sys.exit(f"video_a_category not always Synthforensics; offending rows: {len(bad_a)}")

    # Validation 2: video_b_category is in the expected set.
    expected_b = {"Contestants", "Real (FF++/DFD)"}
    bad_b = df.loc[~df["video_b_category"].isin(expected_b)]
    if len(bad_b):
        unexpected = df["video_b_category"].unique().tolist()
        sys.exit(f"video_b_category contains unexpected values: {unexpected}")

    # Derive is_pristine_pair.
    df["is_pristine_pair"] = df["video_b_category"] == "Real (FF++/DFD)"

    # Derive benchmark_name (handles both competitor and pristine pairs).
    df["benchmark_name"] = df.apply(derive_benchmark_name, axis=1)

    # Derive SF modality and generator from video_a_path.
    df["sf_modality"] = df["video_a_path"].apply(derive_sf_modality)
    df["sf_generator"] = df["video_a_path"].apply(derive_sf_generator)

    # Sanity prints.
    print()
    print("=== Derived columns — unique values ===")
    print(f"is_pristine_pair: {df['is_pristine_pair'].value_counts().to_dict()}")
    print()
    print("benchmark_name (sorted):")
    for name, n in sorted(df["benchmark_name"].value_counts().items()):
        flag = "✓" if (name in EXPECTED_BENCHMARKS or name in EXPECTED_REAL) else "✗ UNEXPECTED"
        print(f"  {name:20s} {n:6,d}  {flag}")
    print()
    print(f"sf_modality: {df['sf_modality'].value_counts().to_dict()}")
    unexpected_mod = set(df["sf_modality"].unique()) - EXPECTED_MODALITIES
    if unexpected_mod:
        print(f"  ✗ UNEXPECTED: {unexpected_mod}")
    print()
    print("sf_generator (sorted):")
    for name, n in sorted(df["sf_generator"].value_counts().items()):
        flag = "✓" if name in EXPECTED_SF_GENERATORS else "✗ UNEXPECTED"
        print(f"  {name:25s} {n:6,d}  {flag}")

    # Cross-validation: every pristine pair must have a real baseline name; every
    # benchmark pair must NOT have FF++/DFD.
    pristine = df.loc[df["is_pristine_pair"]]
    if not set(pristine["benchmark_name"].unique()).issubset(EXPECTED_REAL):
        sys.exit(f"Pristine pairs contain non-real benchmark_name: "
                 f"{pristine['benchmark_name'].unique()}")
    competitor = df.loc[~df["is_pristine_pair"]]
    if EXPECTED_REAL & set(competitor["benchmark_name"].unique()):
        sys.exit("Competitor pairs contain FF++/DFD benchmark_name; check path parser.")

    # Save.
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print()
    print(f"Saved {len(df):,} rows × {df.shape[1]} columns to {OUTPUT_PARQUET.name}")


if __name__ == "__main__":
    main()
