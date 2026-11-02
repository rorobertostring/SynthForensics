"""
Shared configuration for the MM-Det preprocessing and inference pipeline.
Generators are discovered dynamically from the dataset directory structure.
"""
import os
import sys

# MM-Det root = parent of this preprocessing/ directory
MMDET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MMDET_ROOT)

# ── Source data ──────────────────────────────────────────────────────────────
DATASET_ROOT = "/path/to/SynthForensics/Data"
SF_T2V_ROOT  = os.path.join(DATASET_ROOT, "SynthForensics", "T2V", "videos")
SF_I2V_ROOT  = os.path.join(DATASET_ROOT, "SynthForensics", "I2V", "videos")

# ── Output roots ─────────────────────────────────────────────────────────────
PREPROC_ROOT = os.path.join(MMDET_ROOT, "preprocessing")
RECONS_ROOT  = os.path.join(PREPROC_ROOT, "recons")
MMFR_ROOT    = os.path.join(PREPROC_ROOT, "mmfr")
RESULTS_ROOT = os.path.join(PREPROC_ROOT, "results")

# ── Dataset enumerations ──────────────────────────────────────────────────────
COMPRESSIONS = ["raw", "canonical", "crf23", "crf40"]
SPLITS       = ["FF", "DFD"]

# Real dataset path templates (format with compression=...)
REAL_DATASET_TEMPLATES = {
    "FF":  os.path.join(DATASET_ROOT, "FF++",  "real_{compression}_test_set"),
    "DFD": os.path.join(DATASET_ROOT, "DFD",   "real_{compression}_test_set"),
}

# Split ID files
SPLIT_ID_FILES = {
    "FF":  os.path.join(MMDET_ROOT, "SF-FF_fake_ids_140.txt"),
    "DFD": os.path.join(MMDET_ROOT, "SF-DFD_fake_ids_363.txt"),
}

# ── Model checkpoints ─────────────────────────────────────────────────────────
VQVAE_CKPT = os.path.join(MMDET_ROOT, "weights", "vqvae", "model.pt")
MMDET_CKPT = os.path.join(MMDET_ROOT, "weights", "MM-Det", "current_model.pth")
LMM_CKPT   = "sparklexfantasy/llava-7b-1.5-rfrd"

# ── Dynamic generator discovery ───────────────────────────────────────────────

def discover_generators(vtype):
    """
    Discover available generators by listing subdirectories under the first
    available compression folder. Consistent across compressions by design.
    Returns a sorted list of generator names.
    """
    root = SF_T2V_ROOT if vtype == "t2v" else SF_I2V_ROOT
    for compression in COMPRESSIONS:
        comp_dir = os.path.join(root, compression)
        if os.path.isdir(comp_dir):
            return sorted(
                d for d in os.listdir(comp_dir)
                if os.path.isdir(os.path.join(comp_dir, d))
            )
    return []

# ── Derived path helpers ──────────────────────────────────────────────────────

def reals_recons_dir(split, compression):
    return os.path.join(RECONS_ROOT, "reals", split, compression, "0_real")

def fakes_recons_dir(vtype, generator, split, compression):
    return os.path.join(RECONS_ROOT, "fakes", vtype, generator, split, compression, "1_fake")

def reals_mmfr_path(split, compression):
    return os.path.join(MMFR_ROOT, "reals", split, compression, "0_real", "mm_representation.pth")

def fakes_mmfr_path(vtype, generator, split, compression):
    return os.path.join(MMFR_ROOT, "fakes", vtype, generator, split, compression, "1_fake", "mm_representation.pth")

def results_path(vtype, generator, split, compression):
    return os.path.join(RESULTS_ROOT, vtype, compression, split, f"{generator}.json")

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_split_ids(split):
    """Load video IDs from a split txt file (skip comment lines)."""
    ids = []
    with open(SPLIT_ID_FILES[split]) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids

def is_done(directory):
    return os.path.exists(os.path.join(directory, ".done"))

def mark_done(directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, ".done"), "w") as f:
        f.write("done\n")

def all_combinations():
    """Yield (vtype, generator, split, compression) for every discovered combination."""
    for vtype in ("t2v", "i2v"):
        for generator in discover_generators(vtype):
            for split in SPLITS:
                for compression in COMPRESSIONS:
                    yield vtype, generator, split, compression
