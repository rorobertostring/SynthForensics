# MM-Det — SynthForensics Evaluation

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection.

## Environment

Create the conda environment from `environment.yml`, then install LLaVA and the remaining packages. Run from the **MM-Det root directory**:

```bash
conda env create -f environment.yml      # add -n <name> to use a name other than MM_Det
conda activate MM_Det

pip install --no-deps -e LLaVA
pip install --no-deps nvidia-cudnn-cu13==9.19.0.56 nvidia-cublas==13.1.0.3 nvidia-cuda-nvrtc==13.0.88
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir   # optional, not used at inference
```

- `--no-deps` on LLaVA keeps the versions pinned in `environment.yml` (LLaVA's own pins would downgrade `bitsandbytes` and `scikit-learn`); `pip check` will report these two mismatches, which is expected.
- The cuDNN 9.19 step reproduces the environment used for the reported results. Check it with `python -c "import torch; print(torch.backends.cudnn.version())"`, which should print `91900`.
- `--no-cache-dir` avoids an `Invalid cross-device link` error from flash-attn when `/tmp` and the pip cache are on different filesystems.
- Do not use `pip install --force-reinstall`: it also reinstalls `torch` at its latest version.

## Overview

Five scripts in `preprocessing/` run the full SynthForensics evaluation pipeline:

| Script | Purpose |
|---|---|
| `preprocessing/run_preprocessing.py` | Run the full preprocessing pipeline (steps 1 → 2 → 3) in a single command |
| `preprocessing/01_preprocess_reals.py` | Step 1 — Extract frames + VQVAE reconstructions for real videos |
| `preprocessing/02_preprocess_fakes.py` | Step 2 — Extract frames + VQVAE reconstructions for fake videos |
| `preprocessing/03_cache_mmfr.py` | Step 3 — Cache Multi-Modal Forgery Representations (LLaVA features) |
| `preprocessing/run_inference.py` | Step 4 — Run MM-Det inference and compute metrics for all combinations |

Run them in order: preprocessing first (either via `run_preprocessing.py` or step-by-step), then inference.


## Prerequisites

**Split ID files** (already present in the MM-Det root):

```
SF-DFD_fake_ids_363.txt
SF-FF_fake_ids_140.txt
```

**Weights** — the following checkpoints must be present before running:

```
weights/vqvae/model.pt
weights/MM-Det/current_model.pth
```

**Dataset paths** — the pipeline expects the following layout, with `{compression}` in `raw`, `canonical`, `crf23`, `crf40`:

```
$DATASET_ROOT/
├── FF++/real_{compression}_test_set/*.mp4
├── DFD/real_{compression}_test_set/*.mp4
└── SynthForensics/
    ├── T2V/videos/{compression}/{generator}/*_t2v.mp4
    └── I2V/videos/{compression}/{generator}/*_i2v.mp4
```

Set `DATASET_ROOT` at the top of `preprocessing/pipeline_config.py` before running:

```python
DATASET_ROOT = "/path/to/SynthForensics/Data"
SF_T2V_ROOT  = os.path.join(DATASET_ROOT, "SynthForensics", "T2V", "videos")
SF_I2V_ROOT  = os.path.join(DATASET_ROOT, "SynthForensics", "I2V", "videos")

REAL_DATASET_TEMPLATES = {
    "FF":  os.path.join(DATASET_ROOT, "FF++",  "real_{compression}_test_set"),
    "DFD": os.path.join(DATASET_ROOT, "DFD",   "real_{compression}_test_set"),
}
```

Generators are discovered automatically from the dataset directory structure — no manual list needed.


## Step 1–3 — Preprocessing (all-in-one)

Run from the **MM-Det root directory**:

```bash
python preprocessing/run_preprocessing.py
```

This loads the VQVAE once for steps 1 and 2, frees it, then loads LLaVA once for step 3. Each sub-step skips already-completed work automatically (`.done` sentinel files).

Alternatively, run each step individually:

### Step 1 — Preprocess real videos

```bash
python preprocessing/01_preprocess_reals.py
```

Extracts all frames from real videos and generates VQVAE reconstructions for every `(split, compression)` combination.

### Step 2 — Preprocess fake videos

```bash
python preprocessing/02_preprocess_fakes.py
```

Same as step 1 but for fake videos. Generators are discovered dynamically; videos are filtered against the split ID txt files.

### Step 3 — Cache MMFR

```bash
python preprocessing/03_cache_mmfr.py
```

Loads the fine-tuned LLaVA model once and generates one `mm_representation.pth` per `(split, compression, label)`. Requires steps 1 and 2 to be complete.

### Output after preprocessing

```
preprocessing/
├── recons/
│   ├── reals/{split}/{compression}/0_real/
│   │   ├── original/   {video_id}_{frame_idx}.jpg
│   │   └── recons/     {video_id}_{frame_idx}.jpg
│   └── fakes/{t2v,i2v}/{generator}/{split}/{compression}/1_fake/
│       ├── original/   {video_id}_{frame_idx}.jpg
│       └── recons/     {video_id}_{frame_idx}.jpg
└── mmfr/
    ├── reals/{split}/{compression}/0_real/mm_representation.pth
    └── fakes/{t2v,i2v}/{generator}/{split}/{compression}/1_fake/mm_representation.pth
```


## Step 4 — Inference

Run from the **MM-Det root directory**:

```bash
python preprocessing/run_inference.py
```

Loads MM-Det once, then iterates over every `(vtype, generator, split, compression)` combination. For each it builds a combined real+fake dataset, looks up cached MMFR representations, runs the model in sliding-window mode (`window_size=10`), and computes metrics.

Combinations whose JSON already exists are skipped automatically (safe to resume).

### Output

```
preprocessing/results/{vtype}/{compression}/{split}/{generator}.json
```

Each JSON contains:

```json
{
  "vtype": "t2v",
  "generator": "cogvideox",
  "split": "FF",
  "compression": "raw",
  "metrics": {
    "AUROC": ...,
    "AP": ...,
    "EER": ...,
    "ACC": ...,
    "n_real": ...,
    "n_fake": ...
  }
}
```

### Metrics

| Metric | Description |
|---|---|
| `AUROC` | Area Under the ROC Curve |
| `AP` | Average Precision |
| `EER` | Equal Error Rate |
| `ACC` | Accuracy at threshold 0.5 |
| `n_real` / `n_fake` | Number of real and fake samples evaluated |
