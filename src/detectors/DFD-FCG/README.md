# DFD-FCG — SynthForensics Usage

> Original code and documentation: [aiiu-lab/DFD-FCG](https://github.com/aiiu-lab/DFD-FCG). The original repository has no license, so its code is not redistributed here: this folder contains only our files, to be copied into a clone of the original repository.

## Setup

```bash
git clone https://github.com/aiiu-lab/DFD-FCG.git
cd DFD-FCG
git checkout 3c0b840
cp -r /path/to/SynthForensics/src/detectors/DFD-FCG/. .
```

For finetuning and training from scratch, also remove `verbose: false` from the `lr_scheduler` section of `configs/base.yaml`: the PyTorch version of the environment does not support it.

All the following commands are run from this `DFD-FCG` folder.

## Environment

```bash
conda env create -f environment.yml
conda activate dfd-fcg
```

## Weights

Download the pretrained model [checkpoint.zip](https://drive.google.com/file/d/1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk/view?usp=sharing) (from the original DFD-FCG repository), unzip it and put `setting.yaml` and `weights.ckpt` into `weights/zero_shot/`.

## Input format

Raw `.mp4` video files in a flat folder, one folder per split/compression/generator combination. The folders are built with the DeepfakeBench scripts `create_dfb_data.py` (fake) and `create_dfb_real_data.py` (real), see the [DeepfakeBench README](../DeepfakeBench/README.md#input-format):

- Fake: `/path/to/SynthForensics/src/detectors/DeepfakeBench/DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR>/`
- Real: `/path/to/SynthForensics/src/detectors/DeepfakeBench/DFB_Data/real/<DATASET>/real_<COMP>_test_set/`

where `<SPLIT>` is `SF-FF` or `SF-DFD`; `<DATASET>` is the matching real dataset (`FF++` or `DFD`); `_i2v` marks I2V videos; `<COMP>` is `raw` (no suffix for fake folders), `canonical`, `crf23`, or `crf40`.

No preprocessing needed: face detection and cropping are handled internally and cached in `already_processed/<cache_key>/`.

## Usage

**First run** — provide `--real-folder` to run real inference:

```bash
python batch_inference.py \
    /path/to/SynthForensics/src/detectors/DeepfakeBench/DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR> \
    --real-folder /path/to/SynthForensics/src/detectors/DeepfakeBench/DFB_Data/real/<DATASET>/real_<COMP>_test_set \
    --cache-keys  <output_name>
```

`<output_name>` is the name of the result file: results are saved to `results/<output_name>.csv`. It can include subfolders (e.g. `sf-ff/sfff_raw_helios` → `results/sf-ff/sfff_raw_helios.csv`); if omitted, the parent and fake folder names are used (e.g. `SF-FF_fake_test_set_crf23_cogvideox`).

**Subsequent runs** — reuse real scores from an existing CSV (skips real inference entirely):

```bash
python batch_inference.py \
    /path/to/SynthForensics/src/detectors/DeepfakeBench/DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR> \
    --real-csv   results/<previous_output_name>.csv \
    --cache-keys <output_name>
```

Multiple fake folders can be passed in a single run (model loaded once), with one output name per fake folder:

```bash
python batch_inference.py <FAKE1> <FAKE2> ... \
    --real-csv   results/<previous_output_name>.csv \
    --cache-keys <name1> <name2> ...
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `video_folder` | required | One or more fake video folders (positional) |
| `--real-folder` | — | Real video folder (runs real inference) |
| `--real-csv` | — | Existing CSV with real scores — skips real inference |
| `--cache-keys` | parent + fake folder name | Cache key(s) and output name(s), one per fake folder: results are saved to `results/<cache_key>.csv` |
| `--output-keys` | `--cache-keys` | Output name(s), if different from the cache keys |
| `--output-dir` | `results/` | Output folder for the result CSVs |
| `--real-cache-key` | real folder name | Cache key for the real folder |
| `--model-cfg` | `weights/zero_shot/setting.yaml` | Model config |
| `--model-ckpt` | `weights/zero_shot/weights.ckpt` | Model checkpoint |
| `--threshold` | `0.5` | Fake/real classification threshold |
| `--batch-size` | `30` | Clip batch size (~10 GB VRAM) |
| `--precision` | `16` | Inference precision |
| `--ext` | `mp4` | Extension of the input videos |
| `--force-preprocess` | off | Re-run preprocessing even if cache exists |

## Output

One CSV per fake folder at `results/<cache_key>.csv`. The file starts with the metric rows (`acc`, `auc`, `eer`, `ap`, `video_auc` (paper metric)), then per-video rows with score, true label, predicted label.
Preprocessing cache persists in `already_processed/<cache_key>/` and is reused on subsequent runs.

## Finetuning

Training uses the raw videos of all FF++ IDs: build them with the DeepfakeBench script `create_dfb_data.py --train_data` (see the [DeepfakeBench Finetuning section](../DeepfakeBench/README.md#finetuning)), which creates `DFB_Data/SF-FF_full[_i2v]/<GENERATOR>/`.

Face detection preprocessing must be run once per video source (landmark extraction + face cropping). Each source is a root folder in `datasets/` with a `videos/` symlink to the source videos:

- Real: `datasets/real_ffpp/videos` → `/path/to/FF++/original_sequences/youtube/raw/videos`
- Fake: `datasets/fake_<MODALITY>_<GENERATOR>/videos` → `/path/to/SynthForensics/src/detectors/DeepfakeBench/DFB_Data/SF-FF_full[_i2v]/<GENERATOR>`

where `<MODALITY>` is `t2v` or `i2v` (`_i2v` folder suffix for I2V). For each root:

```bash
echo y | python -m src.preprocess.fetch_landmark_bbox \
    --root-dir datasets/<ROOT> --video-dir videos --fdata-dir frame_data --glob-exp "*"

python -m src.preprocess.crop_main_face \
    --root-dir datasets/<ROOT> --video-dir videos --fdata-dir frame_data --glob-exp "*" \
    --crop-dir cropped --mean-face misc/20words_mean_face.npy
```

Then prepare the finetuning dataset and train:

```bash
python prepare_finetune_data.py --splits_dir /path/to/SynthForensics/splits   # run once — populates datasets/sfff_finetune/

python main.py \
    --config configs/base.yaml \
    --config configs/clip/L14/ffg.yaml \
    --config configs/finetune_base.yaml
```

Training logs to Weights & Biases (`configs/logger.yaml`): run `wandb login` first, or `export WANDB_MODE=disabled` to train without it.

`prepare_finetune_data.py` links the cropped videos of all generators (T2V and I2V) and of the real FF++ videos into `datasets/sfff_finetune/{REAL,FAKE}/videos/`, with the train/val/test lists in `csv_files/`, following the FF++ split.

Best checkpoint → `logs/sfff_finetune/checkpoints/best.ckpt` — move to `weights/finetuning/dfdfcg_finetuned.ckpt`.

## Training from scratch

Same face detection preprocessing as finetuning, then:

```bash
python prepare_scratch_data.py --splits_dir /path/to/SynthForensics/splits   # run once — populates datasets/sfff_scratch/

python main.py \
    --config configs/base.yaml \
    --config configs/clip/L14/ffg.yaml \
    --config configs/scratch_base.yaml
```

`prepare_scratch_data.py` does the same as `prepare_finetune_data.py` with the in-domain generators only (CogVideoX, Wan2.1, LTX-2.3, MAGI-1; T2V and I2V), into `datasets/sfff_scratch/`.

Best checkpoint → `logs/sfff_scratch/checkpoints/best.ckpt` — move to `weights/from_scratch/dfdfcg_from_scratch.ckpt`.

Testing with trained weights uses the same inference commands as [Usage](#usage) — replace `--model-ckpt` with the new checkpoint path.
