# DeepfakeBench — SynthForensics Usage

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

## Environment

```bash
conda env create -f environment.yml
conda activate DeepfakeBench
```

`environment.yml` records the environment used for the reported results.

## ProDet code

The original ProDet repository ([beautyremain/ProDet](https://github.com/beautyremain/ProDet)) has no license, so its three files are not redistributed here. From the DeepfakeBench root:

```bash
git clone https://github.com/beautyremain/ProDet.git ProDet_original
git -C ProDet_original checkout 0508562
cp ProDet_original/training/detectors/utils/prodet_api.py training/detectors/utils/prodet_api.py
cp ProDet_original/training/detectors/ProDet_detector.py training/detectors/prodet_detector.py
cp ProDet_original/training/loss/soft_ce_loss.py training/loss/soft_ce_loss.py
```

Then delete two lines from `training/detectors/prodet_detector.py`: `from dataset.sbiplus_dataset_v2 import SBIPlusV2Dataset` and `self.concat_fc.apply(self.linear_init)`; in `training/loss/soft_ce_loss.py`, replace `from utils.registry import LOSSFUNC` with `from metrics.registry import LOSSFUNC`. All six detectors need this step, because `training/detectors/__init__.py` and `training/loss/__init__.py` import these files.

## Required files

1. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools` folder. This file is necessary for Dlib's face detection functionality.
2. Download the backbone weights from the original DeepfakeBench repository ([pretrained.zip](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip) and [I3D_8x8_R50.pth](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.3/I3D_8x8_R50.pth)) and put them into `./training/pretrained`.
3. Download the detector weights and put them into `./weights` with the file names of the table below (`training/test.py`): RECCE and UCF from the [DeepfakeBench releases](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1); Effort, AltFreezing, FTCN and ProDet from the original repositories ([Effort](https://github.com/YZY-stack/Effort-AIGI-Detection), checkpoint trained on FaceForensics++; [AltFreezing](https://github.com/ZhendongWang6/AltFreezing); [FTCN](https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth); [ProDet](https://github.com/beautyremain/ProDet), overall-best checkpoint).
4. Effort downloads `openai/clip-vit-large-patch14` from Hugging Face on first use.

## Input format

Build the fake test set folders from the published SynthForensics dataset, and the real test set folders from the original FaceForensics++ (raw) and DeepFakeDetection (c23) videos. From the DeepfakeBench root:

```bash
python preprocessing/create_dfb_data.py --dataset_root /path/to/SynthForensics

python preprocessing/create_dfb_real_data.py \
    --ffpp_dir /path/to/FF++/original_sequences/youtube/raw/videos \
    --dfd_dir  /path/to/DFD/original_sequences/actors/c23/videos
```

Expected original real videos:

```
FF++/original_sequences/youtube/raw/videos/000.mp4 ...
DFD/original_sequences/actors/c23/videos/01__exit_phone_room.mp4 ...
```

Expected output structure:

- Fake: `DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR>/`
- Real: `DFB_Data/real/<DATASET>/real_<COMP>_test_set/`

where `<SPLIT>` is `SF-FF` or `SF-DFD`; `<DATASET>` is `FF++` or `DFD`; `_i2v` marks I2V videos; `<COMP>` is `raw` (no suffix), `canonical`, `crf23`, or `crf40`.

## Usage

### All combinations

Runs the three steps below on every split / modality / compression / generator, with all detectors. From the DeepfakeBench root:

```bash
python SF_scripts/sf_run_all.py --gpu_id 0
```

Results are written to `results/sf-<split>[-i2v]/<DETECTOR>/`. Completed steps are skipped, so the script can be stopped and resumed.

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `DFB_Data` | Fake test set folders created by `create_dfb_data.py` |
| `--real_root` | `DFB_Data/real` | Real test set folders created by `create_dfb_real_data.py` |
| `--proc_dir` | `preprocessing` | Extracted face crops (`fake/` and `real/` subfolders) |
| `--weights_dir` | `weights` | Detector checkpoints |
| `--gpu_id` | `0` | CUDA device index |
| `--workers` | CPU count (max 8) | Parallel workers for face-crop extraction |
| `--splits` | all | Subset of `ff`, `dfd` |
| `--modalities` | all | Subset of `t2v`, `i2v` |
| `--compressions` | all | Subset of `raw`, `canonical`, `crf23`, `crf40` |
| `--generators` | all | Subset of generators |
| `--detectors` | all | Subset of `recce`, `ucf`, `effort`, `altfreezing`, `ftcn`, `prodet` |

### Single combination

Three steps must be run in order for each `(split, modality, compression, generator)` combination.

**Step 1 — Extract frames**

```bash
cd preprocessing

python custom_preprocess.py \
    --video_dir  /path/to/DeepfakeBench/DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR> \
    --output_dir ./fake/<SPLIT>/<MODALITY>/<COMP>/<GENERATOR> \
    --num_frames 32 \
    --mode fixed_num_frames
```

where `<MODALITY>` is `t2v` or `i2v`. Run once per dataset/compression for real videos too (`--video_dir /path/to/DeepfakeBench/DFB_Data/real/<DATASET>/real_<COMP>_test_set`, `--output_dir ./real/<DATASET>/<COMP>`).

**Step 2 — Generate dataset JSON**

```bash
python custom_rearrange.py \
    --dataset_name    <name> \
    --fake_frames_dir ./fake/<SPLIT>/<MODALITY>/<COMP>/<GENERATOR>/frames \
    --real_frames_dir ./real/<DATASET>/<COMP>/frames \
    --output_dir      ./dataset_json
```

This creates `./dataset_json/<name>.json`; `<name>` is the dataset name passed to `--test_dataset` in Step 3.

**Step 3 — Run inference**

```bash
cd /path/to/DeepfakeBench

python training/test.py \
    --detector_path training/config/detector/<DETECTOR>.yaml \
    --test_dataset  <name> [<name2> ...] \
    --weights_path  weights/<WEIGHTS>.pth \
    --output_dir    results/sf-<split>[-i2v]/<DETECTOR>
```

Multiple dataset names can be passed to `--test_dataset` in a single call.

## Arguments

`create_dfb_data.py`:

| Argument | Description |
|---|---|
| `--dataset_root` | Root of the published SynthForensics dataset |
| `--output_dir` | Output folder (`DFB_Data`) |
| `--train_data` | Also link the videos used for finetuning / training from scratch (`DFB_Data/SF-FF_full[_i2v]/`) |
| `--dry_run` | Print counts without creating anything |

`create_dfb_real_data.py`:

| Argument | Description |
|---|---|
| `--ffpp_dir` / `--dfd_dir` | Original real videos (see Input format) |
| `--output_dir` | Output folder (`DFB_Data/real`) |
| `--workers` | Parallel ffmpeg processes (8) |
| `--dry_run` | Print counts without creating anything |

`create_train_json.py`:

| Argument | Description |
|---|---|
| `--name` | `sfff_finetune` (all generators) or `sfff_scratch` (in-domain generators) |
| `--proc_dir` | Folder with the `fake/` and `real/` face crops (`preprocessing`) |
| `--output_dir` | Where to write the JSON file (`preprocessing/dataset_json`) |

`custom_preprocess.py`:

| Argument | Description |
|---|---|
| `--video_dir` | Folder of `.mp4` files to preprocess |
| `--output_dir` | Output root for extracted frames |
| `--num_frames` | Frames to extract per video (32) |
| `--mode` | Sampling mode (`fixed_num_frames`) |
| `--stride` | Frame stride, only with `--mode fixed_stride` (10) |
| `--workers` | Parallel workers (CPU count, max 8) |

`custom_rearrange.py`:

| Argument | Description |
|---|---|
| `--dataset_name` | Dataset name: the JSON is saved as `<name>.json` and `<name>` is passed to `--test_dataset` in `test.py` |
| `--fake_frames_dir` | Path to `frames/` subfolder of preprocessed fake videos |
| `--real_frames_dir` | Path to `frames/` subfolder of preprocessed real videos |
| `--output_dir` | Where to write the JSON file (`./dataset_json`) |

`training/test.py` — detectors used in this project:

| Detector | `--detector_path` | `--weights_path` |
|---|---|---|
| RECCE | `recce.yaml` | `recce_best.pth` |
| UCF | `ucf.yaml` | `ucf_best.pth` |
| Effort | `effort.yaml` | `effort_clip_L14_trainOn_FaceForensic.pth` |
| AltFreezing | `altfreezing.yaml` | `altfreezing_ckpt.pth` |
| FTCN | `ftcn.yaml` | `ftcn_tt.pth` |
| ProDet | `prodet.yaml` | `ProDet_best.pth` |

## Output

Results are written to `--output_dir`, one CSV per dataset name (`<name>.csv`), with the metrics (`acc`, `auc`, `eer`, `ap`, `video_auc` (paper metric)) followed by the per-video scores.

## Finetuning

Prepare the training data (raw videos of all FF++ IDs, all generators) from the DeepfakeBench root:

```bash
python preprocessing/create_dfb_data.py --dataset_root /path/to/SynthForensics --train_data
```

Extract face crops from `preprocessing/`, for each `<MODALITY>` (`t2v`, `i2v`) and `<GENERATOR>`, and for the real FF++ videos:

```bash
cd preprocessing

python custom_preprocess.py \
    --video_dir  /path/to/DeepfakeBench/DFB_Data/SF-FF_full[_i2v]/<GENERATOR> \
    --output_dir ./fake/SF-FF_full/<MODALITY>/<GENERATOR> \
    --num_frames 32 \
    --mode fixed_num_frames

python custom_preprocess.py \
    --video_dir  /path/to/FF++/original_sequences/youtube/raw/videos \
    --output_dir ./real/FF++_full \
    --num_frames 32 \
    --mode fixed_num_frames
```

Generate the training JSON:

```bash
python create_train_json.py --name sfff_finetune
```

This creates `./dataset_json/sfff_finetune.json`, which defines the `sfff_finetune` dataset: fake videos of all generators (T2V and I2V) and real FF++ videos, split into train/val/test following the FF++ split (checkpoint selection uses the validation split, see `create_train_json.py`).

Finetune from the pretrained detector weights (from the DeepfakeBench root):

```bash
python training/train.py \
    --detector_path training/config/detector/<DETECTOR>_finetune.yaml \
    --weights_path  weights/<WEIGHTS>.pth \
    --train_dataset sfff_finetune \
    --test_dataset  sfff_finetune
```

Config YAMLs: `training/config/detector/<detector>_finetune.yaml` (e.g. `recce_finetune.yaml`). Best checkpoint: `logs/training/<DETECTOR>_<timestamp>/test/sfff_finetune/ckpt_best.pth` → copy to `weights/finetuning/`.

## Training from scratch

Same data preparation and face crops as finetuning, then generate the training JSON:

```bash
cd preprocessing
python create_train_json.py --name sfff_scratch
```

This creates `./dataset_json/sfff_scratch.json`, which defines the `sfff_scratch` dataset: fake videos of the in-domain generators (CogVideoX, Wan2.1, LTX-2.3, MAGI-1; T2V and I2V) and real FF++ videos, split into train/val/test following the FF++ split.

Train from the DeepfakeBench root:

```bash
python training/train.py \
    --detector_path training/config/detector/<DETECTOR>_scratch.yaml \
    --train_dataset sfff_scratch \
    --test_dataset  sfff_scratch
```

Config YAMLs: `training/config/detector/<detector>_scratch.yaml`. Best checkpoint: `logs/training/<DETECTOR>_<timestamp>/test/sfff_scratch/ckpt_best.pth` → copy to `weights/from_scratch/`.

Testing with trained weights uses the same inference commands as [Single combination](#single-combination) — replace `--weights_path` with the new checkpoint. For UCF, also replace `--detector_path` with `ucf_finetune.yaml` (finetuning) or `ucf_scratch.yaml` (training from scratch).
