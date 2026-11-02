# CFM

> Original code and documentation: [LoveSiameseCat/CFM](https://github.com/LoveSiameseCat/CFM). The original repository has no license, so its code is not redistributed here: `inference.py` imports the model definition from the original `test.py` (see Setup).

## Setup

```bash
cd src/detectors/CFM
git clone https://github.com/LoveSiameseCat/CFM.git original
git -C original checkout d1f0590
cp original/test.py .
```

## Environment

```bash
conda env create -f environment.yml
conda activate cfm
```

## Weights

Download the pretrained model [models_params_28.tar](https://drive.google.com/file/d/1iP8k4ffJ-3WcuEg5uyrsY3qOY0ALi83H/view?usp=sharing) (from the original CFM repository) and put it in the CFM folder.

## Input format

CFM uses the face crops extracted by **DeepfakeBench** (256×256): run the DeepfakeBench preprocessing first, i.e. Step 1 of the [DeepfakeBench README](../DeepfakeBench/README.md#single-combination) (or `SF_scripts/sf_run_all.py`, which also extracts them).
Expected folder structure:

```
<folder>/
└── frames/
    ├── video_001/
    │   ├── 000.png
    │   └── ...
    └── video_002/
        └── ...
```

Paths after DeepfakeBench preprocessing:

- Fake: `../DeepfakeBench/preprocessing/fake/<SPLIT>/<MODALITY>/<COMP>/<GENERATOR>/`
- Real: `../DeepfakeBench/preprocessing/real/<DATASET>/<COMP>/`

where `<SPLIT>` is `SF-FF` or `SF-DFD`, `<DATASET>` is the matching real dataset (`FF++` or `DFD`), `<MODALITY>` is `t2v` or `i2v`, and `<COMP>` is `raw`, `canonical`, `crf23`, or `crf40`.

## Usage

```bash
python inference.py \
    --checkpoint models_params_28.tar \
    --device cuda \
    --fake-folder ../DeepfakeBench/preprocessing/fake/<SPLIT>/<MODALITY>/<COMP>/<GENERATOR> \
    --real-folder ../DeepfakeBench/preprocessing/real/<DATASET>/<COMP> \
    --output <output_name>
```

`<output_name>` is the name of the result file: results are saved to `results/<output_name>.csv`. It can include subfolders (e.g. `sf-ff/sfff_raw_helios` → `results/sf-ff/sfff_raw_helios.csv`); if omitted, the fake folder name is used.

Multiple fake folders can be passed in a single run (real inference executes only once and is reused), with one output name per fake folder:

```bash
python inference.py \
    --checkpoint models_params_28.tar \
    --device cuda \
    --fake-folder ../DeepfakeBench/preprocessing/fake/SF-FF/t2v/raw/helios ../DeepfakeBench/preprocessing/fake/SF-FF/t2v/raw/cogvideox \
    --real-folder ../DeepfakeBench/preprocessing/real/FF++/raw \
    --output sf-ff/sfff_raw_helios sf-ff/sfff_raw_cogvideox
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.tar` checkpoint |
| `--fake-folder` | required | One or more fake frame folders |
| `--real-folder` | required | Real frame folder |
| `--output` | fake folder name | Output name(s), one per fake folder: results are saved to `results/<output>.csv` |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--threshold` | `0.5` | Fake/real classification threshold |
| `--test_mode` | off | Process only 2 videos × 32 frames (quick sanity check) |

## Output

Each output CSV contains the metrics (`acc`, `auc`, `eer`, `ap`, `video_auc` (paper metric)) followed by the per-video scores.
