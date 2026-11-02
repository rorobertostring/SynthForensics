# GenConViT — SynthForensics Usage

> Original documentation: [ORIGINAL_README.md](ORIGINAL_README.md)

## Environment

```bash
conda env create -f environment.yml
conda activate genconvit
```

## Weights

Download the pretrained weights from [Hugging Face](https://huggingface.co/Deressa/GenConViT) (from the original GenConViT repository) into `weight/`:

```bash
wget -P weight https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth
wget -P weight https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth
```

## Input format

Raw `.mp4` video files in a flat folder, one folder per split/compression/generator combination. The folders are built with the DeepfakeBench scripts `create_dfb_data.py` (fake) and `create_dfb_real_data.py` (real), see the [DeepfakeBench README](../DeepfakeBench/README.md#input-format):

- Fake: `../DeepfakeBench/DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR>/`
- Real: `../DeepfakeBench/DFB_Data/real/<DATASET>/real_<COMP>_test_set/`

where `<SPLIT>` is `SF-FF` or `SF-DFD`; `<DATASET>` is the matching real dataset (`FF++` or `DFD`); `_i2v` marks I2V videos; `<COMP>` is `raw` (no suffix for fake folders), `canonical`, `crf23`, or `crf40`.

No preprocessing needed: face extraction is handled internally.

## Usage

```bash
python batch_inference.py \
    --fake-folder ../DeepfakeBench/DFB_Data/<SPLIT>_fake_test_set[_i2v][_<COMP>]/<GENERATOR> \
    --real-folder ../DeepfakeBench/DFB_Data/real/<DATASET>/real_<COMP>_test_set \
    --output      <output_name>
```

`<output_name>` is the name of the result file: results are saved to `results/<output_name>.csv`. It can include subfolders (e.g. `sf-ff/sfff_raw_helios` → `results/sf-ff/sfff_raw_helios.csv`); if omitted, the fake folder name is used.

Multiple fake folders can be passed in a single run (model loaded once, real inference runs once and is reused):

```bash
python batch_inference.py \
    --fake-folder <FAKE1> <FAKE2> ... \
    --real-folder <REAL_FOLDER> \
    --output      <name1> <name2> ...
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--fake-folder` | required | One or more fake video folders |
| `--real-folder` | required | Real video folder |
| `--output` | fake folder name | Output name(s), one per fake folder: results are saved to `results/<output>.csv` |
| `--f` | `15` | Frames sampled per video |
| `--e` | `genconvit_ed_inference` | ED branch weights name (in `weight/`) |
| `--v` | `genconvit_vae_inference` | VAE branch weights name (in `weight/`) |
| `--s` | — | Backbone size: `tiny` or `large` |
| `--fp16` | off | Half precision inference |
| `--output-dir` | `results/` | Override output directory |

## Output

One CSV per fake folder at `results/<output>.csv`. The file starts with the metrics (`acc`, `auc`, `eer`, `ap`, `video_auc` (paper metric)), then per-video rows with score, true label, predicted label.

## Finetuning

Training uses the face crops extracted by DeepfakeBench for the training videos: run the data preparation and face-crop extraction of the [DeepfakeBench Finetuning section](../DeepfakeBench/README.md#finetuning) first (`create_dfb_data.py --train_data` and `custom_preprocess.py`). Then:

```bash
python prepare_finetune_data.py   # run once — populates finetune_data/

python train.py -d finetune_data -m ed  -e 25 -p genconvit_ed_inference  -b 32
python train.py -d finetune_data -m vae -e 25 -p genconvit_vae_inference -b 32
```

`prepare_finetune_data.py` links the face crops of all generators (T2V and I2V) and of the real FF++ videos into `finetune_data/{train,valid,test}/{fake,real}/`, following the FF++ split.

Best weights saved as `weight/genconvit_{ed,vae}_<date>_best.pth` — move to `weight/finetuning/genconvit_ed_finetuned.pth` and `weight/finetuning/genconvit_vae_finetuned.pth`.

## Training from scratch

Same DeepfakeBench face crops as finetuning, then:

```bash
python prepare_scratch_data.py   # run once — populates scratch_data/

python train.py -d scratch_data -m ed  -e 50 -b 32
python train.py -d scratch_data -m vae -e 50 -b 32
```

`prepare_scratch_data.py` does the same as `prepare_finetune_data.py` with the in-domain generators only (CogVideoX, Wan2.1, LTX-2.3, MAGI-1; T2V and I2V), into `scratch_data/`.

Best weights saved as `weight/genconvit_{ed,vae}_<date>_best.pth` — move to `weight/from_scratch/genconvit_ed_from_scratch.pth` and `weight/from_scratch/genconvit_vae_from_scratch.pth`.

Testing with trained weights uses the same inference commands as [Usage](#usage) — replace `--e` / `--v` with the trained weight names (e.g. `--e finetuning/genconvit_ed_finetuned --v finetuning/genconvit_vae_finetuned`).
