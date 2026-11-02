# Detectors

This directory contains several state-of-the-art video deepfake detection models. Each subdirectory contains one detector (DeepfakeBench contains six), with its own environment, inference pipeline and a README with instructions (CFM, DFD-FCG and the ProDet files of DeepfakeBench also need their original repository, see the README of their folder).

## Environment Setup

Every detector directory provides an `environment.yml`:

```bash
conda env create -f <Detector>/environment.yml
```

Some detectors need a slightly different procedure, for example extra packages installed after creating the environment, CUDA-specific wheels or a setup script. When a detector's `README.md` has an *Environment* section, follow that section instead of the command above: it reproduces the environment used for the reported results.

## Pretrained Weights

Model weights are not included in this repository. Download them following the original README of each detector (`ORIGINAL_README.md`, or the original repository for CFM and DFD-FCG), which links the official weights released by the original authors, then place them where the detector's `README.md` expects them (it lists the paths and checkpoints used for the reported results).

Training-free detectors such as D3 download their feature encoders automatically from Hugging Face on first use.

## Available Detectors

| Detector | Directory | Original repository | License |
|----------|-----------|---------------------|---------|
| CFM | `CFM/` | [LoveSiameseCat/CFM](https://github.com/LoveSiameseCat/CFM) | none (original code not included, see `CFM/README.md`) |
| D3 | `D3/` | [Zig-HS/D3](https://github.com/Zig-HS/D3) | MIT |
| DeepfakeBench (RECCE, ProDet, UCF, Effort, AltFreezing, FTCN) | `DeepfakeBench/` | [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench); ProDet: [beautyremain/ProDet](https://github.com/beautyremain/ProDet) | CC BY-NC 4.0; ProDet: none (original code not included, see `DeepfakeBench/README.md`) |
| DFD-FCG | `DFD-FCG/` | [aiiu-lab/DFD-FCG](https://github.com/aiiu-lab/DFD-FCG) | none (original code not included, see `DFD-FCG/README.md`) |
| FakeSTormer | `FakeSTormer/` | [10Ring/FakeSTormer](https://github.com/10Ring/FakeSTormer) | MIT |
| GenConViT | `GenConViT/` | [erprogs/GenConViT](https://github.com/erprogs/GenConViT) | GPL-3.0 |
| GenD | `GenD/` | [yermandy/GenD](https://github.com/yermandy/GenD) | MIT |
| LAA-Net | `LAA-Net/` | [10Ring/LAA-Net](https://github.com/10Ring/LAA-Net) | MIT |
| MM-Det | `MM-Det/` | [SparkleXFantasy/MM-Det](https://github.com/SparkleXFantasy/MM-Det) | Apache 2.0 |
| NSG-VD | `NSG-VD/` | [ZSHsh98/NSG-VD](https://github.com/ZSHsh98/NSG-VD) | Apache 2.0 |

Apart from CFM, DFD-FCG and the ProDet files, each folder keeps the license of its original repository (`LICENSE`).
