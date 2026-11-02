#!/usr/bin/env bash
# Create and configure the conda environment for FakeSTormer on A100 (CUDA 11.1 / PyTorch 1.8.0)
set -e

ENV_NAME="fakestormer"

echo "==> Creating conda environment: $ENV_NAME"
conda env create -f environment.yml

echo "==> Installing CUDA 11.1 torch/torchvision wheels (conda resolves pytorch==1.8.0 to a CPU build)"
conda run -n "$ENV_NAME" pip install \
    torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
    --extra-index-url https://download.pytorch.org/whl/cu111

echo "==> Installing retinaface-pytorch 0.0.8 without dependencies (its metadata pins torch==1.9.0)"
conda run -n "$ENV_NAME" pip install --no-deps retinaface-pytorch==0.0.8

echo "==> Installing mmcv-full 1.6.1 (prebuilt cu111/torch1.8.0)"
conda run -n "$ENV_NAME" pip install mmcv-full==1.6.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

echo ""
echo "==> Setup complete."
echo "    Activate the env with: conda activate $ENV_NAME"
echo ""
echo "    NOTE: download pretrained weights from Dropbox and place them in pretrained/"
echo "    NOTE: download shape_predictor_81_face_landmarks.dat and place it in pretrained/"
