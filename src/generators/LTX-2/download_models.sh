#!/bin/bash
# Download the LTX-2.3 weights and the Gemma 3 text encoder used for SynthForensics.
# Revisions are pinned to the ones used to generate the dataset.
set -e

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

hf download Lightricks/LTX-2.3 \
    ltx-2.3-22b-dev.safetensors \
    ltx-2.3-22b-distilled-lora-384.safetensors \
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --revision 5a9c1c680bc66c159f708143bf274739961ecd08 \
    --local-dir "$MODELS_DIR/v2.3"

# Gemma 3 is gated: request access on its Hugging Face page first.
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --revision 68f7ee4fbd59087436ada77ed2d62f373fdd4482 \
    --local-dir "$MODELS_DIR/gemma-3-12b"
