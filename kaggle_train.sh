#!/usr/bin/env bash
set -euo pipefail

# Usage in a Kaggle notebook terminal/cell:
#   bash kaggle_train.sh /kaggle/input/YOUR_DATASET_IMAGE_DIR
#
# The image directory should contain jpg/png/webp/bmp images, searched recursively.

IMAGES_ROOT="${1:-/kaggle/input/YOUR_DATASET_IMAGE_DIR}"
PROJECT_DIR="/kaggle/working/fg_elastica_project"
CONFIG_PATH="${PROJECT_DIR}/configs/mvp_kaggle_9k.yaml"
RUN_NAME="fg_elastica_kaggle_9k_mvp_vgg_fid"
OUTPUT_DIR="/kaggle/working/outputs/${RUN_NAME}"

cd "${PROJECT_DIR}"

python -m pip install -r requirements.txt -r requirements_optional.txt

python prepare_kaggle.py \
  --images-root "${IMAGES_ROOT}" \
  --work-dir /kaggle/working \
  --max-images 10000 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42

python train.py --config "${CONFIG_PATH}"

python evaluate.py \
  --config "${OUTPUT_DIR}/config_resolved.yaml" \
  --checkpoint "${OUTPUT_DIR}/best.pt" \
  --split test
