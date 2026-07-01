#!/usr/bin/env bash
set -euo pipefail

# Kaggle usage:
#   unzip /kaggle/input/fg-elastica-project/fg_elastica_project_kaggle.zip -d /kaggle/working
#   bash /kaggle/working/fg_elastica_project/kaggle_train.sh /kaggle/input/YOUR_DATASET_IMAGE_DIR
#
# Optional environment overrides:
#   MAX_IMAGES=10000 VAL_RATIO=0.1 TEST_RATIO=0.1 SEED=42 bash kaggle_train.sh /kaggle/input/images
#   FID_DURING_TRAIN=false bash kaggle_train.sh /kaggle/input/images

IMAGES_ROOT="${1:-/kaggle/input/YOUR_DATASET_IMAGE_DIR}"
PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/fg_elastica_project}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/mvp_kaggle_9k.yaml}"
RUN_NAME="${RUN_NAME:-fg_elastica_kaggle_9k_mvp_vgg_fid}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/outputs/${RUN_NAME}}"
MAX_IMAGES="${MAX_IMAGES:-10000}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"
FID_DURING_TRAIN="${FID_DURING_TRAIN:-false}"
export CONFIG_PATH

cd "${PROJECT_DIR}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements_optional.txt

python prepare_kaggle.py \
  --images-root "${IMAGES_ROOT}" \
  --work-dir /kaggle/working \
  --max-images "${MAX_IMAGES}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --seed "${SEED}"

if [[ "${FID_DURING_TRAIN}" == "false" ]]; then
  python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ.get("CONFIG_PATH", "/kaggle/working/fg_elastica_project/configs/mvp_kaggle_9k.yaml"))
text = path.read_text(encoding="utf-8")
text = text.replace("fid_during_train: true", "fid_during_train: false")
path.write_text(text, encoding="utf-8")
PY
fi

python train.py --config "${CONFIG_PATH}"

python evaluate.py \
  --config "${OUTPUT_DIR}/config_resolved.yaml" \
  --checkpoint "${OUTPUT_DIR}/best.pt" \
  --split test

echo "Done. Logs:"
echo "  ${OUTPUT_DIR}/log.csv"
echo "  ${OUTPUT_DIR}/bucket_metrics.csv"
echo "  ${OUTPUT_DIR}/eval_test.json"
