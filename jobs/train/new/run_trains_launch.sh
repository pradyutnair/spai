#!/bin/bash

ROOT_DIR="/home/azywot/DL2/spai"
PRETRAINED="/home/azywot/DL2/spai/weights/spai.pth"

# Models -> paths (model checkpoint to fine-tune from)
declare -A PRETRAINED_MODELS=(
  ["clip_cross_attn_after_sca"]="/home/azywot/DL2/spai/weights/spai.pth"
  # add others if needed
)

# Models -> config files
declare -A CONFIGS=(
  ["clip_cross_attn_after_sca"]="/home/azywot/DL2/spai/configs/TEST_spai_after_sca.yaml"
)

# Models -> output dirs
declare -A OUTPUT_DIRS=(
  ["clip_cross_attn_after_sca"]="/home/azywot/DL2/spai/output/17_05_2025/TEST_clip"
)

# Dataset splits or CSVs (can add more)
declare -A DATASETS=(
  ["ldm"]="/home/azywot/DL2/spai/datasets/ldm_train_val_subset.csv"
)

sanitize() {
  echo "$1" | tr -cd 'a-zA-Z0-9._-'
}

for model_name in "${!CONFIGS[@]}"; do
  CONFIG_PATH="${CONFIGS[$model_name]}"
  OUTPUT_DIR="${OUTPUT_DIRS[$model_name]}"

  for ds_name in "${!DATASETS[@]}"; do
    DATA_PATH="${DATASETS[$ds_name]}"
    SAFE_MODEL_NAME=$(sanitize "$model_name")
    SAFE_DS_NAME=$(sanitize "$ds_name")

    TAG="train_${SAFE_MODEL_NAME}_${SAFE_DS_NAME}"

    echo "📤 Submitting training job: model=$SAFE_MODEL_NAME, dataset=$SAFE_DS_NAME"

    sbatch \
      --job-name="$TAG" \
      --output="/home/azywot/DL2/spai/jobs/out_files_train/${TAG}_%A.out" \
      --partition=gpu_h100 --gpus-per-node=1 --cpus-per-task=16 --time=14:00:00 --mem=180G --hint=nomultithread \
      --export=ALL,ROOT_DIR="$ROOT_DIR",CONFIG_PATH="$CONFIG_PATH",PRETRAINED="$PRETRAINED",OUTPUT_DIR="$OUTPUT_DIR",DATA_PATH="$DATA_PATH",TAG="$TAG" \
      /home/azywot/DL2/spai/jobs/train/new/run_train.job
  done
done
