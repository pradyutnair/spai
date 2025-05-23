#!/bin/bash

ROOT_DIR="/home/azywot/DL2/spai"
PRETRAINED="/home/azywot/DL2/spai/weights/spai.pth"
OUTPUT_DIR="/scratch-shared/dl2_spai_models"
MAX_JOBS=4       # Max concurrent jobs allowed
SLEEP_TIME=60    # Seconds to wait before checking again

# Models -> config files
declare -A CONFIGS=(
  ["dino_cross_attn_after_sca"]="/home/azywot/DL2/spai/configs/dino_spai_after_sca.yaml"
  # ["clip_cross_attn_after_sca"]="/home/azywot/DL2/spai/configs/clip_spai_after_sca.yaml"
  # ["clip_cross_attn_before_sca"]="/home/azywot/DL2/spai/configs/clip_spai_before_sca.yaml"
  # ["clip_dual_cross_attn_after_sca"]="/home/azywot/DL2/spai/configs/clip_spai_dual_after_sca.yaml"
  # ["clip_dual_cross_attn_before_sca"]="/home/azywot/DL2/spai/configs/clip_spai_dual_before_sca.yaml"
#   ["convnext_cross_attn_after_sca"]="/home/azywot/DL2/spai/configs/convnext_spai_after_sca.yaml"
#   ["convnext_cross_attn_before_sca"]="/home/azywot/DL2/spai/configs/convnext_spai_before_sca.yaml"
#   ["convnext_dual_cross_attn_after_sca"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_after_sca.yaml"
#   ["convnext_dual_cross_attn_before_sca"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_before_sca.yaml"
)

# Dataset splits or CSVs (can add more)
declare -A DATASETS=(
  # ["ldm"]="/home/azywot/DL2/spai/datasets/ldm_train_val_subset.csv"
  ["ldm_lsun"]="/home/azywot/DL2/spai/datasets/ldm_lsun_train_val_subset.csv"
  # ["chameleon"]="/home/azywot/DL2/spai/datasets/chameleon_dataset_split.csv"
)

sanitize() {
  echo "$1" | tr -cd 'a-zA-Z0-9._-'
}

wait_for_available_slot() {
  while true; do
    CURRENT_JOBS=$(squeue -u "$USER" -h | wc -l)
    if (( CURRENT_JOBS < MAX_JOBS )); then
      break
    fi
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] ⏳ Too many jobs queued ($CURRENT_JOBS). Waiting for available slot..."
    sleep "$SLEEP_TIME"
  done
}

for model_name in "${!CONFIGS[@]}"; do
  CONFIG_PATH="${CONFIGS[$model_name]}"

  for ds_name in "${!DATASETS[@]}"; do
    DATA_PATH="${DATASETS[$ds_name]}"
    SAFE_MODEL_NAME=$(sanitize "$model_name")
    SAFE_DS_NAME=$(sanitize "$ds_name")

    TAG="train_${SAFE_MODEL_NAME}_${SAFE_DS_NAME}"

    wait_for_available_slot

    echo "📤 Submitting training job: model=$SAFE_MODEL_NAME, dataset=$SAFE_DS_NAME"

    sbatch \
      --job-name="$TAG" \
      --output="/home/azywot/DL2/spai/jobs/out_files_train/${TAG}_%A.out" \
      --partition=gpu_h100 \
      --gpus-per-node=1 \
      --cpus-per-task=16 \
      --time=02:00:00 \
      --mem=180G \
      --hint=nomultithread \
      --export=ALL,ROOT_DIR="$ROOT_DIR",CONFIG_PATH="$CONFIG_PATH",PRETRAINED="$PRETRAINED",OUTPUT_DIR="$OUTPUT_DIR",DATA_PATH="$DATA_PATH",TAG="$TAG" \
      /home/azywot/DL2/spai/jobs/train/new/run_train.job
  done
done
