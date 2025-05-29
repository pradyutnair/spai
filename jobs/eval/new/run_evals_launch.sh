#!/bin/bash

# Dynamically determine username and home directory
USER=$(whoami)
HOME_DIR="/home/${USER}"

ROOT_DIR="${HOME_DIR}/spai"
MODEL_DIR="/scratch-shared/dl2_spai_models/finetune"
MAX_JOBS=4
SLEEP_TIME=60

# Test sets
declare -A TEST_SETS=(
  ["dalle2"]="test_set_dalle2.csv"
  ["dalle3"]="test_set_dalle3.csv"
  ["sd1_4"]="test_set_sd1_4.csv"
  ["sdxl"]="test_set_sdxl.csv"
)

# Model -> Model path
declare -A MODELS=(
  ["clip_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_clip_cross_attn_after_sca_chameleon/ckpt_best.pth"
  ["convnext_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_convnext_cross_attn_after_sca_chameleon/ckpt_best.pth"
)

# Model -> Config path
declare -A CONFIGS=(
  ["clip_cross_attn_after_sca_chameleon"]="$ROOT_DIR/configs/clip_spai_after_sca.yaml"
  ["convnext_cross_attn_after_sca_chameleon"]="$ROOT_DIR/configs/convnext_spai_after_sca.yaml"
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

for model_name in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$model_name]}"
  CONFIG_PATH="${CONFIGS[$model_name]}"

  for test_name in "${!TEST_SETS[@]}"; do
    CSV="${TEST_SETS[$test_name]}"
    
    SAFE_MODEL_NAME=$(sanitize "$model_name")
    SAFE_CSV_NAME=$(sanitize "$CSV")

    wait_for_available_slot

    sleep 2
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] 📤 Submitting job: model=$SAFE_MODEL_NAME, test_set=$test_name"

    sbatch \
      --job-name=eval_${SAFE_MODEL_NAME}_${SAFE_CSV_NAME} \
      --output=${ROOT_DIR}/jobs/out_files_eval/eval_${SAFE_MODEL_NAME}_${SAFE_CSV_NAME}_%A.out \
      --partition=gpu_h100 \
      --gpus-per-node=1 \
      --cpus-per-task=16 \
      --time=14:00:00 \
      --mem=180G \
      --hint=nomultithread \
      --export=ALL,MODEL_PATH="$MODEL_PATH",MODEL_NAME="$model_name",CSV_NAME="$CSV",CONFIG_PATH="$CONFIG_PATH",ROOT_DIR="$ROOT_DIR" \
      ${ROOT_DIR}/jobs/spai_eval/new/run_eval.job
  done
done
