#!/bin/bash

# Path to project root (if needed)
ROOT_DIR="/home/azywot/DL2/spai"
MODEL_DIR="/scratch-shared/dl2_spai_models/finetune"

# Test sets
declare -A TEST_SETS=(
  ["flux"]="test_set_flux.csv"
  # ["gigagan"]="test_set_gigagan.csv"
  # ["midjourney"]="test_set_midjourney-v6_1.csv"
  # ["sd3"]="test_set_sd3_fixed.csv"
)

# Model -> Model path
declare -A MODELS=(
  # # CHAMELEON
  ["clip_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_clip_cross_attn_after_sca_chameleon/ckpt_best.pth"
  ["clip_cross_attn_before_sca_chameleon"]="$MODEL_DIR/train_clip_cross_attn_before_sca_chameleon/ckpt_best.pth"
  ["clip_dual_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_clip_dual_cross_attn_after_sca_chameleon/ckpt_best.pth"
  ["clip_dual_cross_attn_before_sca_chameleon"]="$MODEL_DIR/train_clip_dual_cross_attn_before_sca_chameleon/ckpt_best.pth"
  # ["convnext_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_convnext_cross_attn_after_sca_chameleon/ckpt_best.pth"
  # ["convnext_cross_attn_before_sca_chameleon"]="$MODEL_DIR/train_convnext_cross_attn_before_sca_chameleon/ckpt_best.pth"
  # ["convnext_dual_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_convnext_dual_cross_attn_after_sca_chameleon/ckpt_best.pth"
  # ["convnext_dual_cross_attn_before_sca_chameleon"]="$MODEL_DIR/train_convnext_dual_cross_attn_before_sca_chameleon/ckpt_best.pth"
  # # LDM
  ["clip_cross_attn_after_sca_ldm"]="$MODEL_DIR/train_clip_cross_attn_after_sca_ldm/ckpt_best.pth"
  ["clip_cross_attn_before_sca_ldm"]="$MODEL_DIR/train_clip_cross_attn_before_sca_ldm/ckpt_best.pth"
  ["clip_dual_cross_attn_after_sca_ldm"]="$MODEL_DIR/train_clip_dual_cross_attn_after_sca_ldm/ckpt_best.pth"
  ["clip_dual_cross_attn_before_sca_ldm"]="$MODEL_DIR/train_clip_dual_cross_attn_before_sca_ldm/ckpt_best.pth"
  # ["convnext_cross_attn_after_sca_ldm"]="$MODEL_DIR/train_convnext_cross_attn_after_sca_ldm/ckpt_best.pth"
  # ["convnext_cross_attn_before_sca_ldm"]="$MODEL_DIR/train_convnext_cross_attn_before_sca_ldm/ckpt_best.pth"
  # ["convnext_dual_cross_attn_after_sca_ldm"]="$MODEL_DIR/train_convnext_dual_cross_attn_after_sca_ldm/ckpt_best.pth"
  # ["convnext_dual_cross_attn_before_sca_ldm"]="$MODEL_DIR/train_convnext_dual_cross_attn_before_sca_ldm/ckpt_best.pth"
  # # LDM + LSUN
  ["clip_cross_attn_after_sca_ldm_lsun"]="$MODEL_DIR/train_clip_cross_attn_after_sca_ldm_lsun/ckpt_best.pth"
  ["clip_cross_attn_before_sca_ldm_lsun"]="$MODEL_DIR/train_clip_cross_attn_before_sca_ldm_lsun/ckpt_best.pth"
  ["clip_dual_cross_attn_after_sca_ldm_lsun"]="$MODEL_DIR/train_clip_dual_cross_attn_after_sca_ldm_lsun/ckpt_best.pth"
  ["clip_dual_cross_attn_before_sca_ldm_lsun"]="$MODEL_DIR/train_clip_dual_cross_attn_before_sca_ldm_lsun/ckpt_best.pth"
  # ["convnext_cross_attn_after_sca_ldm_lsun"]="$MODEL_DIR/train_convnext_cross_attn_after_sca_ldm_lsun/ckpt_best.pth"
  # ["convnext_cross_attn_before_sca_ldm_lsun"]="$MODEL_DIR/train_convnext_cross_attn_before_sca_ldm_lsun/ckpt_best.pth"
  # ["convnext_dual_cross_attn_after_sca_ldm_lsun"]="$MODEL_DIR/train_convnext_dual_cross_attn_after_sca_ldm_lsun/ckpt_best.pth"
  # ["convnext_dual_cross_attn_before_sca_ldm_lsun"]="$MODEL_DIR/train_convnext_dual_cross_attn_before_sca_ldm_lsun/ckpt_best.pth"
)

# Model -> Config path
declare -A CONFIGS=(
  # # CHAMELEON
  ["clip_cross_attn_after_sca_chameleon"]="/home/azywot/DL2/spai/configs/clip_spai_after_sca.yaml"
  ["clip_cross_attn_before_sca_chameleon"]="/home/azywot/DL2/spai/configs/clip_spai_before_sca.yaml"
  ["clip_dual_cross_attn_after_sca_chameleon"]="/home/azywot/DL2/spai/configs/clip_spai_dual_after_sca.yaml"
  ["clip_dual_cross_attn_before_sca_chameleon"]="/home/azywot/DL2/spai/configs/clip_spai_dual_before_sca.yaml"
  # ["convnext_cross_attn_after_sca_chameleon"]="/home/azywot/DL2/spai/configs/convnext_spai_after_sca.yaml"
  # ["convnext_cross_attn_before_sca_chameleon"]="/home/azywot/DL2/spai/configs/convnext_spai_before_sca.yaml"
  # ["convnext_dual_cross_attn_after_sca_chameleon"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_after_sca.yaml"
  # ["convnext_dual_cross_attn_before_sca_chameleon"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_before_sca.yaml"
  # # LDM
  ["clip_cross_attn_after_sca_ldm"]="/home/azywot/DL2/spai/configs/clip_spai_after_sca.yaml"
  ["clip_cross_attn_before_sca_ldm"]="/home/azywot/DL2/spai/configs/clip_spai_before_sca.yaml"
  ["clip_dual_cross_attn_after_sca_ldm"]="/home/azywot/DL2/spai/configs/clip_spai_dual_after_sca.yaml"
  ["clip_dual_cross_attn_before_sca_ldm"]="/home/azywot/DL2/spai/configs/clip_spai_dual_before_sca.yaml"
  # ["convnext_cross_attn_after_sca_ldm"]="/home/azywot/DL2/spai/configs/convnext_spai_after_sca.yaml"
  # ["convnext_cross_attn_before_sca_ldm"]="/home/azywot/DL2/spai/configs/convnext_spai_before_sca.yaml"
  # ["convnext_dual_cross_attn_after_sca_ldm"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_after_sca.yaml"
  # ["convnext_dual_cross_attn_before_sca_ldm"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_before_sca.yaml"
  # # LDM + LSUN
  ["clip_cross_attn_after_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/clip_spai_after_sca.yaml"
  ["clip_cross_attn_before_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/clip_spai_before_sca.yaml"
  ["clip_dual_cross_attn_after_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/clip_spai_dual_after_sca.yaml"
  ["clip_dual_cross_attn_before_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/clip_spai_dual_before_sca.yaml"
  # ["convnext_cross_attn_after_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/convnext_spai_after_sca.yaml"
  # ["convnext_cross_attn_before_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/convnext_spai_before_sca.yaml"
  # ["convnext_dual_cross_attn_after_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_after_sca.yaml"
  # ["convnext_dual_cross_attn_before_sca_ldm_lsun"]="/home/azywot/DL2/spai/configs/convnext_spai_dual_before_sca.yaml"
)


# Helper function to sanitize strings to allowed SLURM job name chars:
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

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] 📤 Submitting job: model=$SAFE_MODEL_NAME, test_set=$test_name"

    sbatch \
      --job-name=eval_${SAFE_MODEL_NAME}_${SAFE_CSV_NAME} \
      --output=/home/azywot/DL2/spai/jobs/out_files_eval/eval_${SAFE_MODEL_NAME}_${SAFE_CSV_NAME}_%A.out \
      --partition=gpu_h100 \
      --gpus-per-node=1 \
      --cpus-per-task=16 \
      --time=14:00:00 \
      --mem=180G \
      --hint=nomultithread \
      --export=ALL,MODEL_PATH="$MODEL_PATH",MODEL_NAME="$model_name",CSV_NAME="$CSV",CONFIG_PATH="$CONFIG_PATH",ROOT_DIR="$ROOT_DIR" \
      /home/azywot/DL2/spai/jobs/spai_eval/new/run_eval.job
  done
done