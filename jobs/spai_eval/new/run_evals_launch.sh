#!/bin/bash

# Path to project root (if needed)
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/home/azywot/DL2/spai"

# Test sets
declare -A TEST_SETS=(
  ["flux"]="test_set_flux.csv"
  # ["gigagan"]="test_set_gigagan.csv"
  # ["midjourney"]="test_set_midjourney-v6_1.csv"
  # ["sd3"]="test_set_sd3_fixed.csv"
)

# Model -> Model path
declare -A MODELS=(
  ["clip_cross_attn_after_sca_chameleon"]="/home/azywot/DL2/spai/output/train_after_sca_chameleon/finetune/spai_cross_attn_after_sca_chameleon/ckpt_best.pth"
  # ["clip_cross_attn_after_sca_ldm"]="/home/azywot/DL2/spai/output/train_after_sca_ldm/finetune/spai_cross_attn_after_sca_ldm/ckpt_best.pth"
  ["convnext_cross_attn_after_sca_chameleon"]=
)

# Model -> Config path
declare -A CONFIGS=(
  ["clip_cross_attn_after_sca_chameleon"]="/home/azywot/DL2/spai/configs/spai_after_sca.yaml"
  # ["clip_cross_attn_after_sca_ldm"]="/home/azywot/DL2/spai/configs/spai_after_sca.yaml"
)

# Helper function to sanitize strings to allowed SLURM job name chars:
sanitize() {
  echo "$1" | tr -cd 'a-zA-Z0-9._-'
}

for model_name in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$model_name]}"
  CONFIG_PATH="${CONFIGS[$model_name]}"
  for test_name in "${!TEST_SETS[@]}"; do
    CSV="${TEST_SETS[$test_name]}"
    
    # Sanitize model name and CSV name before passing to sbatch
    SAFE_MODEL_NAME=$(sanitize "$model_name")
    SAFE_CSV_NAME=$(sanitize "$CSV")
    
    echo "📤 Submitting job: model=$SAFE_MODEL_NAME, test_set=$test_name"

    sbatch \
        --job-name=eval_${SAFE_MODEL_NAME}_${SAFE_CSV_NAME} \
        --output=/home/azywot/DL2/spai/jobs/out_files_eval/eval_${SAFE_MODEL_NAME}_${SAFE_CSV_NAME}_%A.out \
        --partition=gpu_h100 --gpus-per-node=1 --cpus-per-task=16 --time=14:00:00 --mem=180G --hint=nomultithread \
        --export=ALL,MODEL_PATH="$MODEL_PATH",MODEL_NAME="$model_name",CSV_NAME="$CSV",CONFIG_PATH="$CONFIG_PATH",ROOT_DIR="$ROOT_DIR" \
        /home/azywot/DL2/spai/jobs/spai_eval/new/run_eval.job
  done
done
