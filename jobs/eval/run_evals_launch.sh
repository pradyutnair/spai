#!/bin/bash
# set -e # exit on error

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================

# --- Job Control ---
MAX_JOBS=4
SLEEP_TIME=60

# --- SLURM Parameters (override sbatch defaults in run_eval.job) ---
PARTITION="gpu_h100"
TIME_LIMIT="14:00:00"
GPUS_PER_NODE=1
CPUS_PER_TASK=16
MEMORY="180G"

# --- Python Script Parameters ---
BATCH_SIZE=8
NUM_WORKERS=8
MIN_PATCHES=4
FEATURE_BATCH=400
PREFETCH_FACTOR=1

# --- Path Configuration ---
USER="igodzwon" # NOTE: hange this to your username!
ROOT_DIR="$HOME/spai"
MODEL_DIR="/scratch-shared/dl2_spai_models/finetune"
DATASET_DIR="/scratch-shared/dl2_all_data/testsets"

# --- Experiment Definitions ---
declare -A TEST_SETS=(
  ["MJ6.1"]="$DATASET_DIR/test_set_spai_test_images_midjourney-v6.1.csv"
  ["dalle2"]="$DATASET_DIR/test_set_synthbuster_dalle2.csv"
  ["dalle2_2"]="$DATASET_DIR/test_set_TestSet_dalle_2.csv"
  ["dalle3"]="$DATASET_DIR/test_set_synthbuster_dalle3.csv"
  ["glide"]="$DATASET_DIR/test_set_TestSet_glide_text2img_valid.csv"
  ["flux"]="$DATASET_DIR/test_set_spai_test_images_flux.csv"
  ["gigagan"]="$DATASET_DIR/test_set_spai_test_images_gigagan.csv"
  ["firefly"]="$DATASET_DIR/test_set_synthbuster_firefly.csv"
)

declare -A MODELS=(
  ["scratch_late_fusion"]="/scratch-shared/dl2_spai_models/late_fusion_spai_ldm/train_og_spai_LDM/finetune/train_og_spai_LDM/ckpt_epoch_13.pth" 
  # ["clip_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_clip_cross_attn_after_sca_chameleon/ckpt_best.pth"
  # ["convnext_cross_attn_after_sca_chameleon"]="$MODEL_DIR/train_convnext_cross_attn_after_sca_chameleon/ckpt_best.pth"
)

declare -A CONFIGS=(
  ["scratch_late_fusion"]="/home/igodzwon/spai/configs/spai_latefusion_eval.yaml"
  #["test"]="$ROOT_DIR/configs/spai_latefusion_eval.yaml"
  # ["clip_cross_attn_after_sca_chameleon"]="$ROOT_DIR/configs/clip_spai_after_sca.yaml"
  # ["convnext_cross_attn_after_sca_chameleon"]="$ROOT_DIR/configs/convnext_spai_after_sca.yaml"
)

# ==============================================================================
# SCRIPT LOGIC
# ==============================================================================
# set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT from .env file
# --- Load Environment Variables from .env file ---
ENV_FILE="${ROOT_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
  echo "✅ Loaded environment variables from ${ENV_FILE}"
else
  echo "⚠️ Warning: .env file not found at ${ENV_FILE}. Neptune credentials may be missing."
fi

# --- Helper Functions ---
sanitize() {
  echo "$1" | tr -cd 'a-zA-Z0-9._-'
}

wait_for_available_slot() {
  while true; do
    CURRENT_JOBS=$(squeue -u "$(whoami)" -h --partition "$PARTITION" | wc -l)
    if (( CURRENT_JOBS < MAX_JOBS )); then
      break
    fi
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] ⏳ Too many jobs queued ($CURRENT_JOBS/$MAX_JOBS). Waiting for a slot..."
    sleep "$SLEEP_TIME"
  done
}

# --- Determine which models and tests to run ---
if [ "$#" -eq 0 ]; then
  # No arguments provided, run all combinations
  MODELS_TO_RUN=("${!MODELS[@]}")
  TESTS_TO_RUN=("${!TEST_SETS[@]}")
  echo "🚀 No specific models/tests provided. Running all ${#MODELS_TO_RUN[@]} models on all ${#TESTS_TO_RUN[@]} test sets."
else
  # Use command-line arguments to select models/tests
  MODELS_TO_RUN=()
  TESTS_TO_RUN=()
  for arg in "$@"; do
    [[ -v MODELS[$arg] ]] && MODELS_TO_RUN+=("$arg")
    [[ -v TEST_SETS[$arg] ]] && TESTS_TO_RUN+=("$arg")
  done
  # If one list is empty, default to all from that category
  if [ ${#MODELS_TO_RUN[@]} -eq 0 ]; then MODELS_TO_RUN=("${!MODELS[@]}"); fi
  if [ ${#TESTS_TO_RUN[@]} -eq 0 ]; then TESTS_TO_RUN=("${!TEST_SETS[@]}"); fi
  echo "🚀 Running selected models: [${MODELS_TO_RUN[*]}] on selected test sets: [${TESTS_TO_RUN[*]}]"
fi

# --- Main Job Submission Loop ---
RUN_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")

for model_name in "${MODELS_TO_RUN[@]}"; do
  for test_name in "${TESTS_TO_RUN[@]}"; do
    
    wait_for_available_slot

    # --- Prepare Job-Specific Variables ---
    MODEL_PATH="${MODELS[$model_name]}"
    CONFIG_PATH="${CONFIGS[$model_name]}"
    CSV_NAME="${TEST_SETS[$test_name]}"
    
    SAFE_MODEL_NAME=$(sanitize "$model_name")
    SAFE_TEST_NAME=$(sanitize "$test_name")
    
    JOB_NAME="eval_${SAFE_MODEL_NAME}_${SAFE_TEST_NAME}"
    OUTPUT_DIR="$ROOT_DIR/output/${SAFE_MODEL_NAME}/test_${SAFE_TEST_NAME}/${RUN_TIMESTAMP}"
    NEPTUNE_TAG="eval_${SAFE_MODEL_NAME}_${SAFE_TEST_NAME}_${RUN_TIMESTAMP}"

    echo "-----------------------------------------------------"
    echo "📤 Submitting job: $JOB_NAME"
    echo "   Model: $model_name"
    echo "   Test Set: $test_name"
    echo "   Output Dir: $OUTPUT_DIR"
    echo "-----------------------------------------------------"

    # Export variables that the job script will need
    # sbatch will pass these to the job's environment
    export MODEL_PATH CONFIG_PATH CSV_NAME MODEL_NAME="$model_name"
    export ROOT_DIR OUTPUT_DIR NEPTUNE_TAG
    export TEST_CSV_PATH="${CSV_NAME}"
    export BATCH_SIZE NUM_WORKERS MIN_PATCHES FEATURE_BATCH PREFETCH_FACTOR

    sbatch \
      --job-name="$JOB_NAME" \
      --output="${ROOT_DIR}/outputs/files_eval/${JOB_NAME}_%A.out" \
      --partition="$PARTITION" \
      --gpus-per-node="$GPUS_PER_NODE" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --time="$TIME_LIMIT" \
      --mem="$MEMORY" \
      --export=ALL \
      "${ROOT_DIR}/jobs/eval/run_eval.job"

    sleep 2 # Stagger submissions slightly
  done
done

echo "🎉 All specified evaluation jobs have been submitted."