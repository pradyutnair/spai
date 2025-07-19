#!/bin/bash
# set -e
# set -o pipefail

# ==============================================================================
# SCRIPT SELF-LOCATION
# This makes the script runnable from any directory
# ==============================================================================
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================

# --- Job Control ---
MAX_JOBS=4
SLEEP_TIME=60

# --- SLURM Parameters ---
PARTITION="gpu_h100"
TIME_LIMIT="10:00:00"
GPUS_PER_NODE=1
CPUS_PER_TASK=16
MEMORY="180G"

# --- Python Script Parameters ---
BATCH_SIZE=192
NUM_WORKERS=16
VAL_BATCH_SIZE=256
AMP_OPT_LEVEL="O0"
FEATURE_BATCH=400
PREFETCH_FACTOR=4

# --- Path Configuration ---
USER="azywot" # NOTE: change this to your username!
ROOT_DIR="$HOME/$USER/spai"
PRETRAINED_PATH="${ROOT_DIR}/weights/spai.pth"
OUTPUT_DIR_BASE="/scratch-shared/dl2_spai_models/original_spai" # Base path for outputs

# --- Experiment Definitions ---
# Short Name -> Config File Path
declare -A CONFIGS=(
  ["og_spai"]="${ROOT_DIR}/configs/spai.yaml"
  # ["clip_cross_attn_after_sca"]="${ROOT_DIR}/configs/clip_spai_after_sca.yaml"
  # ["semantic_context"]="${ROOT_DIR}/configs/spai.yaml"
)

# Short Name -> Dataset CSV Path
declare -A DATASETS=(
  ["ldm_coco_lsun"]="/scratch-shared/dl2_all_data/ldm_train_val_trainset.csv"
  # NOTE: add more datasets as needed
)

# ==============================================================================
# SCRIPT LOGIC
# ==============================================================================

# --- Load Environment Variables from .env file in project root ---
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

# --- Determine which models and datasets to run ---
if [ "$#" -eq 0 ]; then
  CONFIGS_TO_RUN=("${!CONFIGS[@]}")
  DATASETS_TO_RUN=("${!DATASETS[@]}")
  echo "🚀 No specific jobs provided. Running all ${#CONFIGS_TO_RUN[@]} configs on all ${#DATASETS_TO_RUN[@]} datasets."
else
  CONFIGS_TO_RUN=()
  DATASETS_TO_RUN=()
  for arg in "$@"; do
    [[ -v CONFIGS[$arg] ]] && CONFIGS_TO_RUN+=("$arg")
    [[ -v DATASETS[$arg] ]] && DATASETS_TO_RUN+=("$arg")
  done
  if [ ${#CONFIGS_TO_RUN[@]} -eq 0 ]; then CONFIGS_TO_RUN=("${!CONFIGS[@]}"); fi
  if [ ${#DATASETS_TO_RUN[@]} -eq 0 ]; then DATASETS_TO_RUN=("${!DATASETS[@]}"); fi
  echo "🚀 Running selected configs: [${CONFIGS_TO_RUN[*]}] on selected datasets: [${DATASETS_TO_RUN[*]}]"
fi

# --- Main Job Submission Loop ---
for config_name in "${CONFIGS_TO_RUN[@]}"; do
  for ds_name in "${DATASETS_TO_RUN[@]}"; do
    
    wait_for_available_slot

    # --- Prepare Job-Specific Variables ---
    CONFIG_PATH="${CONFIGS[$config_name]}"
    DATA_PATH="${DATASETS[$ds_name]}"
    
    SAFE_CONFIG_NAME=$(sanitize "$config_name")
    SAFE_DS_NAME=$(sanitize "$ds_name")
    
    # Create a unique tag and output directory for this specific run
    NEPTUNE_TAG="train_${SAFE_CONFIG_NAME}_${SAFE_DS_NAME}"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${NEPTUNE_TAG}"
    JOB_NAME="$NEPTUNE_TAG"

    echo "-----------------------------------------------------"
    echo "📤 Submitting job: $JOB_NAME"
    echo "   Config: $config_name"
    echo "   Dataset: $ds_name"
    echo "   Output Dir: $OUTPUT_DIR"
    echo "-----------------------------------------------------"

    # Export all variables the job script will need
    export ROOT_DIR CONFIG_PATH PRETRAINED_PATH DATA_PATH OUTPUT_DIR NEPTUNE_TAG
    export BATCH_SIZE NUM_WORKERS VAL_BATCH_SIZE AMP_OPT_LEVEL FEATURE_BATCH PREFETCH_FACTOR

    sbatch \
      --job-name="$JOB_NAME" \
      --output="${ROOT_DIR}/outputs/files_train/${JOB_NAME}_%A.out" \
      --partition="$PARTITION" \
      --gpus-per-node="$GPUS_PER_NODE" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --time="$TIME_LIMIT" \
      --mem="$MEMORY" \
      --export=ALL \
      "${SCRIPT_DIR}/run_train.job"

    sleep 2 # Stagger submissions slightly
  done
done

echo "🎉 All specified training jobs have been submitted."