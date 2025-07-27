#!/bin/bash

# Download weights job launcher
echo "🚀 Submitting weight download job..."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Submit the download job
sbatch "${SCRIPT_DIR}/download_weights.job"

echo "📝 Check job status with: squeue -u $(whoami)"
echo "📋 View output with: tail -f ${ROOT_DIR}/outputs/download_weights_*.out"
