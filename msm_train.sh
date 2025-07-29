
# Run training
export PYTHONPATH=$PYTHONPATH:"$ROOT_DIR"
python spai/main_spatial_masking.py \
    --cfg configs/spai.yaml \
    --data-path datasets/chameleon_dataset_split.csv \
    --output output/image_masking \
    --tag slurm_exp \
    --batch-size 8 \
    --launcher slurm \
    --amp-opt-level "O0" \