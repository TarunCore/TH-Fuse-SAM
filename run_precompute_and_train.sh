#!/bin/bash
echo "TH-Fuse-SAM with precomputed visible masks"

# Define parameters
SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE="vit_h"
DATASET_PATH="/content/kaist-dataset/set00/V000/"
PRECOMPUTED_MASKS_DIR="precomputed_masks"

# Step 1: Precompute masks for visible images
echo "Precomputing masks for visible images..."
python precompute_masks.py \
  --sam_checkpoint $SAM_CHECKPOINT \
  --sam_model_type $SAM_MODEL_TYPE \
  --dataset_path $DATASET_PATH \
  --output_dir $PRECOMPUTED_MASKS_DIR

# Step 2: Train using precomputed masks
echo "Starting training with precomputed masks..."
python train.py \
  --use_sam \
  --sam_checkpoint $SAM_CHECKPOINT \
  --sam_model_type $SAM_MODEL_TYPE \
  --precomputed_masks_dir $PRECOMPUTED_MASKS_DIR \
  --dataset_path $DATASET_PATH \
  --epochs 100

echo "Done!" 