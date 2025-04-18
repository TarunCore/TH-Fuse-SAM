#!/bin/bash

# Run training with optimized SAM parameters
# This script shows different optimization options for speeding up training with SAM

# Option 1: Run with all optimizations enabled (recommended for fastest training)
echo "Starting optimized training with SAM..."
python train.py \
    --use_sam \
    --sam_checkpoint sam_vit_h_4b8939.pth \
    --sam_model_type vit_h \
    --sam_downsample_factor 4 \
    --use_sam_cache \
    --sam_cache_dir sam_cache \
    --sam_update_freq 5 \
    --semantic_loss_weight 0.3

# Option 2: Run with moderate optimizations (balancing speed and accuracy)
# echo "Starting training with moderate SAM optimizations..."
# python train.py \
#     --use_sam \
#     --sam_checkpoint sam_vit_h_4b8939.pth \
#     --sam_model_type vit_h \
#     --sam_downsample_factor 2 \
#     --use_sam_cache \
#     --sam_update_freq 2 \
#     --semantic_loss_weight 0.3

# Option 3: Run with no SAM (for comparison)
# echo "Starting training without SAM for comparison..."
# python train.py

# Explanation of parameters:
# --use_sam: Enable SAM-based semantic guidance
# --sam_checkpoint: Path to SAM model checkpoint (download from Meta AI)
# --sam_model_type: SAM model type (vit_h is highest quality but slowest, vit_b is fastest)
# --sam_downsample_factor: How much to downsample images for SAM (2-4x recommended)
# --use_sam_cache: Enable caching of SAM predictions (saves recomputing on similar images)
# --sam_update_freq: How often to run SAM (1=every batch, 5=every 5 batches)
# --semantic_loss_weight: Weight for the semantic consistency loss 