@echo off
echo TH-Fuse-SAM with precomputed visible masks

REM Define parameters
set SAM_CHECKPOINT=sam_vit_h_4b8939.pth
set SAM_MODEL_TYPE=vit_h
set DATASET_PATH=dataset/train
set PRECOMPUTED_MASKS_DIR=precomputed_masks

REM Step 1: Precompute masks for visible images
echo Precomputing masks for visible images...
python precompute_masks.py ^
  --sam_checkpoint %SAM_CHECKPOINT% ^
  --sam_model_type %SAM_MODEL_TYPE% ^
  --dataset_path %DATASET_PATH% ^
  --output_dir %PRECOMPUTED_MASKS_DIR%

REM Step 2: Train using precomputed masks
echo Starting training with precomputed masks...
python train.py ^
  --use_sam ^
  --sam_checkpoint %SAM_CHECKPOINT% ^
  --sam_model_type %SAM_MODEL_TYPE% ^
  --precomputed_masks_dir %PRECOMPUTED_MASKS_DIR% ^
  --dataset_path %DATASET_PATH% ^
  --epochs 100

echo Done! 