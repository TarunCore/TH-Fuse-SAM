import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from dataset import get_training_data
import utils
from PIL import Image
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Precompute SAM masks for faster training')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', 
                        help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', 
                        choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--dataset_path', type=str, default='dataset/train', 
                        help='Path to training dataset')
    parser.add_argument('--output_dir', type=str, default='precomputed_masks', 
                        help='Directory to save precomputed masks')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()

def combine_masks(masks):
    """
    Combine multiple masks into a single mask
    Strategy: Take the union of all masks
    """
    if len(masks) == 0:
        # Return a placeholder mask if no masks are available
        return np.ones((256, 256), dtype=np.float32) * 0.5
    
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask).astype(np.float32)
        
    return combined_mask

def get_sam_mask(predictor, image_np):
    """Generate SAM mask for an image"""
    # Normalize if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Get masks using SAM in fully automatic mode
    try:
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True
        )
        # Combine masks
        return combine_masks(masks)
    except Exception as e:
        print(f"Error processing image with SAM: {e}")
        # Create a fallback mask if SAM fails
        h, w = image_np.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Only create vi directory since we're only precomputing visible masks
    os.makedirs(os.path.join(args.output_dir, 'vi'), exist_ok=True)
    
    # Load SAM model
    print(f"Loading SAM model ({args.sam_model_type}) from {args.sam_checkpoint}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    
    # Get dataset paths (IR images)
    ir_image_list = get_training_data(args.dataset_path)
    print(f"Found {len(ir_image_list)} IR images in dataset")
    
    # Process each visible image corresponding to an IR image
    for ir_path in tqdm(ir_image_list):
        # Extract filename without extension
        filename = os.path.basename(ir_path).split('.')[0]
        
        # Get corresponding visible image path
        vi_path = ir_path.replace('/lwir/', '/visible/')
        if not os.path.exists(vi_path):
            print(f"Warning: Corresponding visible image not found: {vi_path}")
            continue
        
        # Load visible image only
        vi_img, _ = utils.get_img(vi_path, 'L')
        
        # Convert to numpy arrays in format expected by SAM (H, W, 3)
        vi_np = np.array(vi_img)
        
        # Ensure it's 3-channel for SAM
        if len(vi_np.shape) == 2:
            vi_np = np.repeat(vi_np[:, :, np.newaxis], 3, axis=2)
        
        # Generate mask for visible image only
        vi_mask = get_sam_mask(predictor, vi_np)
        
        # Save visible image mask only
        vi_mask_path = os.path.join(args.output_dir, 'vi', f"{filename}.npy")
        np.save(vi_mask_path, vi_mask)
    
    print(f"Precomputed visible image masks saved to {args.output_dir}/vi")

if __name__ == "__main__":
    main() 