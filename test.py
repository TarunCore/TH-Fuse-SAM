import torch
from torch.autograd import Variable 
import utils
import numpy as np
import time
from fusenet import Fusenet
import os
import argparse
import os.path as osp
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='SemFuse: Test Semantic-Aware Infrared and Visible Image Fusion with Transformer Guidance')
    parser.add_argument('--model_path', type=str, default='models/model_epoch100.pth', help='Path to pretrained model')
    parser.add_argument('--ir_dir', type=str, default='./test_imgs/ir', help='Path to test infrared images directory')
    parser.add_argument('--vi_dir', type=str, default='./test_imgs/vi', help='Path to test visible images directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Path to save fusion results')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM for semantic guidance. If not set, a placeholder attention will be used.')
    parser.add_argument('--precomputed_masks_dir', type=str, default=None, help='Directory containing precomputed SAM masks (to speed up inference)')
    return parser.parse_args()

def load_model(path, sam_checkpoint, sam_model_type, use_sam, precomputed_masks_dir=None):
    if not os.path.exists(path):
        raise ValueError('Invalid model path: {}'.format(path))

    # Initialize model with SAM parameters
    fuse_net = Fusenet(
        sam_checkpoint=sam_checkpoint, 
        sam_model_type=sam_model_type, 
        use_sam=use_sam
    )
    
    # Load state dict
    fuse_net.load_state_dict(torch.load(path))
    
    # Calculate model parameters
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))
    print(f"Using SAM guidance: {'Yes' if use_sam else 'No'}")
    print(f"Using precomputed masks: {'Yes - ' + precomputed_masks_dir if precomputed_masks_dir else 'No'}")

    # Set model to evaluation mode and move to GPU
    fuse_net.eval()
    fuse_net.cuda()
    
    # If using precomputed masks, update the SAM guidance module
    if use_sam and fuse_net.sam_guidance is not None and precomputed_masks_dir is not None:
        fuse_net.sam_guidance.precomputed_masks_dir = precomputed_masks_dir
        print(f"Updated SAM guidance to use precomputed masks from: {precomputed_masks_dir}")

    return fuse_net

def test_main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(
        args.model_path, 
        args.sam_checkpoint, 
        args.sam_model_type, 
        args.use_sam,
        args.precomputed_masks_dir
    )
    
    # Load test images
    print(f"Loading test images from:")
    print(f"  IR: {args.ir_dir}")
    print(f"  VI: {args.vi_dir}")
    
    # Get all image files from directories
    ir_files = [f for f in os.listdir(args.ir_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ir_files.sort()
    
    # Process each image pair
    for filename in ir_files:
        print(f"Processing {filename}")
        
        # Construct full file paths
        ir_path = os.path.join(args.ir_dir, filename)
        vi_path = os.path.join(args.vi_dir, filename)
        
        if not os.path.exists(vi_path):
            print(f"Corresponding visible image not found: {vi_path}")
            continue
        
        # Load images
        ir_tensor = utils.get_test_images(ir_path, mode='L')
        vi_tensor = utils.get_test_images(vi_path, mode='L')
        
        # Move to GPU
        ir_tensor = ir_tensor.cuda()
        vi_tensor = vi_tensor.cuda()
        
        # Fusion
        with torch.no_grad():
            fused = model(vi_tensor, ir_tensor)
            fused_image = fused.squeeze().cpu().numpy()
        
        # Save result
        output_path = os.path.join(args.output_dir, f"fused_{filename}")
        cv2.imwrite(output_path, fused_image * 255)
        print(f"Saved result to {output_path}")
    
    print(f"Fusion complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    test_main()
