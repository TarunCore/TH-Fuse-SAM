import torch
from torch.autograd import Variable 
import utils
import numpy as np
import time
from fusenet import Fusenet
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='SemFuse: Test semantic-guided infrared and visible image fusion')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--vi_path', type=str, default="/content/visible-infrared-wildfire-experiment/m300_grabbed_data_1_51.2/m300_grabbed_data_1_51.2/rgb/108.png", help='Path to visible image')
    parser.add_argument('--ir_path', type=str, default="/content/visible-infrared-wildfire-experiment/m300_grabbed_data_1_51.2/m300_grabbed_data_1_51.2/ir/108.png", help='Path to infrared image')
    parser.add_argument('--output_path', type=str, default="/content/", help='Output directory')
    parser.add_argument('--num_tests', type=int, default=10, help='Number of test iterations')
    return parser.parse_args()

def load_model(path, sam_checkpoint, sam_model_type):
    if not os.path.exists(path):
        raise ValueError('Invalid model path: {}'.format(path))

    # Initialize model with SAM parameters
    fuse_net = Fusenet(sam_checkpoint=sam_checkpoint, sam_model_type=sam_model_type)
    
    # Load state dict
    fuse_net.load_state_dict(torch.load(path))
    
    # Calculate model parameters
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))

    # Set model to evaluation mode and move to GPU
    fuse_net.eval()
    fuse_net.cuda()

    return fuse_net

def generate_fuse_image(model, vi, ir):
    out = model(vi, ir)
    return out

def fuse_test(model, vi_path, ir_path, output_path_root, index):
    if not os.path.exists(vi_path):
        print("Error: {} does not exist".format(vi_path))
        return
    if not os.path.exists(ir_path):
        print("Error: {} does not exist".format(ir_path))
        return
    if not os.path.exists(output_path_root):
        print("Error: {} does not exist".format(output_path_root))
        return

    # Load test images
    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)

    out = utils.get_image(vi_path, height=None, width=None)

    # Move to GPU and create variables
    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)

    # Generate fused image
    with torch.no_grad():
        img_fusion = generate_fuse_image(model, vi_img, ir_img) 

    # Save result
    file_name = 'semfuse_' + str(index) + '.png'
    output_path = output_path_root + file_name

    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    img = img.astype('uint8')
    utils.save_images(output_path, img, out)
    print(f"Saved fused image to: {output_path}")


def main():
    # Parse arguments
    opt = parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    
    with torch.no_grad():
        # Load model
        if not os.path.exists(opt.model_path):
            print("Model file does not exist at: {}".format(opt.model_path))
            return
        
        model = load_model(opt.model_path, opt.sam_checkpoint, opt.sam_model_type) 
        
        # Run tests
        for i in range(opt.num_tests):
            index = i + 1
            visible_path = opt.vi_path
            infrared_path = opt.ir_path
            
            # Check if image files exist
            if not os.path.exists(visible_path):
                print("Visible image does not exist at: {}".format(visible_path))
                return
            if not os.path.exists(infrared_path):
                print("Infrared image does not exist at: {}".format(infrared_path))
                return
            
            # Fuse images
            start = time.time()
            fuse_test(model, visible_path, infrared_path, opt.output_path, index)
            end = time.time()
            print('Fusion time: {:.4f} seconds'.format(end - start))
            
    print('SemFuse testing completed successfully.')

if __name__ == "__main__":
    main()
