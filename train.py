import args
import time
import random
import torch
import torch.nn as nn
import utils
import dataset
from fusenet import Fusenet
from tqdm import tqdm, trange
from torch.optim import Adam
from os.path import join
from loss import final_ssim, TV_Loss
from loss_p import VggDeep,VggShallow
import os
import argparse
from sam_guidance import SAMGuidance
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add command line arguments for SAM model
def parse_args():
    parser = argparse.ArgumentParser(description='SemFuse: Semantic-Aware Infrared and Visible Image Fusion with Transformer Guidance')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM for semantic guidance. If not set, a placeholder attention will be used.')
    parser.add_argument('--semantic_loss_weight', type=float, default=0.3, help='Weight for semantic loss')
    parser.add_argument('--epochs', type=int, default=args.epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=args.batch_size, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=args.learning_rate, help='Learning rate')
    parser.add_argument('--learning_rate_d', type=float, default=args.learning_rate_d, help='Learning rate for discriminator')
    parser.add_argument('--dataset_path', type=str, default=args.dataset_path, help='Dataset path')
    parser.add_argument('--save_model_path', type=str, default=args.save_model_path, help='Path to save model')
    parser.add_argument('--log_interval', type=int, default=args.log_interval, help='Log interval')
    parser.add_argument('--train_num', type=int, default=args.train_num, help='Number of training samples')
    parser.add_argument('--image_height', type=int, default=args.image_height, help='Image height')
    parser.add_argument('--image_width', type=int, default=args.image_width, help='Image width')
    return parser.parse_args()

# Semantic consistency loss
class SemanticConsistencyLoss(nn.Module):
    def __init__(self):
        super(SemanticConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, fused_semantic, ir_semantic, vi_semantic):
        """
        Compute semantic consistency loss between fused image and source images
        """
        # Weighted sum of source semantics (higher weight to more informative regions)
        combined_semantic = torch.max(ir_semantic, vi_semantic)
        
        # Compute loss between fused and combined semantics
        return self.l1_loss(fused_semantic, combined_semantic)

def train(image_lists, opt):
    image_mode = 'L'
    
    # Initialize models
    fusemodel = Fusenet(
        sam_checkpoint=opt.sam_checkpoint, 
        sam_model_type=opt.sam_model_type,
        use_sam=opt.use_sam
    )
    vgg_ir_model = VggDeep() 
    vgg_vi_model = VggShallow()

    # Create SAM guidance for loss computation (only if using SAM)
    if opt.use_sam:
        print("Training with SAM semantic guidance")
        sam_guidance = SAMGuidance(sam_checkpoint=opt.sam_checkpoint, model_type=opt.sam_model_type)
        sam_guidance.cuda()
    else:
        print("Training WITHOUT SAM semantic guidance (using placeholder attention)")
        sam_guidance = None
    
    # Loss functions
    mse_loss = torch.nn.MSELoss() 
    TVLoss = TV_Loss() 
    L1_loss = nn.L1Loss() 
    semantic_loss = SemanticConsistencyLoss()

    # Move models to GPU
    fusemodel.cuda()
    vgg_ir_model.cuda()
    vgg_vi_model.cuda()

    tbar = trange(opt.epochs, ncols=150)
    print('Start training for SemFuse...')

    # Loss tracking
    Loss_model = []
    Loss_ir_feature = [] 
    Loss_vi_feature = [] 
    Loss_semantic = []

    all_ssim_loss = 0
    all_model_loss = 0.
    all_ir_feature_loss = 0.
    all_vi_feature_loss = 0. 
    all_semantic_loss = 0.
    save_num = 0

    for e in tbar:
        print('Epoch %d.....' % e)
        image_set, batches = dataset.load_dataset(image_lists, opt.batch_size)
        fusemodel.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set[batch * opt.batch_size:(batch * opt.batch_size + opt.batch_size)]
            dir1 = "/content/kaist-dataset/set00/V000/visible" 
            dir2 = "/content/kaist-dataset/set00/V000/lwir"
            path1 = []
            path2 = []

            for path in image_paths:
                path1.append(join(dir1, path))
                path2.append(join(dir2, path))

            img_vi = utils.get_train_images_auto(path1, height=opt.image_height, width=opt.image_width, mode=image_mode)
            img_ir = utils.get_train_images_auto(path2, height=opt.image_height, width=opt.image_width, mode=image_mode)

            count += 1

            optimizer_model = Adam(fusemodel.parameters(), opt.learning_rate)
            optimizer_model.zero_grad()

            optimizer_vgg_ir = Adam(vgg_ir_model.parameters(), opt.learning_rate_d)
            optimizer_vgg_ir.zero_grad()

            optimizer_vgg_vi = Adam(vgg_vi_model.parameters(), opt.learning_rate_d)
            optimizer_vgg_vi.zero_grad()

            img_vi = img_vi.cuda()
            img_ir = img_ir.cuda()

            # Forward pass
            outputs = fusemodel(img_vi, img_ir)
            
            # Semantic loss computation (only if using SAM)
            if opt.use_sam and sam_guidance is not None:
                # Get semantic attention maps for loss computation
                semantic_attention = sam_guidance(img_ir, img_vi)
                
                # Resize outputs to match semantic_attention if needed
                resized_outputs = F.interpolate(outputs, size=semantic_attention.shape[2:], mode='bilinear', align_corners=True)
                
                # Generate semantic attention for fused image
                fused_semantic = sam_guidance(resized_outputs, resized_outputs)
                
                # Calculate semantic consistency loss
                semantic_loss_value = semantic_loss(fused_semantic, semantic_attention, semantic_attention)
            else:
                # Set a dummy semantic loss if not using SAM
                semantic_loss_value = torch.tensor(0.0, device=outputs.device)
            
            # Original losses
            ssim_loss_value = 0
            mse_loss_value = 0
            TV_loss_value = 0

            ssim_loss_temp = 1 - final_ssim(img_ir, img_vi, outputs)
            mse_loss_temp = mse_loss(img_ir, outputs) + mse_loss(img_vi, outputs)
            TVLoss_temp = TVLoss(img_ir, outputs) + TVLoss(img_vi, outputs)
            mse_loss_temp = 0
            TVLoss_temp = 0

            ssim_loss_value += ssim_loss_temp
            mse_loss_value += mse_loss_temp
            TV_loss_value += TVLoss_temp

            ssim_loss_value /= len(outputs)
            mse_loss_value /= len(outputs)
            TV_loss_value /= len(outputs)

            # Combined loss with semantic consistency (if using)
            if opt.use_sam:
                model_loss = ssim_loss_value + 0.05 * mse_loss_value + 0.05 * TV_loss_value + opt.semantic_loss_weight * semantic_loss_value
            else:
                model_loss = ssim_loss_value + 0.05 * mse_loss_value + 0.05 * TV_loss_value

            model_loss.backward() 
            optimizer_model.step() 

            # Original VGG losses for IR
            vgg_ir_fuse_out = vgg_ir_model(outputs.detach())[2]
            vgg_ir_out = vgg_ir_model(img_ir)[2]
            per_loss_ir = L1_loss(vgg_ir_fuse_out, vgg_ir_out)
            per_loss_ir_value = 0
            per_loss_ir_temp = per_loss_ir
            per_loss_ir_value += per_loss_ir_temp
            per_loss_ir_value /= len(outputs)
            per_loss_ir_value.backward()
            optimizer_vgg_ir.step()
        
            # Original VGG losses for visible
            vgg_vi_fuse_out = vgg_vi_model(outputs.detach())[0]
            vgg_vi_out = vgg_vi_model(img_vi)[0]
            per_loss_vi = L1_loss(vgg_vi_fuse_out, vgg_vi_out)
            per_loss_vi_value = 0
            per_loss_vi_temp = per_loss_vi
            per_loss_vi_value += per_loss_vi_temp
            per_loss_vi_value /= len(outputs)
            per_loss_vi_value.backward()
            optimizer_vgg_vi.step()
       
            # Track losses
            all_ssim_loss += ssim_loss_value.item()
            all_model_loss = all_ssim_loss
            all_ir_feature_loss += per_loss_ir_value.item()
            all_vi_feature_loss += per_loss_vi_value.item()
            all_semantic_loss += semantic_loss_value.item()

            if (batch + 1) % opt.log_interval == 0:
                if opt.use_sam:
                    mesg = "{}\tEpoch {}:[{}/{}] fusemodel loss: {:.5f} semantic loss: {:.5f}".format(
                        time.ctime(), e + 1, count, batches,
                                      all_model_loss / opt.log_interval,
                                      all_semantic_loss / opt.log_interval)
                else:
                    mesg = "{}\tEpoch {}:[{}/{}] fusemodel loss: {:.5f}".format(
                        time.ctime(), e + 1, count, batches,
                                      all_model_loss / opt.log_interval)
                tbar.set_description(mesg)

                Loss_model.append(all_model_loss / opt.log_interval)
                Loss_ir_feature.append(all_ir_feature_loss / opt.log_interval)
                Loss_vi_feature.append(all_vi_feature_loss / opt.log_interval)
                Loss_semantic.append(all_semantic_loss / opt.log_interval)

                save_num += 1
                all_ssim_loss = 0.
                all_semantic_loss = 0.
                
            if (batch + 1) % (opt.train_num//opt.batch_size) == 0:
                fusemodel.eval()
                fusemodel.cpu()
                save_model_filename = "SemFuse_Epoch_" + str(e) + "_iters_" + str(count) + ("_withSAM" if opt.use_sam else "_noSAM") + ".model"
                save_model_path = os.path.join(opt.save_model_path, save_model_filename)
                torch.save(fusemodel.state_dict(), save_model_path)
                fusemodel.train()
                fusemodel.cuda()
                
    fusemodel.eval()
    fusemodel.cpu()
    save_model_filename = "SemFuse_Final_epoch_" + str(opt.epochs) + ("_withSAM" if opt.use_sam else "_noSAM") + ".model"
    save_model_path = os.path.join(opt.save_model_path, save_model_filename)
    torch.save(fusemodel.state_dict(), save_model_path)

def main():
    # Parse arguments
    opt = parse_args()
    
    # Get training images
    images_path = utils.list_images(opt.dataset_path)
    train_num = opt.train_num 
    images_path = images_path[:train_num]
    random.shuffle(images_path)
    
    # Start training
    train(images_path, opt)

if __name__ == "__main__":
    main()
