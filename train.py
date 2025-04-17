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
# --- SemFuse Imports ---
from segmentation import load_sam_model, generate_semantic_masks
# -----------------------

os.environ["CUDA_VISIBLE_DEVICES"] = '6' # Consider making this configurable or removing hardcoding

def train(image_lists):

    # --- SemFuse Setup ---
    print("Loading SAM model...")
    # Determine device (use cuda if available, otherwise cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = load_sam_model(args.sam_checkpoint_path, args.sam_model_type, device=device)
    print("SAM model loaded.")
    # ---------------------

    image_mode = 'L' # Assuming grayscale inputs
    fusemodel = Fusenet()
    vgg_ir_model = VggDeep()
    vgg_vi_model = VggShallow()

    mse_loss = torch.nn.MSELoss()
    TVLoss = TV_Loss()
    L1_loss = nn.L1Loss()

    # --- Use determined device ---
    fusemodel.to(device)
    vgg_ir_model.to(device)
    vgg_vi_model.to(device)
    # ---------------------------

    tbar = trange(args.epochs, ncols=150)
    print('Start training.....')

    Loss_model = []
    Loss_ir_feature = [] 
    Loss_vi_feature = [] 

    all_ssim_loss = 0
    all_model_loss = 0.
    all_ir_feature_loss = 0.
    all_vi_feature_loss = 0. 
    save_num = 0

    for e in tbar:
        print('Epoch %d.....' % e)
        image_set, batches = dataset.load_dataset(image_lists, args.batch_size)
        fusemodel.train()
        count = 0
        for batch in range(batches):
            image_paths = image_set[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]
            # Make these paths configurable via args.py
            dir1 = "/path/to/your/own/dataset/vi" # <--- UPDATE THIS PATH
            dir2 = "/path/to/your/own/dataset/ir" # <--- UPDATE THIS PATH
            path1 = []
            path2 = []

            for path in image_paths:
                path1.append(join(dir1, path))
                path2.append(join(dir2, path))

            # Load images (ensure they are tensors C, H, W on the correct device)
            img_vi = utils.get_train_images_auto(path1, height=args.image_height, width=args.image_width, mode=image_mode).to(device)
            img_ir = utils.get_train_images_auto(path2, height=args.image_height, width=args.image_width, mode=image_mode).to(device)

            # --- SemFuse: Generate Semantic Maps ---
            # Disable gradients for SAM inference
            with torch.no_grad():
                sem_maps_vi = []
                sem_maps_ir = []
                for i in range(img_vi.shape[0]): # Iterate through batch
                    # SAM expects 3 channels, repeat grayscale if needed (handled in generate_semantic_masks)
                    sem_map_vi = generate_semantic_masks(img_vi[i], sam_model)
                    sem_map_ir = generate_semantic_masks(img_ir[i], sam_model)
                    sem_maps_vi.append(sem_map_vi)
                    sem_maps_ir.append(sem_map_ir)
                # Stack maps into a batch tensor
                sem_maps_vi_batch = torch.stack(sem_maps_vi).to(device)
                sem_maps_ir_batch = torch.stack(sem_maps_ir).to(device)
            # -------------------------------------

            count += 1

            optimizer_model = Adam(fusemodel.parameters(), args.learning_rate)
            optimizer_model.zero_grad()

            optimizer_vgg_ir = Adam(vgg_ir_model.parameters(), args.learning_rate_d)
            optimizer_vgg_ir.zero_grad()

            optimizer_vgg_vi = Adam(vgg_vi_model.parameters(), args.learning_rate_d)
            optimizer_vgg_vi.zero_grad()

            # --- Pass semantic maps to the model ---
            outputs = fusemodel(img_vi, img_ir, sem_maps_vi_batch, sem_maps_ir_batch)
            # ---------------------------------------

            ssim_loss_value = 0
            mse_loss_value = 0
            TV_loss_value = 0

            ssim_loss_temp = 1 - final_ssim(img_ir, img_vi, outputs)
            mse_loss_temp = mse_loss(img_ir,outputs) + mse_loss(img_vi,outputs)
            TVLoss_temp = TVLoss(img_ir,outputs)+TVLoss(img_vi,outputs)
            mse_loss_temp = 0
            TVLoss_temp = 0

            ssim_loss_value += ssim_loss_temp
            mse_loss_value += mse_loss_temp
            TV_loss_value +=TVLoss_temp

            ssim_loss_value /= len(outputs)
            mse_loss_value /= len(outputs)
            TV_loss_value /= len(outputs)

            model_loss = ssim_loss_value + 0.05 * mse_loss_value + 0.05 *  TV_loss_value
            model_loss.backward() 
            optimizer_model.step() 

        
            vgg_ir_fuse_out = vgg_ir_model(outputs.detach())[2]
            vgg_ir_out = vgg_ir_model(img_ir)[2]
            per_loss_ir = L1_loss(vgg_ir_fuse_out, vgg_ir_out)
            per_loss_ir_value = 0
            per_loss_ir_temp = per_loss_ir
            per_loss_ir_value += per_loss_ir_temp
            per_loss_ir_value /= len(outputs)
            per_loss_ir_value.backward()
            optimizer_vgg_ir.step()
        

        
            vgg_vi_fuse_out = vgg_vi_model(outputs.detach())[0]
            vgg_vi_out = vgg_vi_model(img_vi)[0]
            per_loss_vi = L1_loss(vgg_vi_fuse_out, vgg_vi_out)
            per_loss_vi_value = 0
            per_loss_vi_temp = per_loss_vi
            per_loss_vi_value += per_loss_vi_temp
            per_loss_vi_value /= len(outputs)
            per_loss_vi_value.backward()
            optimizer_vgg_vi.step()
       

            all_ssim_loss += ssim_loss_value.item()
            all_model_loss = all_ssim_loss
            all_ir_feature_loss += per_loss_ir_value.item()
            all_vi_feature_loss += per_loss_vi_value.item()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:[{}/{}] fusemodel loss: {:.5f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_model_loss / args.log_interval)
                tbar.set_description(mesg)

                Loss_model.append(all_model_loss / args.log_interval)
                Loss_ir_feature.append(all_ir_feature_loss / args.log_interval)
                Loss_vi_feature.append(all_vi_feature_loss / args.log_interval)

                save_num += 1
                all_ssim_loss = 0.
                
            if (batch + 1) % (args.train_num//args.batch_size) == 0:
                # --- Save model handling ---
                fusemodel.eval()
                # No need to move to CPU if saving state_dict
                # fusemodel.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + ".model"
                save_model_path_full = os.path.join(args.save_model_path, save_model_filename) # Renamed variable
                torch.save(fusemodel.state_dict(), save_model_path_full)
                fusemodel.train()
                # No need to move back to cuda if it was never moved
                # fusemodel.cuda()
                print(f"\nModel saved to {save_model_path_full}") # Added print statement
                # --------------------------

    # --- Final model save ---
    fusemodel.eval()
    # fusemodel.cpu()
    save_model_filename = "Final_epoch_" + str(args.epochs) + ".model"
    save_model_path_final = os.path.join(args.save_model_path, save_model_filename) # Renamed variable
    torch.save(fusemodel.state_dict(), save_model_path_final)
    print(f"\nFinal model saved to {save_model_path_final}")
    # -----------------------

def main():
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
        print(f"Created directory: {args.save_model_path}")

    print()
    # Validate dataset path
    vis_dir = os.path.join(args.dataset_path, 'vi') # Assuming subdirs 'vi' and 'ir'
    ir_dir = os.path.join(args.dataset_path, 'ir')
    if not os.path.exists(vis_dir) or not os.path.exists(ir_dir):
         print(f"Error: Dataset path '{args.dataset_path}' not found or doesn't contain 'vi' and 'ir' subdirectories.")
         print("Please update 'dataset_path' in args.py")
         return # Exit if dataset path is invalid

    # List images from one directory (assuming paired images have same names)
    images_path = utils.list_images(vis_dir)
    if not images_path:
        print(f"Error: No images found in {vis_dir}")
        return

    train_num = min(args.train_num, len(images_path)) # Adjust train_num if dataset is smaller
    print(f"Using {train_num} images for training.")
    images_path = images_path[:train_num]
    random.shuffle(images_path)
    train(images_path)

if __name__ == "__main__":
    main()
