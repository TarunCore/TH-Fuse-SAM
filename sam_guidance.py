import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import os
import time

class SAMGuidance(nn.Module):
    """
    Class to integrate Segment Anything Model (SAM) for semantic guidance
    in infrared and visible image fusion.
    """
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda", 
                 cache_dir=None, downsample_factor=2, use_cache=True):
        super(SAMGuidance, self).__init__()
        
        # Load SAM model
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        
        # Guidance network
        self.guidance_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.guidance_bn1 = nn.BatchNorm2d(32)
        self.guidance_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.guidance_bn2 = nn.BatchNorm2d(64)
        self.guidance_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.guidance_bn3 = nn.BatchNorm2d(64)
        
        # Caching system
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Image fingerprinting for cache lookup
        self.cached_fingerprints = {}
        self.cached_masks = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Downsample factor to reduce computation time
        self.downsample_factor = downsample_factor
        
    def get_image_fingerprint(self, img):
        """
        Create a simple fingerprint for an image to use as cache key
        """
        if isinstance(img, torch.Tensor):
            # Use mean and std as a simple fingerprint
            if img.dim() == 4:  # Batch format
                # Take the first image in batch
                img_flat = img[0].flatten()
                return f"{img_flat.mean().item():.4f}_{img_flat.std().item():.4f}_{img.shape[2]}_{img.shape[3]}"
            else:
                img_flat = img.flatten()
                return f"{img_flat.mean().item():.4f}_{img_flat.std().item():.4f}_{img.shape[1]}_{img.shape[2]}"
        return None
    
    def get_sam_masks(self, ir_img, vi_img):
        """
        Generate SAM masks for both infrared and visible images
        Returns semantic segmentation masks
        """
        # Check cache first if enabled
        ir_fingerprint = self.get_image_fingerprint(ir_img) if self.use_cache else None
        vi_fingerprint = self.get_image_fingerprint(vi_img) if self.use_cache else None
        
        # Try to retrieve from cache
        ir_mask_tensor = None
        vi_mask_tensor = None
        
        if self.use_cache and ir_fingerprint in self.cached_masks and vi_fingerprint in self.cached_masks:
            self.cache_hits += 1
            ir_mask_tensor = self.cached_masks[ir_fingerprint]
            vi_mask_tensor = self.cached_masks[vi_fingerprint]
            return ir_mask_tensor, vi_mask_tensor
        
        self.cache_misses += 1
        
        # Convert to numpy for SAM if needed
        if isinstance(ir_img, torch.Tensor):
            # Handle single-channel grayscale images (B, 1, H, W)
            if ir_img.dim() == 4 and ir_img.shape[1] == 1:
                # Convert to (B, H, W) then to (H, W, 3) for SAM
                ir_np = ir_img.squeeze(1).detach().cpu().numpy()[0]  # Take the first batch item
                ir_np = np.repeat(ir_np[:, :, np.newaxis], 3, axis=2)  # (H, W) -> (H, W, 3)
            else:
                # Use a simple fallback to generate a placeholder mask
                print("Warning: Unexpected IR image format. Using placeholder mask.")
                h, w = ir_img.shape[-2], ir_img.shape[-1]
                ir_np = np.ones((h, w, 3), dtype=np.uint8) * 128
        
        if isinstance(vi_img, torch.Tensor):
            # Handle single-channel grayscale images (B, 1, H, W)
            if vi_img.dim() == 4 and vi_img.shape[1] == 1:
                # Convert to (B, H, W) then to (H, W, 3) for SAM
                vi_np = vi_img.squeeze(1).detach().cpu().numpy()[0]  # Take the first batch item
                vi_np = np.repeat(vi_np[:, :, np.newaxis], 3, axis=2)  # (H, W) -> (H, W, 3)
            else:
                # Use a simple fallback to generate a placeholder mask
                print("Warning: Unexpected VI image format. Using placeholder mask.")
                h, w = vi_img.shape[-2], vi_img.shape[-1]
                vi_np = np.ones((h, w, 3), dtype=np.uint8) * 128
        
        # Normalize if needed
        if ir_np.max() <= 1.0:
            ir_np = (ir_np * 255).astype(np.uint8)
        else:
            ir_np = ir_np.astype(np.uint8)
            
        if vi_np.max() <= 1.0:
            vi_np = (vi_np * 255).astype(np.uint8)
        else:
            vi_np = vi_np.astype(np.uint8)
        
        # Downsample images for faster processing
        if self.downsample_factor > 1:
            original_shape = ir_np.shape[:2]
            new_h, new_w = original_shape[0] // self.downsample_factor, original_shape[1] // self.downsample_factor
            ir_np_small = np.array(torch.nn.functional.interpolate(torch.from_numpy(ir_np).permute(2, 0, 1).unsqueeze(0), 
                                                                   size=(new_h, new_w), 
                                                                   mode='bilinear', 
                                                                   align_corners=False)[0].permute(1, 2, 0).numpy(), 
                                    dtype=np.uint8)
            vi_np_small = np.array(torch.nn.functional.interpolate(torch.from_numpy(vi_np).permute(2, 0, 1).unsqueeze(0), 
                                                                   size=(new_h, new_w), 
                                                                   mode='bilinear', 
                                                                   align_corners=False)[0].permute(1, 2, 0).numpy(), 
                                    dtype=np.uint8)
        else:
            ir_np_small = ir_np
            vi_np_small = vi_np
            original_shape = ir_np.shape[:2]
            
        # Get masks using SAM in fully automatic mode
        try:
            start_time = time.time()
            self.predictor.set_image(ir_np_small)
            ir_masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True
            )
            if self.downsample_factor > 1:
                # Upsample masks back to original size
                ir_masks_list = []
                for mask in ir_masks:
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
                    upsampled_mask = F.interpolate(mask_tensor, size=original_shape, mode='bilinear', align_corners=False)
                    ir_masks_list.append(upsampled_mask[0, 0].numpy())
                ir_masks = np.stack(ir_masks_list)
            ir_time = time.time() - start_time
            
            start_time = time.time()
            self.predictor.set_image(vi_np_small)
            vi_masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True
            )
            if self.downsample_factor > 1:
                # Upsample masks back to original size
                vi_masks_list = []
                for mask in vi_masks:
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
                    upsampled_mask = F.interpolate(mask_tensor, size=original_shape, mode='bilinear', align_corners=False)
                    vi_masks_list.append(upsampled_mask[0, 0].numpy())
                vi_masks = np.stack(vi_masks_list)
            vi_time = time.time() - start_time
            
            print(f"SAM processing times - IR: {ir_time:.2f}s, VI: {vi_time:.2f}s")
            
        except Exception as e:
            print(f"Error processing with SAM: {e}")
            # Create fallback masks if SAM fails
            h, w = ir_np.shape[:2]
            ir_masks = [np.ones((h, w), dtype=np.float32) * 0.5]
            vi_masks = [np.ones((h, w), dtype=np.float32) * 0.5]
        
        # Combine masks to get the most informative ones
        ir_mask = self.combine_masks(ir_masks)
        vi_mask = self.combine_masks(vi_masks)
        
        # Convert to tensors
        ir_mask_tensor = torch.from_numpy(ir_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vi_mask_tensor = torch.from_numpy(vi_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Cache the results if enabled
        if self.use_cache:
            self.cached_masks[ir_fingerprint] = ir_mask_tensor
            self.cached_masks[vi_fingerprint] = vi_mask_tensor
            
            # Print cache statistics occasionally
            if (self.cache_hits + self.cache_misses) % 100 == 0:
                hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100
                print(f"SAM cache stats - Hits: {self.cache_hits}, Misses: {self.cache_misses}, Hit rate: {hit_rate:.1f}%")
        
        return ir_mask_tensor, vi_mask_tensor
    
    def combine_masks(self, masks):
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
    
    def process_guidance(self, mask):
        """
        Process the mask through the guidance network
        """
        x = F.relu(self.guidance_bn1(self.guidance_conv1(mask)))
        x = F.relu(self.guidance_bn2(self.guidance_conv2(x)))
        x = self.guidance_bn3(self.guidance_conv3(x))
        return x
    
    def forward(self, ir_img, vi_img):
        """
        Generate semantic guidance weights from IR and VI images
        """
        ir_mask, vi_mask = self.get_sam_masks(ir_img, vi_img)
        
        # Process masks through guidance network
        ir_guidance = self.process_guidance(ir_mask)
        vi_guidance = self.process_guidance(vi_mask)
        
        # Combine guidance from both modalities
        combined_guidance = ir_guidance + vi_guidance
        
        # Generate attention map (range 0-1)
        semantic_attention = torch.sigmoid(combined_guidance)
        
        return semantic_attention 