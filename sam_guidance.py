import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

class SAMGuidance(nn.Module):
    """
    Class to integrate Segment Anything Model (SAM) for semantic guidance
    in infrared and visible image fusion.
    """
    def __init__(self, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
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
        
    def get_sam_mask(self, vi_img):
        """
        Generate SAM mask for visible image
        Returns semantic segmentation mask
        """
        # Convert to numpy for SAM if needed
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
        if vi_np.max() <= 1.0:
            vi_np = (vi_np * 255).astype(np.uint8)
        else:
            vi_np = vi_np.astype(np.uint8)
        
        # Get masks using SAM in fully automatic mode
        try:
            self.predictor.set_image(vi_np)
            vi_masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True
            )
        except Exception as e:
            print(f"Error processing VI image with SAM: {e}")
            # Create a fallback mask if SAM fails
            h, w = vi_np.shape[:2]
            vi_masks = [np.ones((h, w), dtype=np.float32) * 0.5]
        
        # Combine masks to get the most informative ones
        vi_mask = self.combine_masks(vi_masks)
        
        # Convert back to tensors
        vi_mask_tensor = torch.from_numpy(vi_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        return vi_mask_tensor
    
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
        Generate semantic guidance weights from VI image only
        """
        vi_mask = self.get_sam_mask(vi_img)
        
        # Process mask through guidance network
        vi_guidance = self.process_guidance(vi_mask)
        
        # Use only VI guidance
        semantic_attention = torch.sigmoid(vi_guidance)
        
        return semantic_attention 