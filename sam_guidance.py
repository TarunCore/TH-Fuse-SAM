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
        
    def get_sam_masks(self, ir_img, vi_img):
        """
        Generate SAM masks for both infrared and visible images
        Returns semantic segmentation masks
        """
        # Convert to numpy for SAM if needed
        if isinstance(ir_img, torch.Tensor):
            ir_np = ir_img.detach().cpu().numpy().squeeze()
            if len(ir_np.shape) == 2:  # If grayscale, repeat to make 3 channels
                ir_np = np.repeat(ir_np[..., np.newaxis], 3, axis=2)
            elif ir_np.shape[0] == 1:  # If [1, H, W]
                ir_np = np.repeat(ir_np, 3, axis=0)
                ir_np = np.transpose(ir_np, (1, 2, 0))
        
        if isinstance(vi_img, torch.Tensor):
            vi_np = vi_img.detach().cpu().numpy().squeeze()
            if len(vi_np.shape) == 2:  # If grayscale, repeat to make 3 channels
                vi_np = np.repeat(vi_np[..., np.newaxis], 3, axis=2)
            elif vi_np.shape[0] == 1:  # If [1, H, W]
                vi_np = np.repeat(vi_np, 3, axis=0)
                vi_np = np.transpose(vi_np, (1, 2, 0))
        
        # Normalize if needed
        if ir_np.max() <= 1.0:
            ir_np = (ir_np * 255).astype(np.uint8)
        else:
            ir_np = ir_np.astype(np.uint8)
            
        if vi_np.max() <= 1.0:
            vi_np = (vi_np * 255).astype(np.uint8)
        else:
            vi_np = vi_np.astype(np.uint8)
            
        # Get masks using SAM in fully automatic mode
        self.predictor.set_image(ir_np)
        ir_masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True
        )
        
        self.predictor.set_image(vi_np)
        vi_masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True
        )
        
        # Combine masks to get the most informative ones
        ir_mask = self.combine_masks(ir_masks)
        vi_mask = self.combine_masks(vi_masks)
        
        # Convert back to tensors
        ir_mask_tensor = torch.from_numpy(ir_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        vi_mask_tensor = torch.from_numpy(vi_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        return ir_mask_tensor, vi_mask_tensor
    
    def combine_masks(self, masks):
        """
        Combine multiple masks into a single mask
        Strategy: Take the union of all masks
        """
        if len(masks) == 0:
            return np.zeros((256, 256))
        
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