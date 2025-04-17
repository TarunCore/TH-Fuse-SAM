import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_sam_model(checkpoint_path, model_type="vit_h", device="cuda"):
    """Loads the SAM model."""
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam

def generate_semantic_masks(image_rgb, sam_model, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95, box_nms_thresh=0.7):
    """Generates semantic masks for an input image using SAM."""
    # SAM expects a BGR uint8 image, but input here might be float tensor [0,1] or [0,255]
    # Assuming input is a PyTorch tensor (C, H, W) in range [0, 1]
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).byte()

    # Convert to NumPy array (H, W, C)
    image_np = image_rgb.permute(1, 2, 0).cpu().numpy()

    if image_np.shape[2] == 1: # Handle grayscale images by repeating channels
        image_np = np.repeat(image_np, 3, axis=2)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        # crop_n_layers=0, # Keep default
        # crop_n_points_downscale_factor=1, # Keep default
        box_nms_thresh=box_nms_thresh, # Added for potential overlap reduction
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    masks = mask_generator.generate(image_np)

    # Combine masks into a single semantic map (e.g., by taking the max over overlapping masks)
    # Or return all masks for more complex processing
    if masks:
        # Sort masks by area (optional, might prioritize larger objects)
        # sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        # For simplicity, create a binary map where any segmentation exists
        combined_mask = np.any(np.stack([m['segmentation'] for m in masks], axis=0), axis=0)
        semantic_map = torch.from_numpy(combined_mask).unsqueeze(0).float().to(sam_model.device) # Add channel dim
    else:
        semantic_map = torch.zeros((1, image_np.shape[0], image_np.shape[1]), dtype=torch.float, device=sam_model.device)

    return semantic_map # Returns a single-channel [1, H, W] tensor 