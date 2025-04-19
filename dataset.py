import random
import os
import glob

def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_training_data(dataset_path):
    """Get list of image paths for training"""
    ir_dir = os.path.join(dataset_path, 'ir')
    if not os.path.exists(ir_dir):
        raise ValueError(f"IR image directory not found: {ir_dir}")
    
    # Get all IR images
    image_list = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_list.extend(glob.glob(os.path.join(ir_dir, ext)))
    
    # Sort for consistency
    image_list.sort()
    
    return image_list

