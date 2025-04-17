epochs = 36  
batch_size = 2  
image_height = 256 
image_width = 256
train_num = 7601  
learning_rate = 1e-4
learning_rate_d = 1e-4
log_interval = 10
dataset_path = "/path/to/your/own/dataset" 
save_model_path = "/path/to/your/own/outputs"  

# --- SAM Arguments ---
sam_checkpoint_path = "/path/to/your/sam_vit_h_4b8939.pth" # <--- UPDATE THIS PATH
sam_model_type = "vit_h" # Or "vit_l", "vit_b"
