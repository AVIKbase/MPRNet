"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from utils.image_utils import load_img_as_tensor, save_img_from_tensor # Assuming these are in utils/image_utils.py or similar

# Assuming 'utils' and 'data_RGB' are available in the same directory or Python path
import utils
from data_RGB import get_test_data
from MPRNet import MPRNet # Assuming MPRNet model definition is in MPRNet.py

# --- CPU-ONLY CHANGES: Define device first ---
device = torch.device('cpu') # Explicitly set device to CPU

parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--input_dir', default='./input_images', type=str, help='Directory of input blurry images')
parser.add_argument('--result_dir', default='./output_images', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_deblurring.pth', type=str, help='Path to pre-trained weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test dataset') # Keep 'GoPro' for expected structure

args = parser.parse_args()

# Create results directory if it doesn't exist
utils.mkdir(args.result_dir)

# Define dataset paths based on arguments
dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'test', 'input') # Path to blurry images

# --- Load pre-trained model ---
model_restoration = MPRNet()
# --- Move model to CPU ---
model_restoration.to(device)

# Load weights, explicitly mapping to CPU
utils.load_checkpoint(model_restoration, args.weights, map_location=device) # Added map_location here
print(f"===>Testing using weights: {args.weights}")

model_restoration.eval()

# --- Iterate through test images ---
# The get_test_data function expects the structure: input_dir/dataset/test/input/
# where input_dir is args.input_dir, and dataset is args.dataset (e.g., GoPro)
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False) # num_workers=0, pin_memory=False for CPU on Windows

for i, data in enumerate(tqdm(test_loader), 0):
    # Move input to CPU
    input_ = data[0].to(device) # Changed .cuda() to .to(device)
    filenames = data[1]

    with torch.no_grad():
        restored = model_restoration(input_)

    # Ensure output is on CPU before saving (if it somehow ended up on GPU)
    restored = restored[0].cpu() # Ensure final output is on CPU
    save_img_from_tensor(restored, os.path.join(args.result_dir, filenames[0]))

print(f"Deblurring complete. Results saved to: {args.result_dir}")