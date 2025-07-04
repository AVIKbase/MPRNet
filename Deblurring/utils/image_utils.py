import torch
import numpy as np
import cv2

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    # img is expected to be a NumPy array (H, W, C), BGR for OpenCV
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse) # PSNR for [0, 255] range
    return ps

def load_img_as_tensor(filepath):
    """
    Loads an image from filepath, converts to RGB, normalizes to [0,1],
    transposes to (C, H, W), and converts to a PyTorch tensor.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Image not found at {filepath}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0      # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))        # Change to (C, H, W)
    img = torch.from_numpy(img).float()       # Convert to PyTorch tensor
    return img

def save_img_from_tensor(tensor, filepath):
    """
    Converts a PyTorch tensor (C, H, W, normalized [0,1]) to an image
    and saves it to filepath using OpenCV.
    """
    # Detach from graph, move to CPU, convert to NumPy
    img_np = tensor.detach().cpu().numpy()

    # Transpose back to (H, W, C) from (C, H, W)
    img_np = np.transpose(img_np, (1, 2, 0))

    # Scale to [0, 255] and convert to uint8
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV saving
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath, img_bgr)