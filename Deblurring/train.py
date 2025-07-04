"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir et al. — https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import random
import time

import utils
from data_RGB import get_training_data, get_validation_data
from MPRNet import MPRNet
import losses

from warmup_scheduler import GradualWarmupScheduler
from pdb import set_trace as stx

# ---------- CPU-ONLY SETUP ----------
device = torch.device('cpu')  # Explicitly use CPU

# ---------- Load config ----------
from config import Config
opt = Config('training.yml')

# ---------- Set seeds for reproducibility ----------
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

# ---------- Directory setup ----------
start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ---------- Fix training & validation data paths ----------
train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

# Absolute path example (uncomment below if needed)
# train_dir = 'D:/image_unblur/MPRNet/Datasets/GoPro/train'
# val_dir = 'D:/image_unblur/MPRNet/Datasets/GoPro/val'

# ---------- Model setup ----------
model_restoration = MPRNet()
model_restoration.to(device)

# ---------- Optimizer ----------
new_lr = opt.OPTIM.LR_INITIAL
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# ---------- Scheduler ----------
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

# ---------- Resume checkpoint if enabled ----------
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest, map_location=device)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

# ---------- Loss functions ----------
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()

# kjhdsgukshfgsh

# ---------- Data loaders ----------
print("Resolved train_dir path:", os.path.abspath(train_dir))
print("Exists:", os.path.exists(train_dir))

train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=False, pin_memory=False)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

# ---------- Training loop ----------
best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0

    model_restoration.train()

    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].to(device)
        input_ = data[1].to(device)

        restored = model_restoration(input_)

        loss_char = torch.sum([criterion_char(restored[j], target) for j in range(len(restored))])
        loss_edge = torch.sum([criterion_edge(restored[j], target) for j in range(len(restored))])
        loss = loss_char + (0.05 * loss_edge)

        loss.backward()
        optimizer.step()         # <-- must come before scheduler.step()
        scheduler.step()         # <-- moved here to suppress warning

        epoch_loss += loss.item()

    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].to(device)
            input_ = data_val[1].to(device)

            with torch.no_grad():
                restored = model_restoration(input_)
            restored = restored[0]

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res.cpu(), tar.cpu()))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, psnr_val_rgb, best_epoch, best_psnr))

        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.2f}s\tLoss: {:.4f}\tLearningRate {:.6f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]))
    print("------------------------------------------------------------------")

    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))
