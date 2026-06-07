"""
D3Net Fine-Tuning Script for SIDL Dirty Lens Dataset.

Single-GPU training script (no DDP required).
Uses SIDL degraded-clean pairs with the D3Net residual learning approach:
  output = input + D3Net(input)

Usage:
  python train_sidl.py \
    --train_dir ./data/SIDL/train \
    --val_dir ./data/SIDL/val \
    --n_epochs 100 \
    --batch_size 4 \
    --img_size 128 \
    --lr 0.0001 \
    --model_folder ./ckpt/sidl_finetune
"""

import argparse
import datetime
import os
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import setproctitle
setproctitle.setproctitle("D3Net_SIDL")

from models.base_model import D3Net
from utils.sidl_dataloader import SIDLTrainDataset, SIDLValDataset, denormalize, mean, std
from utils.metrics import calc_ssim
from tqdm import tqdm


def calc_psnr_tensor(output, target):
    """Calculate PSNR between two tensors in [0, 1] range."""
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0).to(output.device) / torch.sqrt(mse))


def validate(model, val_loader, device, img_size):
    """Run validation and return average PSNR and SSIM with breakdown by difficulty."""
    model.eval()
    from collections import defaultdict
    results = defaultdict(lambda: defaultdict(lambda: {"psnr": [], "ssim": []}))

    pbar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            imgs_input = batch["input"].to(device)
            imgs_target = batch["gt"].to(device)

            dec_info = model(imgs_input)
            output = torch.add(imgs_input, dec_info)

            # Denormalize to [0, 1] range for metrics
            output_dn = output.clone()
            target_dn = imgs_target.clone()
            for c in range(3):
                output_dn[:, c].mul_(std[c]).add_(mean[c])
                target_dn[:, c].mul_(std[c]).add_(mean[c])
            output_dn = torch.clamp(output_dn, 0, 1)
            target_dn = torch.clamp(target_dn, 0, 1)

            # Per-image metrics
            for j in range(output_dn.size(0)):
                psnr_val = calc_psnr_tensor(output_dn[j], target_dn[j])
                ssim_val = calc_ssim(output_dn[j].unsqueeze(0), target_dn[j].unsqueeze(0))
                
                ctype = batch["type"][j]
                difficulty = batch["difficulty"][j]
                
                psnr_num = psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val
                ssim_num = ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val
                
                results[ctype][difficulty]['psnr'].append(psnr_num)
                results[ctype][difficulty]['ssim'].append(ssim_num)

    model.train()

    # Generate and print table
    types = sorted(list(results.keys()))
    difficulties = ['easy', 'medium', 'hard']
    
    header = f"{'Difficulty':<12} | "
    for t in types:
        header += f"{t.capitalize():^15} "
    header += f"| {'Average':^15}"
    
    border = "-" * len(header)
    print("\n" + border)
    print(" Validation Results Breakdown (PSNR / SSIM)")
    print(border)
    print(header)
    print(border)
    
    diff_totals = {d: {'psnr': [], 'ssim': []} for d in difficulties}
    type_totals = {t: {'psnr': [], 'ssim': []} for t in types}
    overall_total = {'psnr': [], 'ssim': []}
    
    for d in difficulties:
        row_str = f"{d.capitalize():<12} | "
        for t in types:
            vals = results[t][d]
            if len(vals['psnr']) > 0:
                avg_p = sum(vals['psnr']) / len(vals['psnr'])
                avg_s = sum(vals['ssim']) / len(vals['ssim'])
                cell = f"{avg_p:.2f}/{avg_s:.4f}"
                
                diff_totals[d]['psnr'].append(avg_p)
                diff_totals[d]['ssim'].append(avg_s)
                type_totals[t]['psnr'].extend(vals['psnr'])
                type_totals[t]['ssim'].extend(vals['ssim'])
                overall_total['psnr'].extend(vals['psnr'])
                overall_total['ssim'].extend(vals['ssim'])
            else:
                cell = "N/A"
            row_str += f"{cell:^15} "
            
        if len(diff_totals[d]['psnr']) > 0:
            avg_p = sum(diff_totals[d]['psnr']) / len(diff_totals[d]['psnr'])
            avg_s = sum(diff_totals[d]['ssim']) / len(diff_totals[d]['ssim'])
            cell = f"{avg_p:.2f}/{avg_s:.4f}"
        else:
            cell = "N/A"
        row_str += f"| {cell:^15}"
        print(row_str)
        
    print(border)
    
    avg_row = f"{'Average':<12} | "
    for t in types:
        if len(type_totals[t]['psnr']) > 0:
            avg_p = sum(type_totals[t]['psnr']) / len(type_totals[t]['psnr'])
            avg_s = sum(type_totals[t]['ssim']) / len(type_totals[t]['ssim'])
            cell = f"{avg_p:.2f}/{avg_s:.4f}"
        else:
            cell = "N/A"
        avg_row += f"{cell:^15} "
        
    if len(overall_total['psnr']) > 0:
        avg_p = sum(overall_total['psnr']) / len(overall_total['psnr'])
        avg_s = sum(overall_total['ssim']) / len(overall_total['ssim'])
        cell = f"{avg_p:.2f}/{avg_s:.4f}"
    else:
        cell = "N/A"
        avg_p, avg_s = 0.0, 0.0
        
    avg_row += f"| {cell:^15}"
    print(avg_row)
    print(border + "\n")
    
    return avg_p, avg_s


def main():
    parser = argparse.ArgumentParser(description="D3Net SIDL Fine-Tuning")
    parser.add_argument("--train_dir", type=str, default="./data/SIDL/train",
                        help="Path to SIDL train directory")
    parser.add_argument("--val_dir", type=str, default="./data/SIDL/val",
                        help="Path to SIDL val directory")
    parser.add_argument("-e", "--epoch", type=int, default=0,
                        help="Epoch to resume training from")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Training image crop size")
    parser.add_argument("--val_img_size", type=int, default=512,
                        help="Validation image crop size (use 512 for full resolution)")
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help="Validation batch size (use 1 or 2 to avoid GPU OOM)")
    parser.add_argument("--ADB_blocks", type=int, default=12,
                        help="Number of ADB blocks")
    parser.add_argument("--model_folder", type=str, default="./ckpt/sidl_finetune",
                        help="Checkpoint save directory")
    parser.add_argument("--sample_interval", type=int, default=500,
                        help="Interval for saving sample images")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Epoch interval for saving checkpoints")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Epoch interval for validation")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to pretrained checkpoint (for transfer learning)")
    parser.add_argument("--types", type=str, default="finger,dust,scratch,water,mixed,clean",
                        help="Comma-separated contamination types to train on")
    parser.add_argument("--patience", type=int, default=20,
                        help="Epoch patience for early stopping based on validation PSNR")

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs("images/sidl_training", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------------------
    # Dataset
    # ----------------------------------------
    types = opt.types.split(',')

    train_dataset = SIDLTrainDataset(
        root=opt.train_dir,
        img_size=opt.img_size,
        types=types,
    )

    val_dataset = SIDLValDataset(
        root=opt.val_dir,
        img_size=opt.val_img_size,
        types=types,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.val_batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} iterations per epoch")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} iterations")

    # ----------------------------------------
    # Model
    # ----------------------------------------
    model = D3Net(n_channels=3, out_channels=3, num_adb_blocks=opt.ADB_blocks)
    model = model.to(device)

    # Load pretrained weights if provided
    if opt.pretrained and os.path.exists(opt.pretrained):
        print(f"Loading pretrained weights from: {opt.pretrained}")
        ckpt = torch.load(opt.pretrained, map_location=device)
        if 'net' in ckpt:
            model.load_state_dict(ckpt['net'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print("Pretrained weights loaded.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # ----------------------------------------
    # Loss & Optimizer
    # ----------------------------------------
    criterion = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
    )

    # Resume from checkpoint
    if opt.epoch != 0:
        ckpt_path = f"{opt.model_folder}/generator_{opt.epoch}.pth"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"{opt.model_folder}/generator_latest.pth"
        
        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Resumed from checkpoint: {ckpt_path}")

    # Initialize scheduler after loading optimizer state to maintain correct step alignment
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.n_epochs, eta_min=1e-6, last_epoch=opt.epoch - 1
    )

    # ----------------------------------------
    # Training Loop
    # ----------------------------------------
    best_psnr = 0.0
    best_epoch = opt.epoch

    for epoch in range(opt.epoch, opt.n_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader),    
            total=len(train_loader),
            desc=f"Epoch {epoch:02d}/{opt.n_epochs}",
            leave=False,
        )
        for i, batch in pbar:
            imgs_input = batch["input"].to(device)
            imgs_target = batch["gt"].to(device)

            optimizer.zero_grad()

            # D3Net: residual learning
            dec_info = model(imgs_input)
            output = torch.add(imgs_input, dec_info)

            loss = criterion(output, imgs_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            batches_done = epoch * len(train_loader) + i

            # Update progress bar postfix
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            # Save sample images
            if batches_done % opt.sample_interval == 0:
                with torch.no_grad():
                    img_grid = torch.cat((imgs_input[:2], output[:2], imgs_target[:2]), -1)
                    img_grid = denormalize(img_grid)
                    save_image(
                        img_grid,
                        f"images/sidl_training/{batches_done}.png",
                        nrow=1,
                        normalize=False,
                    )

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Validation
        if (epoch + 1) % opt.val_interval == 0 or epoch == opt.n_epochs - 1:
            val_psnr, val_ssim = validate(model, val_loader, device, opt.val_img_size)
            print(
                f"[Validation] Epoch {epoch}: "
                f"PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                ckpt = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "psnr": val_psnr,
                    "ssim": val_ssim,
                }
                torch.save(ckpt, f"{opt.model_folder}/generator_best.pth")
                print(f"  ★ New best PSNR: {best_psnr:.2f} dB saved!")
            else:
                print(f"  No improvement since epoch {best_epoch} (Best PSNR: {best_psnr:.2f} dB)")

        # Check early stopping patience
        if (epoch - best_epoch) >= opt.patience:
            print(f"\n[Early Stopping] No improvement in validation PSNR for {opt.patience} epochs.")
            print(f"Stopping training at epoch {epoch}. Best epoch was {best_epoch} with PSNR: {best_psnr:.2f} dB")
            break

        # Save periodic checkpoint
        if (epoch + 1) % opt.checkpoint_interval == 0:
            ckpt = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, f"{opt.model_folder}/generator_latest.pth")
            print(f"  Checkpoint saved: generator_latest.pth")

    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in: {opt.model_folder}")


if __name__ == "__main__":
    main()
