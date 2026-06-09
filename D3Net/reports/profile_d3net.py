#!/usr/bin/env python3
"""Report parameter count and approximate GMACs for D3Net.

GMACs are estimated with torch.profiler's FLOP counter and converted as
GMACs = FLOPs / 2 / 1e9. The profiler covers major convolution and matmul
operators; FFT preprocessing and unsupported ops may not be fully counted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
D3NET_ROOT = REPO_ROOT / "D3Net"
sys.path.insert(0, str(D3NET_ROOT))

import models.base_model as base_model  # noqa: E402
from models.base_model import D3Net  # noqa: E402


def patch_fft_preprocess(device: torch.device) -> None:
    def batch_fft_preprocess(img_batch_numpy):
        batch_magnitude = []
        for img_numpy in img_batch_numpy:
            img = np.transpose(img_numpy, (1, 2, 0))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = np.uint8(np.clip(gray_img * 255, 0, 255))
            freq = np.fft.fft2(img_np)
            freq_shift = np.fft.fftshift(freq)
            batch_magnitude.append(np.abs(freq_shift))
        return torch.from_numpy(np.stack(batch_magnitude, axis=0)).float().to(device).unsqueeze(1)

    base_model.batch_fft_preprocess = batch_fft_preprocess

    def gumbel_softmax(logits, temperature=1, hard=False):
        eps = 1e-20
        noise = torch.rand(logits.size(), device=logits.device)
        noise = -torch.log(-torch.log(noise + eps) + eps)
        y = F.softmax((logits + noise) / temperature, dim=-1)
        if not hard:
            return y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    base_model.gumbel_softmax = gumbel_softmax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    patch_fft_preprocess(device)
    model = D3Net(3, 3, 12).to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input size: 1 x 3 x {args.size} x {args.size}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    if args.skip_flops:
        return

    sample = torch.randn(1, 3, args.size, args.size, device=device)
    with torch.no_grad():
        with torch.profiler.profile(with_flops=True, activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            _ = model(sample)

    flops = sum(evt.flops for evt in prof.key_averages() if evt.flops)
    print(f"Profiled FLOPs: {flops:,}")
    print(f"Approx. GMACs: {flops / 2 / 1e9:.4f}")


if __name__ == "__main__":
    main()
