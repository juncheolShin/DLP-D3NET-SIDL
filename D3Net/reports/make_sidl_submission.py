#!/usr/bin/env python3
"""Generate SIDL leaderboard submission images and archive.

The script preserves the official relative layout:

  <output>/<type>/<difficulty>/input/<filename>.png

Local SIDL folders may use ``finger`` while the official submission system
expects ``fingerprint``. The output path is normalized accordingly.

Run from the repository root, for example:

  python D3Net/reports/make_sidl_submission.py \
    --input-root D3Net/data/SIDL/val \
    --checkpoint D3Net/ckpt/sidl_scratch_4090_crop256/generator_best.pth \
    --output-root submissions/sidl_d3net_best256 \
    --archive submissions/sidl_d3net_best256.zip \
    --device cuda
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
D3NET_ROOT = REPO_ROOT / "D3Net"
sys.path.insert(0, str(D3NET_ROOT))

import models.base_model as base_model  # noqa: E402
from models.base_model import D3Net  # noqa: E402


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TYPE_NAME_MAP = {
    "finger": "fingerprint",
}


def patch_fft_preprocess(device: torch.device) -> None:
    """Patch original D3Net preprocessing helpers so they work on CPU or CUDA."""

    def batch_fft_preprocess(img_batch_numpy):
        batch_magnitude = []
        for img_numpy in img_batch_numpy:
            img = np.transpose(img_numpy, (1, 2, 0))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_np = np.uint8(np.clip(gray_img * 255, 0, 255))
            freq = np.fft.fft2(img_np)
            freq_shift = np.fft.fftshift(freq)
            magnitude_spectrum = np.abs(freq_shift)
            batch_magnitude.append(magnitude_spectrum)

        magnitude_array = np.stack(batch_magnitude, axis=0)
        return torch.from_numpy(magnitude_array).float().to(device).unsqueeze(1)

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


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(tensor.device)
    std = STD.to(tensor.device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def iter_input_images(input_root: Path):
    for input_dir in sorted(input_root.glob("*/*/input")):
        if not input_dir.is_dir():
            continue
        for image_path in sorted(input_dir.glob("*.png")):
            yield image_path


def normalize_submission_path(relative_path: Path) -> Path:
    parts = list(relative_path.parts)
    if parts:
        parts[0] = TYPE_NAME_MAP.get(parts[0], parts[0])
    return Path(*parts)


def save_tensor_png(tensor: torch.Tensor, path: Path) -> None:
    image = denormalize(tensor.detach().cpu()).squeeze(0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def build_archive(output_root: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix.lower() == ".zip":
        base_name = archive_path.with_suffix("")
        made = shutil.make_archive(str(base_name), "zip", root_dir=output_root)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix.lower() == ".tgz":
        base_name = archive_path
        if archive_path.suffix.lower() == ".tgz":
            base_name = archive_path.with_suffix("")
        else:
            base_name = archive_path.with_suffix("").with_suffix("")
        made = shutil.make_archive(str(base_name), "gztar", root_dir=output_root)
    else:
        raise ValueError("archive must end with .zip, .tar.gz, or .tgz")
    return Path(made)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit. 0 means all images.")
    parser.add_argument(
        "--exclude-types",
        type=str,
        default="",
        help="Comma-separated local type directories to skip, e.g. clean.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    patch_fft_preprocess(device)

    model = D3Net(3, 3, 12).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    exclude_types = {item.strip() for item in args.exclude_types.split(",") if item.strip()}
    image_paths = [
        path for path in iter_input_images(args.input_root)
        if path.relative_to(args.input_root).parts[0] not in exclude_types
    ]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No PNG inputs found under {args.input_root}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Restoring"):
            relative_path = image_path.relative_to(args.input_root)
            output_path = args.output_root / normalize_submission_path(relative_path)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            residual = model(input_tensor)
            restored = input_tensor + residual
            save_tensor_png(restored, output_path)

    print(f"Saved {len(image_paths)} images to {args.output_root}")
    if args.archive:
        archive = build_archive(args.output_root, args.archive)
        print(f"Created archive: {archive}")


if __name__ == "__main__":
    main()
