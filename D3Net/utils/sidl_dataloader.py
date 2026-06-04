"""
SIDL Dataset Dataloader for D3Net Fine-Tuning.

SIDL dataset structure (after extraction):
  train/
    {finger,dust,scratch,water,mixed,clean}/
      input/   - degraded images (512x512 patches)
      target/  - clean GT images (512x512 patches)

  val/
    {finger,dust,scratch,water,mixed,clean}/
      {easy,medium,hard}/
        input/
        target/
"""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Same normalization as original D3Net
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """Denormalizes image tensors using mean and std"""
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class SIDLTrainDataset(Dataset):
    """
    SIDL Training Dataset.
    Loads degraded-clean pairs from all contamination types.
    Each __getitem__ returns a single pair (not 5 like the original ALL_Dataset).
    """

    def __init__(self, root, img_size=128, types=None):
        """
        Args:
            root: path to train/ directory (e.g. data/SIDL/train)
            img_size: training patch size (will random crop from 512x512)
            types: list of contamination types to use. 
                   Default: ['finger', 'dust', 'scratch', 'water', 'mixed']
        """
        self.root = root
        self.img_size = img_size

        if types is None:
            types = ['finger', 'dust', 'scratch', 'water', 'mixed']

        self.pairs = []  # list of (input_path, target_path, type_name)

        for t in types:
            input_dir = os.path.join(root, t, 'input')
            target_dir = os.path.join(root, t, 'target')

            if not os.path.isdir(input_dir):
                print(f"[WARNING] {input_dir} not found, skipping...")
                continue

            input_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
            for inp_path in input_files:
                fname = os.path.basename(inp_path)
                tgt_path = os.path.join(target_dir, fname)
                if os.path.exists(tgt_path):
                    self.pairs.append((inp_path, tgt_path, t))
                else:
                    print(f"[WARNING] target not found: {tgt_path}")

        print(f"[SIDLTrainDataset] Loaded {len(self.pairs)} pairs from {root}")
        for t in types:
            cnt = sum(1 for _, _, tt in self.pairs if tt == t)
            print(f"  {t}: {cnt}")

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __getitem__(self, index):
        inp_path, tgt_path, ctype = self.pairs[index]

        img_input = Image.open(inp_path).convert('RGB')
        img_target = Image.open(tgt_path).convert('RGB')

        # Get random crop parameters (exactly same for both images)
        w, h = img_input.size
        th = tw = self.img_size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        img_input = TF.crop(img_input, i, j, th, tw)
        img_target = TF.crop(img_target, i, j, th, tw)

        # Random horizontal flip (applied identically)
        if random.random() > 0.5:
            img_input = TF.hflip(img_input)
            img_target = TF.hflip(img_target)

        # Random vertical flip (applied identically)
        if random.random() > 0.5:
            img_input = TF.vflip(img_input)
            img_target = TF.vflip(img_target)

        img_input = self.to_tensor(img_input)
        img_target = self.to_tensor(img_target)

        return {
            "input": img_input,
            "gt": img_target,
            "type": ctype,
        }

    def __len__(self):
        return len(self.pairs)


class SIDLValDataset(Dataset):
    """
    SIDL Validation Dataset.
    Loads degraded-clean pairs from all contamination types and difficulties.
    No random crop - uses center crop or resize to fixed size.
    """

    def __init__(self, root, img_size=128, types=None, difficulties=None):
        """
        Args:
            root: path to val/ directory
            img_size: evaluation image size
            types: contamination types
            difficulties: difficulty levels
        """
        self.root = root
        self.img_size = img_size

        if types is None:
            types = ['finger', 'dust', 'scratch', 'water', 'mixed']
        if difficulties is None:
            difficulties = ['easy', 'medium', 'hard']

        self.pairs = []

        for t in types:
            for d in difficulties:
                input_dir = os.path.join(root, t, d, 'input')
                target_dir = os.path.join(root, t, d, 'target')

                if not os.path.isdir(input_dir):
                    continue

                input_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
                for inp_path in input_files:
                    fname = os.path.basename(inp_path)
                    tgt_path = os.path.join(target_dir, fname)
                    if os.path.exists(tgt_path):
                        self.pairs.append((inp_path, tgt_path, t, d))

        print(f"[SIDLValDataset] Loaded {len(self.pairs)} pairs from {root}")
        for t in types:
            cnt = sum(1 for _, _, tt, _ in self.pairs if tt == t)
            print(f"  {t}: {cnt}")

        self.transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __getitem__(self, index):
        inp_path, tgt_path, ctype, difficulty = self.pairs[index]

        img_input = Image.open(inp_path).convert('RGB')
        img_target = Image.open(tgt_path).convert('RGB')

        img_input = self.transform(img_input)
        img_target = self.transform(img_target)

        return {
            "input": img_input,
            "gt": img_target,
            "type": ctype,
            "difficulty": difficulty,
            "filename": os.path.basename(inp_path),
        }

    def __len__(self):
        return len(self.pairs)
