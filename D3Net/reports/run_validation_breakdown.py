import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.base_model import D3Net  # noqa: E402
from utils.metrics import calc_ssim  # noqa: E402
from utils.sidl_dataloader import SIDLValDataset, mean, std  # noqa: E402


TYPE_ORDER = ["clean", "dust", "finger", "mixed", "scratch", "water"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def psnr_tensor(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(1.0, device=output.device) / torch.sqrt(mse))


def denormalize_batch(batch):
    out = batch.clone()
    for c in range(3):
        out[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(out, 0, 1)


def summarize(results):
    table = {}
    overall = {"psnr": [], "ssim": []}

    for difficulty in DIFFICULTY_ORDER:
        row = {}
        row_psnr = []
        row_ssim = []
        for ctype in TYPE_ORDER:
            vals = results[ctype][difficulty]
            if vals["psnr"]:
                psnr = sum(vals["psnr"]) / len(vals["psnr"])
                ssim = sum(vals["ssim"]) / len(vals["ssim"])
                row[ctype] = {"psnr": psnr, "ssim": ssim}
                row_psnr.append(psnr)
                row_ssim.append(ssim)
                overall["psnr"].extend(vals["psnr"])
                overall["ssim"].extend(vals["ssim"])
            else:
                row[ctype] = None
        row["average"] = {
            "psnr": sum(row_psnr) / len(row_psnr),
            "ssim": sum(row_ssim) / len(row_ssim),
        } if row_psnr else None
        table[difficulty] = row

    avg_row = {}
    for ctype in TYPE_ORDER:
        vals = {"psnr": [], "ssim": []}
        for difficulty in DIFFICULTY_ORDER:
            vals["psnr"].extend(results[ctype][difficulty]["psnr"])
            vals["ssim"].extend(results[ctype][difficulty]["ssim"])
        avg_row[ctype] = {
            "psnr": sum(vals["psnr"]) / len(vals["psnr"]),
            "ssim": sum(vals["ssim"]) / len(vals["ssim"]),
        } if vals["psnr"] else None
    avg_row["average"] = {
        "psnr": sum(overall["psnr"]) / len(overall["psnr"]),
        "ssim": sum(overall["ssim"]) / len(overall["ssim"]),
    }
    table["average"] = avg_row
    return table


def markdown_table(table):
    headers = ["Difficulty"] + [t.capitalize() for t in TYPE_ORDER] + ["Average"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---" + "|---:" * (len(headers) - 1) + "|",
    ]
    for difficulty in DIFFICULTY_ORDER + ["average"]:
        label = difficulty.capitalize()
        row = [label]
        for key in TYPE_ORDER + ["average"]:
            cell = table[difficulty][key]
            if cell is None:
                row.append("-")
            else:
                row.append(f"{cell['psnr']:.2f} / {cell['ssim']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run(args):
    if not torch.cuda.is_available():
        # CPU fallback for local report generation. The model's FFT helper calls
        # `.cuda()` directly, so make that a no-op only when CUDA is unavailable.
        torch.Tensor.cuda = lambda self, *a, **kw: self

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SIDLValDataset(
        root=args.val_dir,
        img_size=args.val_img_size,
        types=TYPE_ORDER,
        difficulties=DIFFICULTY_ORDER,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = D3Net(n_channels=3, out_channels=3, num_adb_blocks=args.adb_blocks).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["net"])
    model.eval()

    results = defaultdict(lambda: defaultdict(lambda: {"psnr": [], "ssim": []}))
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validating {args.name}", leave=True):
            imgs_input = batch["input"].to(device, non_blocking=True)
            imgs_target = batch["gt"].to(device, non_blocking=True)
            output = model(imgs_input) + imgs_input

            output_dn = denormalize_batch(output)
            target_dn = denormalize_batch(imgs_target)

            for j in range(output_dn.size(0)):
                ctype = batch["type"][j]
                difficulty = batch["difficulty"][j]
                p = psnr_tensor(output_dn[j], target_dn[j])
                s = calc_ssim(output_dn[j].unsqueeze(0), target_dn[j].unsqueeze(0))
                results[ctype][difficulty]["psnr"].append(
                    p.item() if isinstance(p, torch.Tensor) else p
                )
                results[ctype][difficulty]["ssim"].append(
                    s.item() if isinstance(s, torch.Tensor) else s
                )

    table = summarize(results)
    output = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_psnr": ckpt.get("psnr"),
        "checkpoint_ssim": ckpt.get("ssim"),
        "val_img_size": args.val_img_size,
        "batch_size": args.batch_size,
        "num_samples": len(dataset),
        "device": str(device),
        "table": table,
        "markdown": markdown_table(table),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.name}.json"
    md_path = args.out_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(output["markdown"] + "\n", encoding="utf-8")

    print(output["markdown"])
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-dir", default=str(ROOT / "data" / "SIDL" / "val"))
    parser.add_argument("--val-img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--adb-blocks", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "validation_breakdowns")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
