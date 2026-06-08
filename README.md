# D3Net Adaptation for Dirty-Lens Smartphone Image Restoration

This project adapts **D3Net (Dynamic Degradation Decomposition Network)** to the **SIDL Smartphone Images with Dirty Lenses** dataset. The goal is to study whether an all-in-one image restoration backbone can handle real smartphone lens contamination such as dust, fingerprints, scratches, water drops, and mixed residue.

The project was developed for **Deep Learning Programming** as a controlled, limited-budget adaptation study. It focuses on model adaptation, patch-size ablation, validation breakdowns, and failure analysis rather than claiming state-of-the-art performance on SIDL.

## Highlights

- Adapted the original D3Net codebase for paired SIDL dirty-lens restoration.
- Added a SIDL-specific dataloader, single-GPU training script, validation breakdowns, and qualitative result generation.
- Compared the adapted D3Net result against SIDL paper baselines including **AirNet** and **DiffUIR**.
- Analyzed patch-size/refinement behavior across 128, 256, and 512 crop settings.
- Identified a cross-difficulty trade-off: continued training improved Hard cases while slightly reducing Easy PSNR.

## Current Best Result

The current best validation checkpoint is from continued 256x256 training.

| Setting | Checkpoint | PSNR | SSIM |
| --- | ---: | ---: | ---: |
| 256x256 continued training | epoch 73 | **23.86 dB** | **0.8419** |
| 256x256 main run | epoch 34 | 23.82 dB | 0.8398 |
| 256x256 low-LR polishing | best observed | 23.76 dB | 0.8409 |
| 512x512 refinement | best observed | 23.27 dB | 0.8336 |
| 128x128 baseline | epoch 34 | 23.33 dB | 0.8357 |

Compared with the previous epoch-34 checkpoint, the epoch-73 checkpoint improved overall PSNR by **+0.04 dB** and SSIM by **+0.0021**.

## Key Findings

- **256x256 crop training was the most stable setting.** It balanced spatial context, crop diversity, and batch stability better than 128 or 512 crop settings.
- **512x512 refinement did not improve performance.** Although it provides more spatial context, it also reduces batch size to 1 and changes the crop distribution.
- **Hard cases improved with continued training.** Hard average PSNR increased from 20.33 dB to 20.59 dB.
- **Easy cases slightly regressed.** Easy average PSNR decreased from 27.00 dB to 26.66 dB, suggesting a cross-difficulty trade-off.
- **Loss-metric mismatch remains a limitation.** Training uses L1 loss on normalized RGB tensors, while evaluation uses PSNR/SSIM on denormalized [0, 1] images.

## Project Artifacts

| Artifact | Path |
| --- | --- |
| Final report PDF | [`project_artifacts/report/DLP_SIDL_D3NET.pdf`](project_artifacts/report/DLP_SIDL_D3NET.pdf) |
| Presentation slides PDF | [`project_artifacts/slides/D3Net-based Dirty Lens Image Restoration Slides.pdf`](project_artifacts/slides/D3Net-based%20Dirty%20Lens%20Image%20Restoration%20Slides.pdf) |
| Poster PDF | [`project_artifacts/poster/D3Net-based Dirty Lens Image Restoration Poster.pdf`](project_artifacts/poster/D3Net-based%20Dirty%20Lens%20Image%20Restoration%20Poster.pdf) |
| Experiment summary | [`D3Net/reports/sidl_experiment_summary.md`](D3Net/reports/sidl_experiment_summary.md) |
| Validation breakdowns | [`D3Net/reports/validation_breakdowns/`](D3Net/reports/validation_breakdowns/) |
| Qualitative assets | [`D3Net/reports/assets/`](D3Net/reports/assets/) |

## Repository Structure

```text
DLP-D3NET-SIDL/
  README.md
  D3Net/
    train_sidl.py
    test.py
    models/
    utils/
      sidl_dataloader.py
    reports/
      sidl_experiment_summary.md
      validation_breakdowns/
      assets/
  project_artifacts/
    report/
      DLP_SIDL_D3NET.pdf
      main.tex
    slides/
      index.html
      style.css
      D3Net-based Dirty Lens Image Restoration Slides.pdf
    poster/
      index.html
      style.css
      D3Net-based Dirty Lens Image Restoration Poster.pdf
```

## Dataset

This project uses the **SIDL Smartphone Images with Dirty Lenses** dataset.

- Benchmark page: <https://sidl-benchmark.github.io/>
- Data format: paired degraded-clean image restoration
- Validation axes: contamination type and difficulty level
- Contamination types: `clean`, `dust`, `finger`, `water`, `scratch`, `mixed`
- Difficulty levels: `easy`, `medium`, `hard`

Expected local structure:

```text
D3Net/data/SIDL/
  train/
    dust/input, dust/target
    finger/input, finger/target
    scratch/input, scratch/target
    water/input, water/target
    mixed/input, mixed/target
  val/
    dust/easy/input, dust/easy/target
    dust/medium/input, dust/medium/target
    dust/hard/input, dust/hard/target
    ...
```

Input and target filenames must match because this is paired restoration.

## Environment

The recommended environment follows the original D3Net setup.

```bash
conda create -n d3net-sidl python=3.8 -y
conda activate d3net-sidl
cd D3Net
pip install -r requirements.txt
```

Recommended runtime:

- Python 3.8
- CUDA-capable GPU
- PyTorch 2.0.0 with CUDA 11.8, or a PyTorch build compatible with your local CUDA version

## Training

Main 256x256 SIDL training:

```bash
cd D3Net
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --n_epochs 100 \
  --batch_size 8 \
  --img_size 256 \
  --val_img_size 512 \
  --val_batch_size 1 \
  --lr 0.0001 \
  --model_folder ./ckpt/sidl_scratch_4090_crop256
```

Resume continued training:

```bash
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --epoch 64 \
  --n_epochs 100 \
  --batch_size 8 \
  --img_size 256 \
  --val_img_size 512 \
  --val_batch_size 1 \
  --lr 0.0001 \
  --model_folder ./ckpt/sidl_scratch_4090_crop256
```

512 refinement from a 256 checkpoint:

```bash
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --pretrained ./ckpt/sidl_scratch_4090_crop256/generator_best.pth \
  --n_epochs 20 \
  --batch_size 1 \
  --img_size 512 \
  --val_img_size 512 \
  --lr 0.00001 \
  --model_folder ./ckpt/sidl_refine512_from256best_lr1e5
```

## Evaluation

During validation, D3Net predicts a residual correction:

```text
restored image = input image + D3Net(input image)
```

The restored image and target are denormalized using ImageNet mean/std, clamped to `[0, 1]`, and evaluated with PSNR/SSIM. Metrics are aggregated by contamination type and difficulty level.

Example test command:

```bash
cd D3Net
python test.py \
  --image_path ./data/SIDL/val/finger/easy/input \
  --target_data_dir ./data/SIDL/val/finger/easy/target \
  --save_path ./images/test_finger_easy \
  --epoch best \
  --model_folder ./ckpt/sidl_scratch_4090_crop256 \
  --img_width 512 \
  --img_height 512
```

## Limitations

- The original D3Net training scale is much larger, roughly 1200 epochs. This project's best checkpoint is epoch 73, so results should be interpreted as limited-budget adaptation.
- AirNet and DiffUIR values come from the SIDL paper test setting, while D3Net values here are from this project's validation split.
- The training objective and evaluation metric are not perfectly aligned: L1 loss is computed on normalized tensors, while PSNR/SSIM are computed on denormalized images.
- Checkpoints and dataset files may be excluded from a public repository due to file size and dataset distribution constraints.

## Future Work

- Longer training or pretrained D3Net initialization
- 512 crop curriculum instead of direct 512 refinement
- Charbonnier loss, L1 + small MSE hybrid loss, or denormalized-space MSE polishing
- Difficulty-aware sampling or weighting
- Frequency-aware and edge-aware losses
- Controlled comparison against AirNet and DiffUIR under the same split and compute budget

## References

- SIDL Benchmark: <https://sidl-benchmark.github.io/>
- D3Net Repository: <https://github.com/codeshop715/D3Net>
- D3Net Paper: <https://arxiv.org/html/2502.19068v1>
