# D3Net Adaptation for Dirty-Lens Smartphone Image Restoration

This project adapts **D3Net (Dynamic Degradation Decomposition Network)** to the **SIDL Smartphone Images with Dirty Lenses** dataset. The goal is to study whether an all-in-one image restoration backbone can handle real smartphone lens contamination such as dust, fingerprints, scratches, water drops, and mixed residue.

The project was developed for **Deep Learning Programming** as a controlled, limited-budget adaptation study. It focuses on model adaptation, patch-size ablation, validation breakdowns, and failure analysis rather than claiming state-of-the-art performance on SIDL.

## Highlights

- Adapted the original D3Net codebase for paired SIDL dirty-lens restoration.
- Added a SIDL-specific dataloader, single-GPU training script, validation breakdowns, and qualitative result generation.
- Compared the adapted D3Net result against SIDL paper baselines including **AirNet** and **DiffUIR**.
- Analyzed patch-size/refinement behavior across 128, 256, and 512 crop settings.
- Submitted the epoch-73 checkpoint to the official SIDL leaderboard and reported test-set performance, GMACs, and parameter count.

## Official Leaderboard Result

The model submitted to the official SIDL leaderboard uses the epoch-73 checkpoint from continued 256x256 training.

| Setting | Checkpoint | Test PSNR | Test SSIM |
| --- | ---: | ---: | ---: |
| 256x256 continued training | epoch 73 | **24.55 dB** | **0.8507** |
| 256x256 main run | epoch 34 | 23.82 dB | 0.8398 |
| 256x256 low-LR polishing | best observed | 23.76 dB | 0.8409 |
| 512x512 refinement | best observed | 23.27 dB | 0.8336 |
| 128x128 baseline | epoch 34 | 23.33 dB | 0.8357 |

The leaderboard submission reports **771 GMACs** and **43.8M parameters**.

## Key Findings

- **256x256 crop training was the most stable setting.** It balanced spatial context, crop diversity, and batch stability better than 128 or 512 crop settings.
- **512x512 refinement did not improve performance.** Although it provides more spatial context, it also reduces batch size to 1 and changes the crop distribution.
- **Official test-set average is 24.55 dB / 0.8507.** This is the reported leaderboard result for the epoch-73 checkpoint.
- **Hard cases remain difficult.** Official Hard average is 20.92 dB / 0.8064, while Easy average is 28.02 dB / 0.9051.
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

## Official Leaderboard Breakdown

| Difficulty | Clean | Dust | Fingerprint | Water | Scratch | Mixed | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 33.20 / 0.9603 | 25.10 / 0.8926 | 26.95 / 0.8994 | 26.45 / 0.8915 | 28.84 / 0.9123 | 27.56 / 0.8745 | 28.02 / 0.9051 |
| Medium | 30.57 / 0.9132 | 22.53 / 0.7980 | 25.50 / 0.8554 | 23.73 / 0.8207 | 25.69 / 0.8620 | 20.18 / 0.7950 | 24.70 / 0.8407 |
| Hard | 27.63 / 0.9033 | 20.21 / 0.8029 | 18.09 / 0.6940 | 19.47 / 0.7950 | 21.32 / 0.8341 | 18.82 / 0.8089 | 20.92 / 0.8064 |
| Average | 30.47 / 0.9256 | 22.61 / 0.8312 | 23.51 / 0.8163 | 23.22 / 0.8357 | 25.28 / 0.8695 | 22.19 / 0.8261 | 24.55 / 0.8507 |

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

- The original D3Net training scale is much larger, roughly 1200 epochs. This project's submitted checkpoint is epoch 73, so results should be interpreted as limited-budget adaptation.
- AirNet and DiffUIR values come from the SIDL paper table. D3Net values here are from the official leaderboard submission unless explicitly marked as internal validation results.
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
