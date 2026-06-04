# D3Net Fine-Tuning for SIDL Dirty Lens Image Restoration

본 프로젝트는 All-in-One Image Restoration 모델인 **D3Net (Dynamic Degradation Decomposition Network)** 을 **SIDL Smartphone Image Dirty Lens Dataset**에 fine-tuning하여, 실제 스마트폰 렌즈 오염으로 인해 발생한 이미지 열화를 복원하는 실험 프로젝트이다.

SIDL 데이터셋은 fingerprint, dust, scratch, water drop, mixed debris 등 실제 렌즈 오염으로 촬영된 degraded-clean image pair를 포함한다. 본 프로젝트에서는 D3Net의 degradation decomposition 구조가 dirty lens restoration 문제로 얼마나 잘 transfer되는지 확인하고, contamination type 및 difficulty별 성능을 PSNR/SSIM으로 분석한다.

## 현재 구현 상태

- 원본 D3Net 코드 기반 유지
- SIDL paired dataset 전용 dataloader 추가: `D3Net/utils/sidl_dataloader.py`
- SIDL single-GPU fine-tuning script 추가: `D3Net/train_sidl.py`
- validation에서 contamination type별, difficulty별 PSNR/SSIM 출력
- checkpoint 및 학습 중간 복원 이미지 저장

## 프로젝트 구조

```text
DLP-D3NET-SIDL/
  README.md
  project_overview.md
  metadata.json
  train_patch.tar
  D3Net/
    train_sidl.py
    test.py
    train.py
    requirements.txt
    data/
      SIDL/
        train/
        val/
    ckpt/
      sidl_finetune/
    images/
      sidl_training/
    models/
    utils/
```

## 실행 환경

권장 환경은 원본 D3Net과 현재 `requirements.txt` 기준이다.

- OS: Linux 권장
- Python: 3.8.x 권장
- CUDA GPU 권장
- PyTorch: 2.0.0 + CUDA 11.8
- Torchvision: 0.15.1 + CUDA 11.8

환경 생성 예시는 다음과 같다.

```bash
conda create -n d3net-sidl python=3.8 -y
conda activate d3net-sidl
cd D3Net
pip install -r requirements.txt
```

CUDA 버전에 따라 `torch`, `torchvision`, `torchaudio` 설치가 실패할 수 있다. 이 경우 현재 GPU/CUDA 환경에 맞는 PyTorch wheel을 먼저 설치한 뒤 `requirements.txt`를 설치한다.

주의: `train_sidl.py`는 CUDA가 없으면 CPU도 선택할 수 있지만, 현재 `test.py`는 내부에서 `.cuda()`를 직접 호출하므로 GPU 환경에서 실행하는 것을 기준으로 한다.

## 필요 데이터셋

필수 데이터셋은 **SIDL Smartphone Image Dirty Lens Dataset**이다.

- Benchmark page: https://sidl-benchmark.github.io/
- 사용 데이터: SIDL 512x512 patchified train/validation image pairs
- 오염 유형: `finger`, `dust`, `scratch`, `water`, `mixed`
- 검증 난이도: `easy`, `medium`, `hard`

현재 프로젝트 루트에는 SIDL patch archive로 보이는 `train_patch.tar`와 `metadata.json`이 있으며, 학습 코드는 압축 해제 후 아래 구조를 사용한다.

로컬 데이터에는 `clean`, `all` 폴더도 있을 수 있지만, 기본 학습 옵션은 `finger,dust,scratch,water,mixed`만 사용한다.

```text
D3Net/data/SIDL/
  train/
    finger/
      input/
      target/
    dust/
      input/
      target/
    scratch/
      input/
      target/
    water/
      input/
      target/
    mixed/
      input/
      target/
  val/
    finger/
      easy/
        input/
        target/
      medium/
        input/
        target/
      hard/
        input/
        target/
    dust/
      easy|medium|hard/
        input/
        target/
    scratch/
      easy|medium|hard/
        input/
        target/
    water/
      easy|medium|hard/
        input/
        target/
    mixed/
      easy|medium|hard/
        input/
        target/
```

`input/`에는 dirty lens degraded image를, `target/`에는 같은 파일명을 가진 clean GT image를 둔다. Paired restoration이므로 input과 target 파일명이 반드시 일치해야 한다.

## SIDL Fine-Tuning 실행

아래 명령은 현재 프로젝트의 주 실행 경로이다.

```bash
cd D3Net
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --n_epochs 100 \
  --batch_size 4 \
  --img_size 128 \
  --val_img_size 512 \
  --val_batch_size 1 \
  --lr 0.0001 \
  --model_folder ./ckpt/sidl_finetune
```

GPU 메모리가 부족하면 먼저 아래 옵션을 줄인다.

- `--batch_size`
- `--val_batch_size`
- `--img_size`
- `--val_img_size`

특정 오염 유형만 학습하려면 `--types`를 사용한다.

```bash
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --types finger,dust,scratch \
  --n_epochs 50 \
  --batch_size 4 \
  --model_folder ./ckpt/sidl_finetune_subset
```

## Pretrained / Resume

원본 D3Net 또는 이전 실험 checkpoint에서 transfer learning을 시작하려면 `--pretrained`를 사용한다.

```bash
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --pretrained ./ckpt/pretrained/generator.pth \
  --model_folder ./ckpt/sidl_finetune
```

중단된 학습을 이어서 실행하려면 `--epoch`를 지정한다. 지정한 epoch checkpoint가 없으면 `generator_latest.pth`를 사용한다.

```bash
python train_sidl.py \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --epoch 50 \
  --n_epochs 100 \
  --model_folder ./ckpt/sidl_finetune
```

## 평가 및 결과 확인

`train_sidl.py`는 `--val_interval`마다 validation을 실행하고, 아래 항목을 출력한다.

- 전체 평균 PSNR / SSIM
- contamination type별 PSNR / SSIM
- `easy`, `medium`, `hard` difficulty별 PSNR / SSIM

checkpoint는 다음 위치에 저장된다.

```text
D3Net/ckpt/sidl_finetune/
  generator_best.pth
  generator_latest.pth
```

학습 중간 샘플 이미지는 다음 위치에 저장된다. 한 이미지 안에 input, restored output, target이 나란히 저장되어 정성 평가에 사용할 수 있다.

```text
D3Net/images/sidl_training/
```

## Test Script 사용

`D3Net/test.py`는 지정한 input directory의 이미지를 복원하고, 같은 파일명을 가진 GT directory와 비교해 PSNR/SSIM을 계산한다. 현재 스크립트는 `--target_data_dir` 이미지를 직접 열기 때문에 GT directory가 필요하다.

```bash
cd D3Net
python test.py \
  --image_path ./data/SIDL/val/finger/easy/input \
  --target_data_dir ./data/SIDL/val/finger/easy/target \
  --save_path ./images/test_finger_easy \
  --epoch best \
  --model_folder ./ckpt/sidl_finetune \
  --img_width 512 \
  --img_height 512
```

주의: 현재 `test.py`는 checkpoint 파일명을 `generator_{epoch}.pth` 형식으로 찾는다. 따라서 `--epoch best`는 `generator_best.pth`, `--epoch latest`는 `generator_latest.pth`를 사용한다.

```bash
python test.py \
  --image_path ./data/SIDL/val/finger/easy/input \
  --target_data_dir ./data/SIDL/val/finger/easy/target \
  --save_path ./images/test_finger_easy_latest \
  --epoch latest \
  --model_folder ./ckpt/sidl_finetune \
  --img_width 512 \
  --img_height 512
```

복원 결과 concat 이미지는 다음 형식의 경로에 저장된다.

```text
{save_path}_concat/
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--train_dir` | `./data/SIDL/train` | SIDL train directory |
| `--val_dir` | `./data/SIDL/val` | SIDL validation directory |
| `--n_epochs` | `100` | 총 학습 epoch |
| `--batch_size` | `4` | train batch size |
| `--img_size` | `128` | train random crop size |
| `--val_img_size` | `512` | validation center crop size |
| `--val_batch_size` | `1` | validation batch size |
| `--lr` | `1e-4` | Adam learning rate |
| `--types` | `finger,dust,scratch,water,mixed` | 사용할 오염 유형 |
| `--model_folder` | `./ckpt/sidl_finetune` | checkpoint 저장 위치 |
| `--val_interval` | `5` | validation 실행 주기 |
| `--checkpoint_interval` | `5` | latest checkpoint 저장 주기 |
| `--patience` | `20` | validation PSNR 기준 early stopping patience |

## 결과 정리 시 포함할 항목

보고서나 발표 자료에는 다음 항목을 정리하면 된다.

- validation 전체 평균 PSNR / SSIM
- contamination type별 PSNR / SSIM
- difficulty별 PSNR / SSIM
- input / restored / target 비교 이미지
- zero-shot D3Net과 SIDL fine-tuning D3Net 비교
- 실패 사례: scratch, water drop, mixed debris 등 어려운 유형의 시각적 artifact

## README 업데이트 원칙

앞으로 코드나 실험 설정을 바꾸면 README도 함께 갱신한다.

- 실행 명령이 바뀌면 `SIDL Fine-Tuning 실행`과 `Test Script 사용` 섹션 수정
- dataloader 구조가 바뀌면 `필요 데이터셋` 섹션 수정
- checkpoint, sample image, metric 저장 방식이 바뀌면 `평가 및 결과 확인` 섹션 수정
- 새 loss, pretrained model, ablation이 추가되면 `현재 구현 상태`와 `결과 정리 시 포함할 항목`에 반영

## References

- SIDL Benchmark: https://sidl-benchmark.github.io/
- D3Net Repository: https://github.com/codeshop715/D3Net
- D3Net Paper: https://arxiv.org/html/2502.19068v1
- All-in-One Image Restoration Survey: https://github.com/Harbinzzy/All-in-One-Image-Restoration-Survey
