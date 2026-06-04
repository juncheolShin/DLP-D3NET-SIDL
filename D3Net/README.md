# D3Net SIDL Experiment Code

이 디렉터리는 원본 D3Net 구현을 기반으로 SIDL dirty lens image restoration fine-tuning을 수행하는 코드 영역이다. 프로젝트 전체 설명과 실험 문서는 상위 디렉터리의 `README.md`를 우선 참고한다.

## Environment

권장 환경은 다음과 같다.

- Python 3.8.x
- Linux
- CUDA GPU
- PyTorch 2.0.0 + CUDA 11.8
- Torchvision 0.15.1 + CUDA 11.8

`test.py`는 현재 내부에서 `.cuda()`를 직접 호출하므로 GPU 환경을 기준으로 실행한다.

설치 예시:

```bash
conda create -n d3net-sidl python=3.8 -y
conda activate d3net-sidl
pip install -r requirements.txt
```

## Required Dataset

현재 SIDL 전용 코드는 다음 데이터셋 구조를 요구한다.

```text
data/SIDL/
  train/
    finger|dust|scratch|water|mixed/
      input/
      target/
  val/
    finger|dust|scratch|water|mixed/
      easy|medium|hard/
        input/
        target/
```

필수 데이터셋:

- SIDL Smartphone Image Dirty Lens Dataset
- 512x512 patchified train/validation degraded-clean pairs 권장
- Benchmark page: https://sidl-benchmark.github.io/

`input/`과 `target/`의 파일명은 반드시 동일해야 한다.

## SIDL Fine-Tuning

```bash
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

특정 오염 유형만 사용할 수 있다.

```bash
python train_sidl.py \
  --types finger,dust,scratch \
  --train_dir ./data/SIDL/train \
  --val_dir ./data/SIDL/val \
  --model_folder ./ckpt/sidl_finetune_subset
```

## Outputs

학습 결과는 다음 위치에 저장된다.

```text
ckpt/sidl_finetune/
  generator_best.pth
  generator_latest.pth

images/sidl_training/
  {batch_step}.png
```

`train_sidl.py` validation은 전체 평균뿐 아니라 contamination type별, difficulty별 PSNR/SSIM table을 출력한다.

## Test

```bash
python test.py \
  --image_path ./data/SIDL/val/finger/easy/input \
  --target_data_dir ./data/SIDL/val/finger/easy/target \
  --save_path ./images/test_finger_easy \
  --epoch best \
  --model_folder ./ckpt/sidl_finetune \
  --img_width 512 \
  --img_height 512
```

`test.py`는 `generator_{epoch}.pth`를 읽는다. 예를 들어 `--epoch best`는 `ckpt/sidl_finetune/generator_best.pth`를 사용한다.

## Original D3Net Training

원본 all-in-one restoration 학습 코드는 `train.py`에 남아 있다. 이 코드는 distributed training과 원본 D3Net dataloader 구조를 사용하므로, SIDL 실험에는 `train_sidl.py`를 우선 사용한다.

원본 실행 예시는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 train.py \
  --epoch 0 \
  --n_epochs 2000 \
  --train_datasets your_Datasets \
  --model_folder your_model_folder
```

## Acknowledgement

이 코드는 D3Net 구현을 기반으로 하며, 원본 D3Net은 BasicSR, RDN, HorNet 등 여러 공개 구현을 참고한다.
