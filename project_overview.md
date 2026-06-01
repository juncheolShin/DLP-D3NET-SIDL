# D3Net Fine-Tuning 실험 계획: SIDL Dirty Lens Image Restoration

## 1. 프로젝트 주제

**Dynamic Degradation Decomposition Network Fine-Tuning for Smartphone Dirty Lens Image Restoration**

본 프로젝트는 All-in-One Image Restoration 계열 모델인 **D3Net**을 SIDL 데이터셋에 fine-tuning하여, 스마트폰 렌즈 오염으로 발생하는 실제 이미지 열화를 복원하는 것을 목표로 한다. SIDL은 fingerprint, dust, scratch, water drop, mixed debris 등 실제 렌즈 오염으로 촬영된 degraded-clean paired dataset이며, 300개 static scene과 1,588개 degraded-clean image pair를 제공한다. 또한 학습 효율을 위해 512×512 patchified image도 제공한다. ([SIDL Benchmark][1])

## 2. 왜 D3Net인가?

D3Net은 **Dynamic Degradation Decomposition Network**의 약자이며, 여러 degradation을 하나의 모델에서 처리하는 **All-in-One Image Restoration** 모델이다. D3Net 논문은 기존 all-in-one restoration 모델들이 복잡하고 모호한 degradation type을 다루는 데 어려움을 가진다고 보고, 이를 해결하기 위해 **frequency-domain degradation feature**와 **spatial-domain image feature**를 함께 사용하는 구조를 제안한다. ([arXiv][2])

SIDL의 dirty lens degradation은 일반적인 noise, blur, rain, haze와 달리 렌즈 앞의 오염물에 의해 생기는 실제 촬영 열화다. 예를 들어 dust는 국소적인 spot artifact, scratch는 edge-like high-frequency artifact, water drop은 local blur/halo/refraction, fingerprint는 smear와 contrast degradation을 만든다. 따라서 D3Net의 **frequency-spatial degradation analysis**는 SIDL 문제와 개념적으로 잘 맞는다. 다만 D3Net의 기존 성능이 SIDL에서 그대로 보장되는 것은 아니므로, 본 프로젝트의 핵심은 “D3Net의 degradation-adaptive 구조가 dirty lens domain으로 얼마나 transfer되는지 실험적으로 검증하는 것”이다.

## 3. D3Net 핵심 개념 쉽게 이해하기

### 3.1 전체 구조

D3Net은 크게 두 가지 branch로 구성된다.

| 구성 요소                             | 역할                                                                |
| --------------------------------- | ----------------------------------------------------------------- |
| Restoration Reconstruction Branch | 실제 이미지를 복원하는 U-Net 계열 main branch                                 |
| Degradation Decomposition Branch  | 입력 이미지의 degradation 특성을 분석하고, 복원 branch가 어떤 방식으로 복원할지 보조하는 branch |

D3Net 논문에 따르면 restoration branch는 일반적인 U-Net 구조를 사용하고, shallow degradation feature를 degradation decomposition branch로 전달한다. 이 decomposition branch는 CDDA가 만든 prompt를 이용해 degradation-adaptive feature를 생성하고, 이를 다시 restoration branch에 feedback하여 복원 성능을 높인다. 또한 D3Net은 별도의 degradation classification loss 없이 reconstruction loss만으로 end-to-end 학습된다. ([arXiv][2])

쉽게 말하면, 일반 U-Net이 “입력 이미지를 보고 바로 깨끗한 이미지로 바꾸는 모델”이라면, D3Net은 중간에 **“이 이미지가 어떤 방식으로 망가졌는지 먼저 분석하고, 그에 맞는 복원 경로를 선택하는 모델”**에 가깝다.

### 3.2 CDDA: Cross-Domain Degradation Analyzer

CDDA는 D3Net의 핵심 모듈 중 하나다. 역할은 **이미지의 degradation을 공간 영역과 주파수 영역 양쪽에서 분석하는 것**이다.

일반 CNN은 주로 spatial domain, 즉 픽셀 공간에서 패턴을 본다. 하지만 degradation은 frequency domain에서 더 분명하게 드러나는 경우가 있다. D3Net 논문은 noise는 high-frequency component로 나타나고, blur는 high-frequency weakening으로 나타나며, raindrop은 horizontal/diagonal stripe 형태로, haze는 low-frequency component 감소로 관찰될 수 있다고 설명한다. ([arXiv][2])

SIDL에 대응해서 생각하면 다음처럼 해석할 수 있다.

| SIDL 오염      | 공간 영역 특징                     | 주파수 영역 관점              |
| ------------ | ---------------------------- | ---------------------- |
| Dust         | 작은 점, blob, local occlusion  | 국소 고주파 성분 증가           |
| Scratch      | 선형 edge-like artifact        | 특정 방향성 고주파 성분          |
| Water drop   | halo, local blur, refraction | 고주파 약화 + 국소 패턴         |
| Fingerprint  | smear, contrast drop         | 중저주파 왜곡 + texture 흐림   |
| Mixed debris | 여러 artifact 혼합               | frequency pattern이 복합적 |

CDDA는 이런 frequency feature와 spatial feature를 cross-attention 방식으로 결합해 두 종류의 prompt를 만든다.

| Prompt                        | 의미                                           |
| ----------------------------- | -------------------------------------------- |
| Degradation Correction Prompt | “어디를 어떻게 고쳐야 하는가”에 가까운 correction 정보         |
| Strategy Prompt               | “어떤 복원 전략을 선택해야 하는가”에 가까운 global strategy 정보 |

즉 CDDA는 “이 이미지는 scratch가 강하니까 edge-like artifact를 줄여야 한다”, “이 이미지는 water drop이 강하니까 local blur와 halo를 복원해야 한다” 같은 정보를 네트워크 내부 feature 형태로 만들어 주는 역할을 한다.

### 3.3 DDM: Dynamic Decomposition Mechanism

DDM은 CDDA가 만든 prompt를 바탕으로 실제 복원 경로를 동적으로 선택하는 모듈이다. 논문에서는 DDM이 **Decision Units**와 **Adaptive Decomposition Blocks**로 구성된다고 설명한다. ([arXiv][2])

쉽게 말하면 DDM은 여러 개의 복원 block을 준비해두고, 입력 이미지 상태에 따라 어떤 block을 얼마나 사용할지 결정한다.

예를 들어,

* dust 이미지에서는 local spot 제거에 유리한 block이 더 활성화될 수 있고,
* scratch 이미지에서는 선형 artifact 제거에 유리한 block이 더 활성화될 수 있으며,
* water drop 이미지에서는 blur/halo 보정에 유리한 block이 더 활성화될 수 있다.

D3Net은 학습 중 discrete한 선택을 바로 하면 gradient가 끊기기 때문에, **Gumbel-Softmax**를 사용해 미분 가능한 방식으로 block 선택을 학습한다. 논문은 Gumbel-Softmax를 통해 training 중에는 continuous gradient를 유지하고, inference 때는 hard decision으로 바꿀 수 있다고 설명한다. ([arXiv][2])

DDM의 효과는 논문 ablation에서도 꽤 명확하다. DDM을 제거한 경우 평균 31.38 PSNR / 0.892 SSIM / 39.30M params / 41.99G FLOPs였고, DDM을 사용한 경우 32.36 PSNR / 0.901 SSIM / 37.80M params / 33.67G FLOPs로 성능과 계산량이 모두 개선되었다고 보고된다. ([arXiv][2])

핵심만 요약하면:

> CDDA는 “이미지가 어떻게 망가졌는지 분석”하고, DDM은 “그 분석 결과에 따라 복원 전략을 동적으로 선택”한다.

## 4. 전체 실험 순서

## Phase 0. 실험 목표 확정

### 목표

D3Net을 SIDL 데이터셋에 fine-tuning하여 dirty lens restoration 성능을 평가한다.

### 핵심 질문

1. 기존 all-in-one restoration 모델인 D3Net은 SIDL dirty lens domain에 zero-shot으로 일반화되는가?
2. SIDL fine-tuning을 하면 성능이 얼마나 개선되는가?
3. D3Net의 frequency-spatial degradation decomposition 구조는 SIDL의 contamination type별로 어떤 차이를 보이는가?
4. 선택적으로 DINOv2 feature loss를 추가하면 PSNR/SSIM 또는 시각적 품질이 개선되는가?

### 최종 산출물

* 실험보고서: 최대 8페이지
* 포스터
* 발표자료
* 5분 내외 발표영상

## Phase 1. Repository 및 환경 구축

### 목표

D3Net 공식 코드가 정상적으로 실행되는지 확인한다.

### 작업

1. D3Net repository clone
2. README 기준 dependency 설치
3. 기본 training script 실행 가능 여부 확인
4. sample image inference 또는 dummy training 실행

D3Net repository는 training code를 제공하며, README에는 dataset을 `data/train`에 배치하고 `train.py`를 실행하는 예시 command가 제시되어 있다. ([GitHub][3])

### 체크포인트

* `train.py`가 import error 없이 실행되는가?
* GPU memory error가 발생하지 않는가?
* dataloader가 정상적으로 batch를 반환하는가?
* output image 저장 코드가 있는가?
* checkpoint 저장 및 resume이 가능한가?

### 실패 시 우선 확인할 것

| 문제                   | 가능 원인                                |
| -------------------- | ------------------------------------ |
| CUDA OOM             | patch size, batch size 과다            |
| image shape mismatch | RGB/BGR, HWC/CHW 변환 오류               |
| loss가 NaN            | normalization, learning rate, AMP 문제 |
| output이 이상한 색        | RGB/BGR 또는 0~1/0~255 scale 오류        |
| PSNR이 비정상적으로 낮음      | degraded-clean pair mismatch 가능성     |

## Phase 2. SIDL 데이터셋 구조 파악

### 목표

SIDL의 degraded-clean pair를 D3Net dataloader가 읽을 수 있는 구조로 변환한다.

SIDL은 240 train scenes, 20 validation scenes, 40 test scenes로 나뉘며, difficulty는 clean-dirty PSNR 기준으로 Easy/Medium/Hard로 구분된다. ([SIDL Benchmark][1])

### 권장 데이터 사용 방식

처음부터 full-resolution 4032×3024 이미지를 사용하지 말고, SIDL에서 제공하는 **512×512 patchified images**를 사용한다. SIDL 사이트는 효율적인 batch training을 위해 pre-cropped 512×512 train/validation/test patch를 제공한다. ([SIDL Benchmark][1])

### 권장 폴더 구조

```text
data/
  SIDL/
    train/
      degraded/
      clean/
    val/
      degraded/
      clean/
    test/
      degraded/
      clean/
    metadata/
      train_meta.csv
      val_meta.csv
      test_meta.csv
```

### metadata에 넣으면 좋은 정보

| column             | 설명                                           |
| ------------------ | -------------------------------------------- |
| degraded_path      | 오염 이미지 경로                                    |
| clean_path         | GT clean image 경로                            |
| contamination_type | fingerprint / dust / scratch / water / mixed |
| difficulty         | easy / medium / hard                         |
| scene_id           | scene 단위 식별자                                 |
| patch_id           | patch 단위 식별자                                 |

### 중요한 점

SIDL의 test set은 leaderboard 제출용일 수 있으므로, 과제 실험에서는 먼저 train/val 중심으로 실험하고, test는 최종 결과 생성용으로만 사용하는 것이 안전하다. 보고서에는 validation set 기준 PSNR/SSIM을 main result로 제시하면 된다.

## Phase 3. D3Net Dataloader 수정

### 목표

기존 D3Net dataloader를 SIDL paired restoration task에 맞게 수정한다.

### 기본 입력/출력

```text
Input  : degraded image
Target : clean image
Output : restored image
```

### Dataloader 요구사항

1. degraded image와 clean image를 pair로 읽기
2. 동일한 crop/augmentation을 두 이미지에 적용
3. image를 RGB 기준으로 통일
4. tensor scale을 0~1로 통일
5. validation에서는 random crop 없이 deterministic하게 평가
6. metadata를 반환할 수 있으면 type별/difficulty별 분석이 쉬움

### Dataset 반환 형식 예시

```python
return {
    "input": degraded_tensor,
    "target": clean_tensor,
    "type": contamination_type,
    "difficulty": difficulty,
    "filename": filename,
}
```

### sanity check

학습 전에 반드시 degraded와 clean pair를 나란히 저장해서 확인한다.

```text
debug_pairs/
  000_input.png
  000_target.png
  001_input.png
  001_target.png
```

이 단계가 매우 중요하다. Paired restoration에서 pair가 하나라도 어긋나면 loss는 내려가더라도 결과가 흐릿하거나 이상하게 수렴한다.

## Phase 4. Overfit Sanity Check

### 목표

전체 학습 전에 코드와 데이터 pairing이 정상인지 확인한다.

### 방법

train image 20~50개만 사용해서 D3Net을 overfit시킨다.

### 기대 결과

| 항목           | 정상 징후                               |
| ------------ | ----------------------------------- |
| train loss   | 빠르게 감소                              |
| train PSNR   | 확실히 증가                              |
| output image | target과 점점 유사해짐                     |
| color        | target과 색감이 비슷함                     |
| artifact     | dust/scratch/water artifact가 일부 제거됨 |

### 실패 판단

| 증상             | 의심 원인                                      |
| -------------- | ------------------------------------------ |
| loss가 거의 안 내려감 | learning rate, dataloader, target mismatch |
| output이 회색/검정  | normalization 오류                           |
| output이 색이 틀어짐 | RGB/BGR 오류                                 |
| PSNR이 말이 안 됨   | 0~1/0~255 scale 오류                         |
| train도 복원이 안 됨 | model input/output range 불일치               |

이 sanity check를 통과하기 전까지는 본 학습으로 넘어가면 안 된다.

## Phase 5. Zero-Shot Evaluation

### 목표

기존 D3Net이 SIDL에 바로 일반화되는지 확인한다.

### 실험

기존 pretrained checkpoint가 있다면 SIDL validation set에 바로 inference한다.

### 기록할 항목

| Metric                | 설명                     |
| --------------------- | ---------------------- |
| PSNR                  | pixel-level fidelity   |
| SSIM                  | structural similarity  |
| type별 PSNR/SSIM       | contamination type별 성능 |
| difficulty별 PSNR/SSIM | Easy/Medium/Hard별 성능   |
| qualitative crop      | 시각적 artifact 비교        |

### 보고서에서의 의미

Zero-shot 성능이 낮으면 오히려 보고서 스토리가 좋아진다.
“기존 all-in-one restoration benchmark에서 강한 모델도 SIDL dirty-lens domain에는 직접 일반화되지 않으며, domain-specific fine-tuning이 필요하다”는 주장을 할 수 있다.

## Phase 6. Vanilla D3Net Fine-Tuning

### 목표

D3Net을 SIDL train set으로 fine-tuning하여 domain adaptation 효과를 확인한다.

### 기본 설정

| 항목         | 권장값                          |
| ---------- | ---------------------------- |
| Input size | 512×512 patch                |
| Batch size | GPU에 맞춰 1~8                  |
| Optimizer  | 기존 D3Net 설정 우선 사용            |
| Loss       | 기존 reconstruction loss 우선 사용 |
| Epoch      | 먼저 짧게 10~30 epoch 테스트 후 확장   |
| Validation | 매 epoch 또는 일정 interval마다 수행  |
| Checkpoint | best PSNR 기준 저장              |

### 실험 이름

```text
E1_D3Net_ZeroShot
E2_D3Net_SIDL_Finetune
```

### 핵심 비교

| 비교                                   | 해석                                    |
| ------------------------------------ | ------------------------------------- |
| zero-shot vs fine-tuned              | SIDL domain adaptation 효과             |
| Easy vs Medium vs Hard               | contamination severity에 따른 robustness |
| fingerprint/dust/scratch/water/mixed | 오염 유형별 취약점                            |

## Phase 7. Loss Ablation

### 목표

Dirty lens restoration에서 어떤 loss가 적합한지 비교한다.

### 실험 후보

| 실험          | Loss                |
| ----------- | ------------------- |
| E3          | L1                  |
| E4          | Charbonnier         |
| E5          | L1 + SSIM           |
| E6 optional | L1 + frequency loss |

### 예상 trade-off

| Loss           | 장점                               | 단점                           |
| -------------- | -------------------------------- | ---------------------------- |
| L1             | 안정적, PSNR에 유리                    | perceptual quality가 심심할 수 있음 |
| Charbonnier    | outlier에 robust, 복원 task에서 자주 사용 | L1 대비 개선이 작을 수 있음            |
| L1 + SSIM      | 구조 보존 강조                         | PSNR이 떨어질 가능성                |
| Frequency loss | scratch/water artifact에 도움 가능    | 구현 및 weight tuning 필요        |

### 우선순위

시간이 부족하면 **L1 vs Charbonnier**만 비교해도 충분하다.

## Phase 8. Contamination Type별 분석

### 목표

D3Net이 어떤 오염에 강하고 약한지 분석한다.

### 분석 기준

| Type        | 봐야 할 qualitative point                  |
| ----------- | --------------------------------------- |
| Dust        | 작은 점이 제거되는가? 주변 texture가 뭉개지는가?         |
| Scratch     | 선형 artifact가 줄어드는가? 실제 edge도 같이 사라지는가?  |
| Water       | halo와 local blur가 줄어드는가? 왜곡된 영역이 복원되는가? |
| Fingerprint | smear와 contrast 저하가 개선되는가?              |
| Mixed       | 여러 artifact가 동시에 있을 때 과보정이 생기는가?        |

### 보고서 포인트

단순 평균 PSNR보다 type별 결과가 더 중요하다. SIDL은 실제 dirty lens contamination dataset이므로, 모델이 어떤 오염에 취약한지 분석하는 것이 과제의 핵심 실험으로 보인다.

## Phase 9. Difficulty별 분석

### 목표

Easy/Medium/Hard contamination severity에 따라 D3Net 성능이 어떻게 변하는지 분석한다.

SIDL은 clean-dirty PSNR 기준으로 Easy, Medium, Hard difficulty band를 제공한다. 사이트 설명에 따르면 기존 SOTA 모델들도 Medium/Hard contamination에서 성능 저하가 두드러지며, 특히 water droplet 주변 halo와 fingerprint edge smearing이 남는 문제가 관찰된다. ([SIDL Benchmark][1])

### 표 구성 예시

| Model                 | Easy PSNR | Medium PSNR | Hard PSNR | Avg PSNR |
| --------------------- | --------: | ----------: | --------: | -------: |
| D3Net zero-shot       |         - |           - |         - |        - |
| D3Net fine-tuned      |         - |           - |         - |        - |
| D3Net + best loss     |         - |           - |         - |        - |
| D3Net + DINO optional |         - |           - |         - |        - |

### 해석 방향

* Easy에서만 좋아지고 Hard에서 약하면: 모델이 약한 오염 제거에는 성공하지만 severe contamination 복원에는 부족함.
* Hard에서 개선폭이 크면: D3Net의 dynamic degradation decomposition이 강한 오염에서 효과적일 가능성.
* Medium/Hard에서 SSIM이 낮으면: 구조적 복원 실패 또는 halo/smearing 잔류 가능성.

## Phase 10. Optional: DINOv2 Feature Loss 추가

### 목표

D3Net fine-tuning 이후, DINOv2 feature loss를 추가하여 semantic/structural prior가 복원 품질에 도움 되는지 확인한다.

### 구조

```text
degraded image → D3Net → restored image
clean image ───────────────→ GT

restored image → frozen DINOv2 → feature
clean image    → frozen DINOv2 → feature

L_DINO = distance(DINO(restored), DINO(clean))
```

### 전체 loss

```text
L_total = L_rec + λ_dino * L_DINO
```

### 추천 ablation

| 실험  | λ_dino |
| --- | -----: |
| E7  |   0.00 |
| E8  |   0.01 |
| E9  |   0.05 |
| E10 |   0.10 |

### 주의점

DINO loss는 PSNR을 반드시 올려주는 loss가 아니다. DINO feature는 semantic/structural consistency에는 도움이 될 수 있지만, pixel-level fidelity를 직접 최적화하지 않는다. 따라서 DINO loss를 크게 주면 결과가 시각적으로는 자연스러워질 수 있어도 PSNR은 떨어질 수 있다.

### 보고서에서의 해석

DINO 결과가 좋으면:

> DINOv2 feature prior가 dirty lens restoration에서 scene structure 보존에 도움을 주었다.

DINO 결과가 나쁘면:

> DINO feature는 semantic consistency에는 유리하지만, SIDL의 평가 지표인 PSNR/SSIM 중심 pixel-level restoration과는 trade-off를 보였다.

즉 성공해도 실패해도 분석 포인트가 생긴다.

## 5. 최종 실험 목록

### 필수 실험

| ID | 실험명                          | 목적                            |
| -- | ---------------------------- | ----------------------------- |
| E1 | D3Net Zero-Shot              | 기존 model의 SIDL 일반화 성능 확인      |
| E2 | D3Net Fine-Tuning            | SIDL domain adaptation 효과 확인  |
| E3 | L1 Loss Fine-Tuning          | 기본 pixel restoration baseline |
| E4 | Charbonnier Loss Fine-Tuning | robust reconstruction loss 비교 |
| E5 | Type-wise Evaluation         | contamination type별 취약점 분석    |
| E6 | Difficulty-wise Evaluation   | Easy/Medium/Hard severity 분석  |

### 선택 실험

| ID  | 실험명                    | 목적                                                |
| --- | ---------------------- | ------------------------------------------------- |
| E7  | D3Net + SSIM Loss      | 구조 보존 효과 확인                                       |
| E8  | D3Net + Frequency Loss | scratch/water artifact에 frequency loss가 도움 되는지 확인 |
| E9  | D3Net + DINOv2 Loss    | semantic/structural prior 효과 확인                   |
| E10 | DINO λ Ablation        | DINO loss weight trade-off 분석                     |

## 6. 결과 정리 방식

## 6.1 정량 결과 표

### 전체 평균 결과

| Model               | PSNR ↑ | SSIM ↑ | Params | Notes           |
| ------------------- | -----: | -----: | -----: | --------------- |
| D3Net Zero-Shot     |      - |      - |      - | pretrained only |
| D3Net Fine-Tuned    |      - |      - |      - | SIDL adaptation |
| D3Net + Charbonnier |      - |      - |      - | loss ablation   |
| D3Net + DINOv2      |      - |      - |      - | optional        |

### contamination type별 결과

| Model            | Fingerprint | Dust | Scratch | Water | Mixed | Avg |
| ---------------- | ----------: | ---: | ------: | ----: | ----: | --: |
| D3Net Zero-Shot  |           - |    - |       - |     - |     - |   - |
| D3Net Fine-Tuned |           - |    - |       - |     - |     - |   - |
| Best Model       |           - |    - |       - |     - |     - |   - |

### difficulty별 결과

| Model            | Easy | Medium | Hard | Avg |
| ---------------- | ---: | -----: | ---: | --: |
| D3Net Zero-Shot  |    - |      - |    - |   - |
| D3Net Fine-Tuned |    - |      - |    - |   - |
| Best Model       |    - |      - |    - |   - |

## 6.2 정성 결과 구성

각 figure는 다음 순서로 구성하면 좋다.

```text
Input degraded | D3Net zero-shot | D3Net fine-tuned | Best model | Clean GT
```

### crop 추천

| 오염          | crop 위치                  |
| ----------- | ------------------------ |
| Dust        | 작은 점이 많은 영역              |
| Scratch     | 선형 스크래치가 배경/edge와 겹치는 영역 |
| Water       | 물방울 경계와 halo 주변          |
| Fingerprint | 흐릿한 smear가 생긴 영역         |
| Mixed       | 여러 오염이 겹친 영역             |

## 7. 보고서 구조 제안

## 1. Introduction

* 스마트폰 카메라의 렌즈 오염은 실제 사용 환경에서 자주 발생하지만, 기존 image restoration 연구는 controlled degradation에 집중하는 경우가 많다.
* SIDL은 실제 dirty lens contamination을 다루는 paired dataset이다.
* 본 프로젝트는 all-in-one restoration 모델인 D3Net을 SIDL에 fine-tuning하여 dirty lens restoration 문제에 적용한다.

## 2. Method

* D3Net 구조 요약
* CDDA 설명
* DDM 설명
* SIDL fine-tuning setting
* optional DINOv2 feature loss

## 3. Experimental Setup

* Dataset: SIDL 512×512 patch
* Train/val split
* Metrics: PSNR, SSIM
* Training details
* 비교 실험 목록

## 4. Results

* zero-shot vs fine-tuning
* loss ablation
* type별 결과
* difficulty별 결과
* qualitative result

## 5. Discussion

* D3Net이 잘 복원한 오염
* D3Net이 실패한 오염
* Hard contamination에서의 한계
* DINO loss의 효과 또는 trade-off
* SIDL dirty lens domain과 기존 all-in-one benchmark 간 domain gap

## 6. Conclusion

* D3Net fine-tuning은 SIDL dirty lens restoration에 효과적이었는지 요약
* frequency-spatial degradation decomposition이 dirty lens 복원에 갖는 가능성 정리
* 향후 연구: DINO prior, lens-contamination-specific prompt, RAW-domain restoration 등

## 8. 가장 중요한 구현 우선순위

1. **D3Net 코드 실행 확인**
2. **SIDL dataloader 이식**
3. **overfit sanity check**
4. **D3Net zero-shot evaluation**
5. **D3Net SIDL fine-tuning**
6. **type/difficulty별 분석**
7. **loss ablation**
8. **DINOv2 loss optional 추가**

절대 처음부터 DINO까지 넣고 시작하면 안 된다. 먼저 D3Net vanilla fine-tuning이 정상적으로 돌아가야 한다. 그 다음 DINO를 붙여야 실패 원인을 분리할 수 있다.

## 9. 예상 리스크와 대응

| 리스크                                    | 설명                                                       | 대응                              |
| -------------------------------------- | -------------------------------------------------------- | ------------------------------- |
| D3Net 기존 degradation과 SIDL의 domain gap | D3Net은 주로 noise/rain/haze/blur/low-light benchmark에서 평가됨 | zero-shot 성능을 domain gap 근거로 사용 |
| dataloader pairing 오류                  | paired restoration에서 가장 치명적                              | debug pair image 저장             |
| full-res 학습 불가능                        | 4032×3024는 GPU 부담 큼                                      | 512×512 patch 사용                |
| DINO loss가 PSNR 저하                     | perceptual feature와 pixel metric 간 trade-off             | optional ablation으로만 사용         |
| Hard contamination 복원 실패               | severe water/fingerprint는 정보 손실이 큼                       | 실패 사례 분석으로 보고서에 포함              |
| mixed debris 성능 저하                     | 여러 오염이 중첩되어 decomposition이 어려움                           | type-wise analysis에서 별도 논의      |

## 10. 최종 추천 결론

이번 프로젝트의 가장 안전한 목표는 **“D3Net을 SIDL에 성공적으로 fine-tuning하고, dirty lens contamination type별/difficulty별 성능을 분석하는 것”**이다.

DINO는 넣으면 좋지만, 메인 성공 조건으로 두면 위험하다. 메인 스토리는 D3Net의 **CDDA + DDM 구조가 SIDL의 실제 렌즈 오염 복원에 얼마나 잘 transfer되는지 검증**하는 것으로 잡고, DINO는 시간이 남을 때 **DINO-Regularized D3Net**이라는 확장 실험으로 붙이는 것이 가장 현실적이다.

[1]: https://sidl-benchmark.github.io/ "SIDL: A Real-World Dataset for Restoring Smartphone Images with Dirty Lenses"
[2]: https://arxiv.org/html/2502.19068v1 "Dynamic Degradation Decomposition Network for All-in-One Image Restoration"
[3]: https://github.com/codeshop715/D3Net "GitHub - MingC715/D3Net · GitHub"
