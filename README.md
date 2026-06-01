# D3Net Fine-Tuning for SIDL Dirty Lens Image Restoration

## Project Overview

본 프로젝트는 All-in-One Image Restoration 모델인 **D3Net (Dynamic Degradation Decomposition Network)** 을 **SIDL Smartphone Image Dirty Lens Dataset**에 fine-tuning하여, 실제 스마트폰 렌즈 오염으로 인해 발생한 이미지 열화를 복원하는 것을 목표로 한다.

SIDL 데이터셋은 fingerprint, dust, scratch, water drop, mixed debris 등 실제 렌즈 오염으로 촬영된 degraded-clean image pair를 포함한다. 본 프로젝트에서는 D3Net이 기존 all-in-one restoration benchmark에서 학습한 degradation decomposition 능력을 dirty lens restoration 문제에 얼마나 잘 transfer할 수 있는지 실험적으로 확인한다.

## Main Idea

D3Net은 입력 이미지의 degradation을 고정된 방식으로 처리하지 않고, 이미지마다 다른 degradation 특성을 분석하여 동적으로 복원 전략을 선택하는 모델이다.

특히 D3Net의 핵심 구조인 **CDDA (Cross-Domain Degradation Analyzer)** 와 **DDM (Dynamic Decomposition Mechanism)** 은 spatial-domain feature와 frequency-domain degradation feature를 함께 활용한다. 이는 dust, scratch, water drop, fingerprint처럼 공간적 패턴과 주파수 특성이 모두 중요한 dirty lens contamination 문제와 잘 맞는다고 판단하였다.

본 프로젝트의 기본 실험은 다음과 같다.

- D3Net zero-shot evaluation on SIDL
- D3Net fine-tuning on SIDL
- contamination type별 성능 분석
- Easy / Medium / Hard difficulty별 성능 분석
- optional: DINOv2 feature loss를 활용한 semantic prior 추가 실험

## References

- SIDL Benchmark  
  https://sidl-benchmark.github.io/

- All-in-One Image Restoration Survey  
  https://github.com/Harbinzzy/All-in-One-Image-Restoration-Survey

- D3Net Repository  
  https://github.com/codeshop715/D3Net

- D3Net Paper: Dynamic Degradation Decomposition Network for All-In-One Image Restoration  
  https://arxiv.org/html/2502.19068v1
