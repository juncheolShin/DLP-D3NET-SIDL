# D3Net SIDL Final Project Artifacts

이 폴더는 SIDL dirty lens image restoration 최종 프로젝트 산출물 세트를 포함한다.

## Contents

- `report/`: Overleaf 업로드용 LaTeX 보고서
  - `main.tex`
  - `references.bib`
  - `figures/`
  - `tables/`
- `poster/`: 브라우저에서 PDF 출력 가능한 HTML 포스터
  - `index.html`
  - `style.css`
  - `assets/`
- `slides/`: 5분 발표 녹화용 HTML 발표자료
  - `index.html`
  - `style.css`
  - `script.js`
  - `assets/`

## Source Data Used

- Drive example materials:
  - `AAAI_SIDL.pdf`
  - `AAAI2025_SIDL_poster.pdf`
  - `AAAI2025_SIDL_poster.pptx`
- Local experiment summary:
  - `D3Net/reports/sidl_experiment_summary.md`
  - `D3Net/reports/validation_breakdowns/*.md`
  - `D3Net/reports/assets/sidl_examples_best256_panel.png`
- Original D3Net paper figures:
  - arXiv source `2502.19068`
  - Figure 2: `report/figures/d3net_architecture_paper.pdf`
  - Figure 3: `report/figures/d3net_ddm_paper.pdf`
  - HTML용 PNG 변환본은 `poster/assets/`와 `slides/assets/`에 포함
- Original D3Net all-in-one benchmark:
  - AirNet / PromptIR / IDR / InstructIR / D3Net reference comparison
  - SIDL validation과 직접 비교하지 않고 backbone 선택 근거로만 사용

## Overleaf Upload

1. `project_artifacts/report/` 폴더 전체를 zip으로 압축한다.
2. Overleaf에서 새 프로젝트를 만들고 zip 파일을 업로드한다.
3. Compiler를 `XeLaTeX`로 설정한다.
4. `main.tex`를 컴파일한다.

## Poster PDF Export

1. 브라우저에서 `project_artifacts/poster/index.html`을 연다.
2. Print 또는 Save as PDF를 선택한다.
3. 배율은 기본적으로 100%에 맞추고, 배경 그래픽 출력을 켠다.

## Slides Usage

1. 브라우저에서 `project_artifacts/slides/index.html`을 연다.
2. 좌우 방향키로 슬라이드를 넘긴다.
3. PDF가 필요하면 브라우저 Print 또는 Save as PDF를 사용한다.

## TODO

- 작성자/과목 표기는 `20223081 신준철`, `Deep Learning Programming`으로 설정 완료.
- 보고서의 model parameter 수와 GMAC 측정값이 필요하면 별도 측정 후 `Experimental Setup`에 추가.
- 256 low-LR polishing이나 추가 refinement run의 최종값이 새로 확정되면 표를 업데이트.
- qualitative example을 더 엄선하려면 Hard Dust / Hard Mixed failure case를 추가 생성.
- 원 논문 figure를 제출물에 그대로 사용할 때는 수업/학회 제출 규정에 맞춰 citation 및 재사용 가능 범위를 확인.
