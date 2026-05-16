# GitHub Public Release Design

**Date:** 2026-05-17
**Branch:** batchfy → master

## Goal

공개 GitHub 저장소로 정리. 대상: 논문 재현 연구자 + 일반 사용자 모두.

---

## 섹션 1: 노트북 분리 (private repo)

- `gh repo create virtuosonet-dev --private` 로 private repo 생성
- 현재 `notebooks/` 전체를 거기 push
- 이 repo에서는 `notebooks/`를 git untrack + `.gitignore` 추가

---

## 섹션 2: 파일 정리 (batchfy 브랜치)

### 삭제
- `prototype_script.py`
- `upload_checkpoint.py`
- `Schumann/`
- `rencon/` 내부 output 파일 (.mid, .png) → `.gitignore`에 추가

### scripts/ 폴더로 이동
- `elongate_pedal.py`
- `modify_xml.py`

### 유지
- `download_and_test.py` (루트)
- `test_pieces/bps_17_1/` 하나만 예시로 유지

### .gitignore 추가
- `notebooks/`
- `test_pieces/*` (bps_17_1 제외)
- `rencon/`
- `Schumann/`

---

## 섹션 3: README 재작성

기존 README는 구버전 API(`model_run.py`, PyTorch 0.4.1 등) 기준으로 작성되어 완전 재작성.

### 구성

```
# VirtuosoNet
한 줄 소개 + 논문 링크 (ICML 2019, ISMIR 2019)

> 🏆 RenCon 2025 최종 평가 1위 알고리즘
> 결과: https://arxiv.org/abs/2605.02059

## Quick Start
uv/pip 설치 → download_and_test.py 실행

## Installation
pyproject.toml 기반 (uv / pip)

## Usage
python -m virtuoso 명령어 + 주요 옵션 표

## Pretrained Model
RenCon 2025 제출 및 1위 달성 버전
HuggingFace: dasaem/virtuosonet
모델: HAN+GRU

## Supported Composers
16명 목록

## Citation
ICML 2019, ISMIR 2019 bibtex
```

---

## 섹션 4: master 머지

- batchfy에서 정리 완료 후 master로 머지
- 접근 방법 A: batchfy에서 정리 → master 머지
