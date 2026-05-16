# GitHub Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** batchfy 브랜치를 정리하고 README를 새로 작성한 뒤 master로 머지하여 GitHub 공개 준비를 완료한다.

**Architecture:** 파일 정리(노트북 분리, 불필요 스크립트 삭제, test_pieces 축소) → parser.py 하드코딩 경로 수정 → README 재작성 → master 머지 순서로 진행. 각 Task는 독립적으로 커밋한다.

**Tech Stack:** git, gh CLI, Python 3.10, uv

---

## File Map

| 파일 | 변경 |
|------|------|
| `prototype_script.py` | 삭제 |
| `upload_checkpoint.py` | 삭제 |
| `elongate_pedal.py` | `scripts/elongate_pedal.py`로 이동 |
| `modify_xml.py` | `scripts/modify_xml.py`로 이동 |
| `notebooks/` | git untrack + `.gitignore` 추가, 내용은 private repo로 이동 |
| `test_pieces/` | bps_17_1 제외 전부 git untrack + `.gitignore` 추가 |
| `.gitignore` | notebooks/, test_pieces/*, Schumann/, rencon/*.mid, rencon/*.png 추가 |
| `virtuoso/parser.py` | 하드코딩된 `/home/svcapp/`, `/home/teo/` 경로 제거 |
| `README.md` | 완전 재작성 |

---

## Task 1: private repo에 노트북 백업

**Files:**
- 없음 (git 외부 작업)

- [ ] **Step 1: gh로 private repo 생성**

```bash
gh repo create virtuosonet-dev --private --description "VirtuosoNet development notebooks and experiments"
```

Expected: `✓ Created repository dasaem/virtuosonet-dev on GitHub`

- [ ] **Step 2: 노트북을 임시 디렉토리에 복사 후 push**

```bash
cd /tmp
git clone git@github.com:dasaem/virtuosonet-dev.git
cp -r /home/teo/userdata/virtuosoNet/notebooks /tmp/virtuosonet-dev/
cd virtuosonet-dev
git add notebooks/
git commit -m "backup: move development notebooks from main repo"
git push origin main
cd /home/teo/userdata/virtuosoNet
```

Expected: 노트북 파일들이 https://github.com/dasaem/virtuosonet-dev 에 올라감

---

## Task 2: notebooks/ git untrack 및 .gitignore 추가

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: git에서 notebooks 디렉토리 untrack**

```bash
git rm -r --cached notebooks/
```

Expected: `rm 'notebooks/evaluation_metric.ipynb'` 등 파일 목록 출력

- [ ] **Step 2: .gitignore에 notebooks/ 추가**

`.gitignore` 파일에 다음 줄을 추가한다 (기존 내용 유지, 맨 아래에 추가):

```
notebooks/
```

- [ ] **Step 3: 커밋**

```bash
git add .gitignore
git commit -m "chore: move notebooks to private repo, remove from tracking"
```

---

## Task 3: 불필요한 루트 스크립트 정리

**Files:**
- Delete: `prototype_script.py`, `upload_checkpoint.py`
- Create: `scripts/` 디렉토리
- Move: `elongate_pedal.py` → `scripts/elongate_pedal.py`
- Move: `modify_xml.py` → `scripts/modify_xml.py`

- [ ] **Step 1: prototype_script.py, upload_checkpoint.py 삭제**

```bash
git rm prototype_script.py upload_checkpoint.py
```

Expected: `rm 'prototype_script.py'`, `rm 'upload_checkpoint.py'`

- [ ] **Step 2: scripts/ 디렉토리 만들고 유틸리티 스크립트 이동**

```bash
mkdir -p scripts
git mv elongate_pedal.py scripts/elongate_pedal.py
git mv modify_xml.py scripts/modify_xml.py
```

- [ ] **Step 3: 커밋**

```bash
git commit -m "chore: remove dev scripts, move utilities to scripts/"
```

---

## Task 4: test_pieces 정리 (bps_17_1만 유지)

**Files:**
- Modify: `.gitignore`

현재 git에 273개의 test_pieces 파일이 추적되고 있다. bps_17_1 하나만 남기고 나머지를 untrack한다.

- [ ] **Step 1: bps_17_1 제외 모든 test_pieces untrack**

```bash
git ls-files test_pieces/ | grep -v "^test_pieces/bps_17_1/" | xargs git rm --cached
```

Expected: 수백 개의 `rm 'test_pieces/...'` 출력

- [ ] **Step 2: .gitignore에 test_pieces 규칙 추가**

`.gitignore`에 다음을 추가한다:

```
test_pieces/*
!test_pieces/bps_17_1/
!test_pieces/bps_17_1/**
Schumann/
rencon/*.mid
rencon/*.png
```

- [ ] **Step 3: 커밋**

```bash
git add .gitignore
git commit -m "chore: keep only bps_17_1 as example, gitignore the rest"
```

---

## Task 5: parser.py 하드코딩 경로 수정

**Files:**
- Modify: `virtuoso/parser.py`

현재 4곳에 `/home/svcapp/...` 또는 `/home/teo/...` 같은 절대 경로가 기본값으로 박혀있다.

- [ ] **Step 1: 하드코딩 경로 수정**

`virtuoso/parser.py` 에서 다음 줄들을 수정한다:

```python
# 18번째 줄 - xml_path 기본값
# 변경 전:
default=Path('/home/svcapp/userdata/dev/virtuosoNet/test_pieces/bps_5_1/musicxml_cleaned.musicxml')
# 변경 후:
default=Path('test_pieces/bps_17_1/musicxml_cleaned.musicxml')

# 22번째 줄 - valid_xml_dir 기본값
# 변경 전:
default=Path('/home/teo/userdata/datasets/chopin_cleaned/')
# 변경 후:
default=Path('datasets/chopin_cleaned/')

# 50번째 줄 - checkpoints_dir 기본값
# 변경 전:
default=Path('/home/teo/userdata/virtuosonet_checkpoints/')
# 변경 후:
default=Path('checkpoints/')

# 54번째 줄 - checkpoint 기본값
# 변경 전:
default=Path('/home/svcapp/userdata/dev/virtuosoNet/isgn_best.pt')
# 변경 후:
default=Path('pretrained_weights/checkpoint_best.pt')
```

- [ ] **Step 2: 수정 결과 확인**

```bash
grep -n "svcapp\|/home/teo" virtuoso/parser.py
```

Expected: 출력 없음

- [ ] **Step 3: 커밋**

```bash
git add virtuoso/parser.py
git commit -m "fix: replace hardcoded absolute paths with relative defaults"
```

---

## Task 6: README 재작성

**Files:**
- Modify: `README.md`

기존 README는 PyTorch 0.4.1, `model_run.py -code=isgn` 등 구버전 API를 가리키므로 완전히 교체한다.

- [ ] **Step 1: README.md 전체 교체**

`README.md`를 아래 내용으로 완전히 교체한다:

````markdown
# VirtuosoNet

An AI pianist system that reads a MusicXML score and generates a human-like expressive performance MIDI.

Based on:
- [**Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance**](http://proceedings.mlr.press/v97/jeong19a.html) (ICML 2019)
- [**VirtuosoNet: A Hierarchical RNN-based system for modeling expressive piano performance**](http://archives.ismir.net/ismir2019/paper/000112.pdf) (ISMIR 2019)

> **🏆 RenCon 2025 — 1st place**
> The pretrained weights in this repo are the algorithm submitted to [RenCon 2025](https://arxiv.org/abs/2605.02059), which ranked 1st in the final evaluation.

---

## Quick Start

Download the pretrained model and run inference on a sample piece in one command:

```bash
python download_and_test.py
```

This downloads the checkpoint from HuggingFace and runs inference on `test_pieces/bps_17_1/musicxml_cleaned.musicxml` (Beethoven Piano Sonata No.17).

---

## Installation

**With uv (recommended):**

```bash
uv sync
```

**With pip:**

```bash
pip install -e .
```

Requires Python 3.10 and CUDA 11.8 (for GPU). See `pyproject.toml` for full dependencies.

---

## Usage

```bash
python -m virtuoso \
  --session_mode=inference \
  --checkpoint=pretrained_weights/han_measnote_gru/checkpoint_best.pt \
  --yml_path=pretrained_weights/han_measnote_gru/han_measnote_gru.yml \
  --xml_path=<path/to/your/score.musicxml> \
  --composer=<ComposerName>
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--xml_path` | Path to input MusicXML file | `test_pieces/bps_17_1/musicxml_cleaned.musicxml` |
| `--composer` | Composer name (see list below) | required |
| `--checkpoint` | Path to model checkpoint `.pt` file | `pretrained_weights/checkpoint_best.pt` |
| `--yml_path` | Path to model config `.yml` file | — |
| `--output_path` | Directory for output MIDI | `test_result/` |
| `--qpm_primo` | Override initial tempo (BPM) | from score |
| `--boolPedal` | Apply pedal threshold (True/False) | `False` |

Output MIDI is saved to `test_result/` by default.

---

## Pretrained Model

The model available on [HuggingFace (`dasaem/virtuosonet`)](https://huggingface.co/dasaem/virtuosonet) is the **HAN+GRU** architecture submitted to RenCon 2025, where it achieved **1st place** in the final evaluation.

Download manually:

```bash
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="dasaem/virtuosonet", filename="checkpoint_best.pt", local_dir="pretrained_weights/han_measnote_gru")
hf_hub_download(repo_id="dasaem/virtuosonet", filename="han_measnote_gru.yml", local_dir="pretrained_weights/han_measnote_gru")
```

---

## Supported Composers

The model was trained on performances by the following 16 composers. Pass the composer name with `--composer`:

`Bach`, `Balakirev`, `Beethoven`, `Brahms`, `Chopin`, `Debussy`, `Glinka`, `Haydn`, `Liszt`, `Mozart`, `Prokofiev`, `Rachmaninoff`, `Ravel`, `Schubert`, `Schumann`, `Scriabin`

> Tip: The composer does not have to match the actual composer of the piece. Among the above, `Bach`, `Beethoven`, `Chopin`, `Haydn`, `Liszt`, `Mozart`, `Ravel`, and `Schubert` have the most training data and tend to give the best results.

---

## A Note on Pedal

Sustain pedal is encoded as MIDI CC 64. Different MIDI players interpret this differently:

- **Logic Pro X**: activates pedal if value > 0
- **Disklavier**: threshold is ~64

If the output sounds too wet (too much pedal), use `--boolPedal=True` to apply a threshold.
If the output sounds too dry (no pedal), your MIDI player may not support pedal CC events.

---

## Citation

If you use the **pretrained weights** to generate performances, please cite the ISMIR 2019 paper (VirtuosoNet):

```bibtex
@inproceedings{jeong2019virtuosonet,
  title={VirtuosoNet: A Hierarchical RNN-based System for Modeling Expressive Piano Performance},
  author={Jeong, Dasaem and Kwon, Taegyun and Kim, Yoojin and Lee, Kyungu and Nam, Juhan},
  booktitle={Proceedings of the 20th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2019}
}
```

If you use the **graph-based score encoding**, please also cite the ICML 2019 paper:

```bibtex
@inproceedings{jeong2019graph,
  title={Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance},
  author={Jeong, Dasaem and Kwon, Taegyun and Kim, Yoojin and Nam, Juhan},
  booktitle={Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year={2019}
}
```

---

Contact: dasaem.jeong@gmail.com
````

- [ ] **Step 2: 커밋**

```bash
git add README.md
git commit -m "docs: rewrite README for public release with RenCon 2025 result"
```

---

## Task 7: master로 머지

- [ ] **Step 1: master 브랜치로 전환**

```bash
git checkout master
```

- [ ] **Step 2: batchfy를 master로 머지**

```bash
git merge batchfy --no-ff -m "merge: batchfy → master for public release"
```

Expected: 충돌 없이 머지 완료

- [ ] **Step 3: origin/master로 push**

```bash
git push origin master
```

- [ ] **Step 4: 확인**

```bash
git log --oneline -5
```

Expected: 방금 머지 커밋이 맨 위에 표시됨
