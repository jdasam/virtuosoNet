# VirtuosoNet

An AI pianist system that reads a MusicXML score and generates a human-like expressive performance MIDI.

Based on:
- [**Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance**](http://proceedings.mlr.press/v97/jeong19a.html) (ICML 2019)
- [**VirtuosoNet: A Hierarchical RNN-based System for Modeling Expressive Piano Performance**](http://archives.ismir.net/ismir2019/paper/000112.pdf) (ISMIR 2019)

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

```python
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
