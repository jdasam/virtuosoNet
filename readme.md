pyScoreParser
======================

### musicxml & performance midi alignment managing tool.
this repository can be used for:
- parse score xml into a note sequence while preserving score information.
- match midi notes in performance midi to score note, and make combined representation.

### usage

```
# load piece, which contains score and performances

from pyScoreParser.data_class import PieceData


performances = ['test_examples/Beethoven/32-1/DupreeF03.mid']  # should be list
score_xml = 'test_examples/Beethoven/32-1/musicxml_cleaned.musicxml'

piece = PieceData(xml_path)
```
> test_examples/Beethoven/32-1/musicxml_cleaned_score.mid
>
> Number of mismatched notes:  43
>
> Performance path is  test_examples/Beethoven/32-1/DupreeF03.mid
>
> Number of Matched Notes: 5296, unmatched notes: 67

```
print(piece.__dict__.keys())
```
> dict_keys(['meta', 'performances', 'score', 'xml_obj', 'xml_notes', 'num_notes', 'notes_graph', 'score_midi_notes', 'score_match_list', 'score_pairs', 'measure_positions', 'beat_positions', 'section_positions', 'score_features'])

```
# check first note in score

print(len(piece.score.xml_notes))
print(piece.score.xml_notes[0])
```
> 5363
>
> {duration: 60, midi_ticks: 27.5, seconds: 0.20833333333333334, pitch: Eb4, MIDI pitch: 63, voice: 5, velocity: 64} (@time: 0.0) (@xml: 0)

```
# check first note in performance
print(len(piece.performances[0].midi_notes))
print(piece.performances[0].midi_notes[0])
```
> 5531
>
> Note(start=0.504807, end=0.560897, pitch=63, velocity=91)

```
# check alignment
print(len(piece.performances[0].match_between_xml_perf))
print(piece.performances[0].match_between_xml_perf[:10])
```
> 5363
>
> [0, 1, 2, 3, 4, 7, 6, 5, 8, 9]a


## load dataset

checkout test_dataset.py

use [yamaha dataset](https://github.com/mac-marg-pianist/chopin_cleaned) or  [emotion dataset](https://github.com/mac-marg-pianist/EmotionData).

if save, save ScoreData and PerformData into .dat format.

if not save, load from pre-saved file