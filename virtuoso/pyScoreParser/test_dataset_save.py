
import argparse
from .data_class import DataSet, DEFAULT_SCORE_FEATURES, DEFAULT_PERFORM_FEATURES
from pathlib import Path
from .data_for_training import PairDataset
from .feature_extraction import ScoreExtractor, PerformExtractor

target = 'refactory/test_examples'
xml_name = 'musicxml_cleaned.musicxml'

print(f'save and load {target}')
class SmallSet(DataSet):
    def __init__(self, path=target):
        super().__init__(path, save=True)

    def load_data(self):
        path = Path(self.path)
        xml_list = sorted(path.glob(f'**/{xml_name}'))
        score_midis = [xml.parent / 'midi_cleaned.mid' for xml in xml_list]
        composers = ['Beethoven']

        perform_lists = []
        for xml in xml_list:
            midis = sorted(xml.parent.glob('*.mid')) + sorted(xml.parent.glob('*.MID'))
            midis = [str(midi) for midi in midis if midi.name not in ['midi.mid', 'midi_cleaned.mid']]
            midis = [midi for midi in midis if not 'XP' in midi]
            perform_lists.append(midis)

        # Path -> string wrapper
        xml_list = [str(xml) for xml in xml_list]
        score_midis = [str(midi) for midi in score_midis]
        return xml_list, score_midis, perform_lists, composers

if __name__ == '__main__':
    dataset = SmallSet()

    for piece in dataset.pieces:
        piece.extract_perform_features(DEFAULT_PERFORM_FEATURES)
        piece.extract_score_features(DEFAULT_SCORE_FEATURES)
    pair_data = PairDataset(dataset)

   
    pair_data.update_dataset_split_type()
    pair_data.update_mean_stds_of_entire_dataset()
    pair_data.save_features_for_virtuosoNet('test')