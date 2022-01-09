import os
import random
import numpy as np
import pickle

from numpy.lib.npyio import save
from . import dataset_split
from pathlib import Path
from tqdm import tqdm

NORM_FEAT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 
                    'qpm_primo',
                    # "section_tempo",
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 
                            'beat_tempo', 'beat_dynamics', 'measure_tempo', 'measure_dynamics')

VNET_COPY_DATA_KEYS = ('note_location', 'align_matched', 'articulation_loss_weight')
VNET_INPUT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 
                'qpm_primo',
                # "section_tempo",
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                          'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                          'slur_beam_vec',  'composer_vec', 'notation', 
                          'tempo_primo'
                          )

VNET_OUTPUT_KEYS = ('beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut')
VNET_BEAT_KEYS = ('beat_tempo', 'beat_dynamics')
VNET_MEAS_KEYS = ('measure_tempo', 'measure_dynamics')
                            # , 'beat_tempo', 'beat_dynamics', 'measure_tempo', 'measure_dynamics')


class ScorePerformPairData:
    def __init__(self, piece, perform):
        self.piece_path = piece.meta.xml_path
        self.perform_path = perform.midi_path
        self.graph_edges = piece.notes_graph
        self.features = {**piece.score_features, **perform.perform_features}
        self.split_type = None
        self.features['num_notes'] = piece.num_notes


class PairDataset:
    def __init__(self, dataset):
        self.dataset_path = dataset.path
        self.data_pairs = []
        self.feature_stats = None
        for piece in dataset.pieces:
            for performance in piece.performances:
                if performance is not None \
                        and 'align_matched' in performance.perform_features\
                        and len(performance.perform_features['align_matched']) - sum(performance.perform_features['align_matched']) < 800:
                    self.data_pairs.append(ScorePerformPairData(piece, performance))
    
    def __len__(self):
      return len(self.data_pairs)

    def __getitem__(self, idx):
      return self.data_pairs[idx]

    def get_squeezed_features(self, target_feat_keys):
        squeezed_values = dict()
        for feat_type in target_feat_keys:
            squeezed_values[feat_type] = []
        for pair in self.data_pairs:
            for feat_type in target_feat_keys:
                if isinstance(pair.features[feat_type], list):
                    squeezed_values[feat_type] += pair.features[feat_type]
                else:
                    squeezed_values[feat_type].append(pair.features[feat_type])
        return squeezed_values

    def update_mean_stds_of_entire_dataset(self, target_feat_keys=NORM_FEAT_KEYS):
        squeezed_values = self.get_squeezed_features(target_feat_keys)
        self.feature_stats = cal_mean_stds(squeezed_values, target_feat_keys)

    def update_dataset_split_type(self, valid_set_list=dataset_split.VALID_LIST, test_set_list=dataset_split.TEST_LIST):
        for pair in self.data_pairs:
            path = pair.piece_path
            for valid_name in valid_set_list:
                if valid_name in path:
                    pair.split_type = 'valid'
                    break
            else:
                for test_name in test_set_list:
                    if test_name in path:
                        pair.split_type = 'test'
                        break

            if pair.split_type is None:
                pair.split_type = 'train'

    def shuffle_data(self):
        random.shuffle(self.data_pairs)

    def save_features_for_virtuosoNet(self, save_folder, update_stats=True, valid_set_list=dataset_split.VALID_LIST, test_set_list=dataset_split.TEST_LIST):
        '''
        Convert features into format of VirtuosoNet training data
        :return: None (save file)
        '''
        def _flatten_path(file_path):
            return '_'.join(file_path.parts)

        save_folder = Path(save_folder)
        split_types = ['train', 'valid', 'test']

        if not save_folder.is_dir():
            save_folder.mkdir()
            for split in split_types:
                if not (save_folder / split).is_dir():
                    (save_folder / split).mkdir()
    
        if update_stats:
            self.update_mean_stds_of_entire_dataset()
        self.update_dataset_split_type(valid_set_list=valid_set_list, test_set_list=test_set_list)
        
        for pair_data in tqdm(self.data_pairs):
            formatted_data = dict()
            formatted_data['input_data'], formatted_data['output_data'], formatted_data['meas_level_data'], formatted_data['beat_level_data'] = \
                 convert_feature_to_VirtuosoNet_format(pair_data.features, self.feature_stats)
            for key in VNET_COPY_DATA_KEYS:
                formatted_data[key] = pair_data.features[key]
            formatted_data['graph'] = pair_data.graph_edges
            formatted_data['score_path'] = pair_data.piece_path
            formatted_data['perform_path'] = pair_data.perform_path

            save_name = _flatten_path(
                Path(pair_data.perform_path).relative_to(Path(self.dataset_path))) + '.dat'

            with open(save_folder / pair_data.split_type / save_name, "wb") as f:
                pickle.dump(formatted_data, f, protocol=2)
  
        with open(save_folder / "stat.dat", "wb") as f:
            pickle.dump({'stats': self.feature_stats, 
                         'input_keys': VNET_INPUT_KEYS, 
                         'output_keys': VNET_OUTPUT_KEYS, 
                         'measure_keys': VNET_MEAS_KEYS}, f, protocol=2)
        

def get_feature_from_entire_dataset(dataset, target_score_features, target_perform_features):
    # e.g. feature_type = ['score', 'duration'] or ['perform', 'beat_tempo']
    output_values = dict()
    for feat_type in (target_score_features + target_perform_features):
        output_values[feat_type] = []
    for piece in dataset.pieces:
        for performance in piece.performances:
            for feat_type in target_score_features:
                # output_values[feat_type] += piece.score_features[feat_type]
                output_values[feat_type].append(piece.score_features[feat_type])
            for feat_type in target_perform_features:
                output_values[feat_type].append(performance.perform_features[feat_type])
    return output_values


def normalize_feature(data_values, target_feat_keys):
    for feat in target_feat_keys:
        concatenated_data = [note for perf in data_values[feat] for note in perf]
        mean = sum(concatenated_data) / len(concatenated_data)
        var = sum(pow(x-mean,2) for x in concatenated_data) / len(concatenated_data)
        # data_values[feat] = [(x-mean) / (var ** 0.5) for x in data_values[feat]]
        for i, perf in enumerate(data_values[feat]):
            data_values[feat][i] = [(x-mean) / (var ** 0.5) for x in perf]
    return data_values


def normalize_pedal_value(pedal_value):
    return (pedal_value - 64)/128

# def combine_dict_to_array():

def cal_mean_stds_of_entire_dataset(dataset, target_features):
    '''
    :param dataset: DataSet class
    :param target_features: list of dictionary keys of features
    :return: dictionary of mean and stds
    '''
    output_values = dict()
    for feat_type in (target_features):
        output_values[feat_type] = []

    for piece in dataset.pieces:
        for performance in piece.performances:
            for feat_type in target_features:
                if feat_type in piece.score_features:
                    output_values[feat_type] += piece.score_features[feat_type]
                elif feat_type in performance.perform_features:
                    output_values[feat_type] += performance.perform_features[feat_type]
                else:
                    print('Selected feature {} is not in the data'.format(feat_type))

    stats = cal_mean_stds(output_values, target_features)

    return stats


def cal_mean_stds(feat_datas, target_features):
    stats = dict()
    for feat_type in target_features:
        mean = sum(feat_datas[feat_type]) / len(feat_datas[feat_type])
        var = sum((x-mean)**2 for x in feat_datas[feat_type]) / len(feat_datas[feat_type])
        stds = var ** 0.5
        if stds == 0:
            stds = 1
        stats[feat_type] = {'mean': mean, 'stds':stds}
    return stats


def convert_feature_to_VirtuosoNet_format(feature_data, stats, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS, meas_keys=VNET_MEAS_KEYS, beat_keys=VNET_BEAT_KEYS):
    if 'num_notes' not in feature_data.keys():
        feature_data['num_notes'] = len(feature_data[input_keys[0]])

    def check_if_global_and_normalize(key):
        value = feature_data[key]
        if not isinstance(value, list) or len(value) != feature_data['num_notes']:  # global features like qpm_primo, tempo_primo, composer_vec
            value = [value] * feature_data['num_notes']
        if key in stats:  # if key needs normalization,
            value = [(x - stats[key]['mean']) / stats[key]['stds'] for x in value]
        return value

    def add_to_list(alist, item):
        if isinstance(item, list):
            alist += item
        else:
            alist.append(item)
        return alist

    def cal_dimension(data_with_all_features):
        total_length = 0
        for feat_data in data_with_all_features:
            if isinstance(feat_data[0], list):
                length = len(feat_data[0])
            else:
                length = 1
            total_length += length
        return total_length

    def make_feat_to_array(keys):
        datas = [] 
        for key in keys:
            value = check_if_global_and_normalize(key)
            datas.append(value)
        dimension = cal_dimension(datas)
        array = np.zeros((feature_data['num_notes'], dimension))
        current_idx = 0
        for value in datas:
            if isinstance(value[0], list):
                length = len(value[0])
                array[:, current_idx:current_idx + length] = value
            else:
                length = 1
                array[:,current_idx] = value
            current_idx += length
        return array

    input_array = make_feat_to_array(input_keys)
    output_array = make_feat_to_array(output_keys)
    meas_array = make_feat_to_array(meas_keys)
    beat_array = make_feat_to_array(beat_keys)
    # for key in input_keys:
    #     value = check_if_global_and_normalize(key)
    #     input_data.append(value)
    # for key in output_keys:
    #     value = check_if_global_and_normalize(key)
    #     output_data.append(value)
    # for key in meas_keys:
    #     value = check_if_global_and_normalize(key)
    #     meas_data.append(value)

    # input_dimension = cal_dimension(input_data)
    # output_dimension = cal_dimension(output_data)
    # meas_dimension = cal_dimension(meas_data)

    # input_array = np.zeros((feature_data['num_notes'], input_dimension))
    # output_array = np.zeros((feature_data['num_notes'], output_dimension))
    # meas_array = np.zeros((feature_data['num_notes'], meas_dimension))

    # current_idx = 0

    # for value in input_data:
    #     if isinstance(value[0], list):
    #         length = len(value[0])
    #         input_array[:, current_idx:current_idx + length] = value
    #     else:
    #         length = 1
    #         input_array[:,current_idx] = value
    #     current_idx += length
    # current_idx = 0
    # for value in output_data:
    #     if isinstance(value[0], list):
    #         length = len(value[0])
    #         output_array[:, current_idx:current_idx + length] = value
    #     else:
    #         length = 1
    #         output_array[:,current_idx] = value
    #     current_idx += length
    # current_idx = 0
    # for value in meas_data:
    #     if isinstance(value[0], list):
    #         length = len(value[0])
    #         meas_array[:, current_idx:current_idx + length] = value
    #     else:
    #         length = 1
    #         meas_array[:,current_idx] = value
    #     current_idx += length

    return input_array, output_array, meas_array, beat_array


