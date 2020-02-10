import os
import random
import numpy as np
import pickle
from . import dataset_split

NORM_FEAT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 'qpm_primo')

VNET_COPY_DATA_KEYS = ('note_location', 'align_matched', 'articulation_loss_weight')
VNET_INPUT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                          'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                          'slur_beam_vec',  'composer_vec', 'notation', 'tempo_primo')

VNET_OUTPUT_KEYS = ('beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 'qpm_primo', 'beat_tempo', 'beat_dynamics',
                            'measure_tempo', 'measure_dynamics')


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
        self.data_pairs = []
        self.feature_stats = None
        for piece in dataset.pieces:
            for performance in piece.performances:
                self.data_pairs.append(ScorePerformPairData(piece, performance))

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
        # TODO: the split
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

    def save_features_for_virtuosoNet(self, save_name='training_data'):
        '''
        Convert features into format of VirtuosoNet training data
        :return: None (save file)
        '''
        training_data = []
        validation_data = []
        test_data = []

        for pair_data in self.data_pairs:
            formatted_data = dict()
            formatted_data['input_data'], formatted_data['output_data'] = convert_feature_to_VirtuosoNet_format(pair_data.features, self.feature_stats)
            for key in VNET_COPY_DATA_KEYS:
                formatted_data[key] = pair_data.features[key]
            formatted_data['graph'] = pair_data.graph_edges

            if pair_data.split_type == 'train':
                training_data.append(formatted_data)
            elif pair_data.split_type == 'valid':
                validation_data.append(formatted_data)
            elif pair_data.split_type == 'test':
                test_data.append(formatted_data)

        with open(save_name + ".dat", "wb") as f:
            pickle.dump({'train': training_data, 'valid': validation_data}, f, protocol=2)
        with open(save_name + "_test.dat", "wb") as f:
            pickle.dump(test_data, f, protocol=2)
        with open(save_name + "_stat.dat", "wb") as f:
            pickle.dump(self.feature_stats, f, protocol=2)

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


def convert_feature_to_VirtuosoNet_format(feature_data, stats, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS):
    input_data = []
    output_data = []

    def check_if_global_and_normalize(key):
        value = feature_data[key]
        if not isinstance(value, list) or len(value) != feature_data['num_notes']:  # global features like qpm_primo, tempo_primo, composer_vec
            value = [value] * feature_data['num_notes']
        if key in stats:  # if key needs normalization,
            value = [(x - stats[key]['mean']) / stats[key]['stds'] for x in value]
        return value

    for key in input_keys:
        value = check_if_global_and_normalize(key)
        input_data.append(value)
    for key in output_keys:
        value = check_if_global_and_normalize(key)
        output_data.append(value)

    input_data = np.asarray(input_data).transpose()
    output_data = np.asarray(output_data).transpose()

    return input_data, output_data
