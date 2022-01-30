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
                  "section_tempo",
                  'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                  'beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                  'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                  'pedal_refresh', 'pedal_cut', 
                  'beat_tempo', 'beat_dynamics', 'measure_tempo', 'measure_dynamics')

PRESERVE_FEAT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 'following_rest')


VNET_COPY_DATA_KEYS = ('note_location', 'align_matched', 'articulation_loss_weight')
VNET_INPUT_KEYS =  ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 
                    'qpm_primo',
                    "section_tempo",
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
  def __init__(self, piece, perform, exclude_long_graces=False):
    self.piece_path = piece.meta.xml_path
    self.perform_path = perform.midi_path
    self.graph_edges = piece.notes_graph
    self.features = {**piece.score_features, **perform.perform_features}
    self.split_type = None
    self.features['num_notes'] = piece.num_notes
    if exclude_long_graces:
      self._exclude_long_graces()

  def _exclude_long_graces(self, max_grace_order=5):
    for i, order in enumerate(self.features['grace_order']):
      if order < -max_grace_order:
        self.features['align_matched'][i] = 0
        self.features['onset_deviation'][i] = 0.0
    for i, dev in enumerate(self.features['onset_deviation']):
      if abs(dev) > 4:
        self.features['align_matched'][i] = 0
        self.features['onset_deviation'][i] = 0.0


class PairDataset:
  def __init__(self, dataset, exclude_long_graces=False):
    self.dataset_path = dataset.path
    self.data_pairs = []
    self.feature_stats = None
    for piece in dataset.pieces:
      for performance in piece.performances:
        if performance is None:
          continue
        if 'align_matched' not in performance.perform_features:
          continue
        len_notes =  len(performance.perform_features['align_matched'])
        num_aligned_notes = sum(performance.perform_features['align_matched'])
        if len_notes - num_aligned_notes > 800:
          continue
        if len_notes > num_aligned_notes * 1.5:
          continue
        self.data_pairs.append(ScorePerformPairData(piece, performance, exclude_long_graces))
        # if performance is not None \
        #         and 'align_matched' in performance.perform_features\
        #         and len(performance.perform_features['align_matched']) - sum(performance.perform_features['align_matched']) < 800:
        #     self.data_pairs.append(ScorePerformPairData(piece, performance, exclude_long_graces))
  
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

  def save_features_for_virtuosoNet(self, 
                                    save_folder, 
                                    update_stats=True, 
                                    valid_set_list=dataset_split.VALID_LIST, 
                                    test_set_list=dataset_split.TEST_LIST, 
                                    input_key_list=VNET_INPUT_KEYS,
                                    output_key_list=VNET_OUTPUT_KEYS):
      '''
      Convert features into format of VirtuosoNet training data
      :return: None (save file)
      '''
      def _flatten_path(file_path):
          return '_'.join(file_path.parts)

      save_folder = Path(save_folder)
      split_types = ['train', 'valid', 'test']

      save_folder.mkdir(parents=True, exist_ok=True)
      for split in split_types:
        (save_folder / split).mkdir(exist_ok=True)
  
      if update_stats:
          self.update_mean_stds_of_entire_dataset()
      self.update_dataset_split_type(valid_set_list=valid_set_list, test_set_list=test_set_list)
      
      feature_converter = FeatureConverter(self.feature_stats, self.data_pairs[0].features, input_key_list, output_key_list)

      for pair_data in tqdm(self.data_pairs):
          # formatted_data = dict()
          formatted_data = feature_converter(pair_data.features)
          # formatted_data['input_data'], formatted_data['output_data'], formatted_data['meas_level_data'], formatted_data['beat_level_data'] = \
          #       convert_feature_to_VirtuosoNet_format(pair_data.features, self.feature_stats, input_keys=input_key_list, output_keys=output_key_list)
          for key in VNET_COPY_DATA_KEYS:
              formatted_data[key] = pair_data.features[key]
          formatted_data['graph'] = pair_data.graph_edges
          formatted_data['score_path'] = pair_data.piece_path
          formatted_data['perform_path'] = pair_data.perform_path

          save_name = _flatten_path(
              Path(pair_data.perform_path).relative_to(Path(self.dataset_path))) + '.pkl'

          with open(save_folder / pair_data.split_type / save_name, "wb") as f:
              pickle.dump(formatted_data, f, protocol=2)

      with open(save_folder / "stat.pkl", "wb") as f:
          pickle.dump({'stats': self.feature_stats, 
                        'input_keys': input_key_list, 
                        'output_keys': output_key_list, 
                        'measure_keys': VNET_MEAS_KEYS,
                        'key_to_dim': feature_converter.key_to_dim_idx
                        }, f, protocol=2)


# def combine_dict_to_array():




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


class FeatureConverter:
  def __init__(self, stats, sample_feature_data=None, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS, beat_keys=VNET_BEAT_KEYS, meas_keys=VNET_MEAS_KEYS):
    self.stats = stats
    self.keys = {'input': input_keys, 'output': output_keys, 'beat': beat_keys, 'meas': meas_keys}

    self.preserve_keys = PRESERVE_FEAT_KEYS

    if sample_feature_data is not None:
      self._init_with_sample_data(sample_feature_data)
  
  def _init_with_sample_data(self, sample_feature_data):
    if 'num_notes' not in sample_feature_data.keys():
      sample_feature_data['num_notes'] = len(sample_feature_data[self.keys['input'][0]])
    self._preserve_feature_before_normalization(sample_feature_data)
    self.dim = {}
    self.key_to_dim_idx = {}
    if sample_feature_data is not None:
      for key_type in self.keys:
        selected_type_features = []
        for key in self.keys[key_type]:
          value = self._check_if_global_and_normalize(sample_feature_data, key)
          selected_type_features.append(value)
        dimension, key_to_dim_idx = self._cal_dimension(selected_type_features, self.keys[key_type])
        self.dim[key_type] = dimension
        self.key_to_dim_idx[key_type] = key_to_dim_idx

  def _check_if_global_and_normalize(self, feature_data, key):
    value = feature_data[key]
    if not isinstance(value, list) or len(value) != feature_data['num_notes']:  # global features like qpm_primo, tempo_primo, composer_vec
        value = [value] * feature_data['num_notes']
    if key in self.stats:  # if key needs normalization,
        value = [(x - self.stats[key]['mean']) / self.stats[key]['stds'] for x in value]
    return value

  def _cal_dimension(self, data_with_all_features, keys):
    assert len(data_with_all_features) == len(keys)
    total_length = 0
    key_to_dim_idx = {}
    for feat_data, key in zip(data_with_all_features, keys):
      if isinstance(feat_data[0], list):
        length = len(feat_data[0])
      else:
        length = 1
      key_to_dim_idx[key] = (total_length, total_length+length)
      total_length += length
    return total_length, key_to_dim_idx
  
  def _preserve_feature_before_normalization(self, feature_data):
    for key in self.preserve_keys:
      if key in feature_data:
        new_key_name = key + '_unnorm'
        if new_key_name not in feature_data:
          feature_data[new_key_name] = feature_data[key][:]
        if new_key_name not in self.keys['input']:
          self.keys['input'] = tuple(list(self.keys['input']) + [new_key_name])

  def make_feat_to_array(self, feature_data, key_type):
    if key_type == 'input':
      self._preserve_feature_before_normalization(feature_data)
    datas = [self._check_if_global_and_normalize(feature_data, key) for key in self.keys[key_type]]
    if hasattr(self, 'dim'):
      dimension = self.dim[key_type]
    else:
      dimension = self._cal_dimension(datas, self.keys[key_type])
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

  def __call__(self, feature_data):
    '''
    feature_data (dict): score or perform features in dict
    '''
    if 'num_notes' not in feature_data.keys():
      feature_data['num_notes'] = len(feature_data[self.keys['input'][0]])
    self._preserve_feature_before_normalization(feature_data)
    output = {}
    for key_type in self.keys:
      output[key_type] = self.make_feat_to_array(feature_data, key_type)
    return output

    

'''
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

    def cal_dimension(data_with_all_features, keys):
        assert len(data_with_all_features) == len(keys)
        total_length = 0
        key_to_dim_idx = {}
        for feat_data, key in zip(data_with_all_features, keys):
            if isinstance(feat_data[0], list):
                length = len(feat_data[0])
            else:
                length = 1
            key_to_dim_idx[key] = (total_length, total_length+length)
            total_length += length
        return total_length, key_to_dim_idx

    def make_feat_to_array(keys):
        datas = [] 
        for key in keys:
            value = check_if_global_and_normalize(key)
            datas.append(value)
        dimension, key_to_dim_idx = cal_dimension(datas, keys)
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
    return input_array, output_array, meas_array, beat_array


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



def cal_mean_stds_of_entire_dataset(dataset, target_features):
    # :param dataset: DataSet class
    # :param target_features: list of dictionary keys of features
    # :return: dictionary of mean and stds
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

'''