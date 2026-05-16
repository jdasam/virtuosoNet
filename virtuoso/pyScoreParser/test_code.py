#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import data_for_training as dft
from . import data_class
import _pickle as cPickle
from pathlib import Path


# test_piece = data_class.PieceData('EmotionData/Liszt.dante-sonata_s161_no7_.mm_35-54.musicxml', 'name')
# print(test_piece)

'''

data_set = data_class.DataSet('EmotionData/', 'name')
data_set.load_all_performances()
data_set.save_dataset('EmotionData.dat')

with open('EmotionData.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()

# for piece in data_set.pieces:
#     piece.score_features = {}

data_set.update_dataset()
data_set.save_dataset('EmotionData.dat')
#
# data_set._sort_performances()
# target_features = ['beat_tempo', 'tempo_fluctuation', 'velocity', 'articulation', 'beat_dynamics', 'measure_dynamics']
# data_set.extract_selected_features(target_features)
# features = data_set.features_to_list(target_features)
# features = data_set.get_average_by_perform(features)
# data_set.save_features_as_csv(features, target_features, path='features_by_piecewise_average.csv')
#
#
# measure_level_features = data_set.get_average_feature_by_measure(target_features)
# data_set.save_features_by_features_as_csv(measure_level_features, target_features)

'''


'''

data_set = data_class.DataSet('MIDI_XML_sarah/', 'folder')
data_set.load_all_performances()
data_set.save_dataset('SarahData.dat')

with open('SarahData.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()

target_features = ['velocity', 'articulation']
data_set._divide_by_tag(['amateur', 'professional'])
data_set._sort_performances()
data_set.extract_selected_features(target_features)
data_set.default_performances = data_set.flattened_performs_by_tag
features = data_set.features_to_list(target_features)
features = data_set.get_average_by_perform(features)
data_set.save_features_as_csv(features, target_features, path='features_by_piecewise_average.csv')
features = data_set.features_to_list(target_features)
data_set.save_features_by_features_as_csv(features, target_features, path='note.csv')

'''


data_set = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', 'folder')
data_set.load_all_performances()
# score_extractor = feat_ext.ScoreExtractor(['composer_vec'])
for piece in data_set.pieces:
    piece.meta.composer = 'Haydn'
#     piece.score_features['composer_vec'] = score_extractor.get_composer_vec(piece)
data_set.extract_all_features()
data_set.save_dataset('HaydnTest.dat')

with open('HaydnTest.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()

# perform_extractor = feat_ext.PerformExtractor(['beat_dynamics', 'measure_dynamics'])
# for piece in data_set.pieces:
#     for performance in piece.performances:
#         performance.perform_features = perform_extractor.extract_perform_features(piece, performance)
# data_set.save_dataset('HaydnTest.dat')


pair_data = dft.PairDataset(data_set)
pair_data.update_dataset_split_type()
pair_data.update_mean_stds_of_entire_dataset()
pair_data.save_features_for_virtuosoNet('HaydnTestFeature')

