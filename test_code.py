#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from .musicxml_parser import MusicXMLDocument
# from .midi_utils import midi_utils as midi_utils
# from . import xml_matching
# from . import xml_midi_matching as matching
# from . import score_as_graph as score_graph
# import pickle
from . import data_class
import _pickle as cPickle

data_set = data_class.DataSet('EmotionData/', 'name')
data_set.load_all_performances()
data_set.extract_all_features()
data_set.save_dataset()

with open('data_set.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()


target_features = ['articulation', 'attack_deviation', 'abs_deviation', 'tempo_fluctuation', 'left_hand_velocity', 'right_hand_velocity',
                   'velocity', 'right_hand_attack_deviation', 'left_hand_attack_deviation', 'pedal_at_start', 'pedal_at_end', 'pedal_refresh', 'pedal_cut']
# data_set._divide_by_tag(['amateur', 'professional'])
data_set._sort_performances()
# data_set.default_performances = data_set.flattened_performs_by_tag
data_set.extract_selected_features(target_features)
features = data_set.features_to_list(target_features)
features = data_set.get_average_by_perform(features)
data_set.save_features_as_csv(features, target_features, path='features_by_piecewise_average.csv')


measure_level_features = data_set.get_average_feature_by_measure(target_features)
data_set.save_features_by_features_as_csv(measure_level_features, target_features)

features = data_set.features_to_list(target_features)
data_set.save_features_by_features_as_csv(features, target_features, path='note.csv')
