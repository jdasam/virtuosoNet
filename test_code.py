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
from pathlib import Path

data_set = data_class.DataSet('EmotionData/', 'name')
data_set.load_all_performances()
data_set.save_dataset('EmotionData.dat')
#
# test_piece = data_class.PieceData('EmotionData/Chopin.nocturne_op9_no2_.mm_1-12.musicxml', 'name')
# print(test_piece.xml_notes[202].note_duration.duration)

#
#
with open('EmotionData.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()

data_set.extract_all_features()
#
# target_features = ['articulation']
# data_set.extract_selected_features(target_features)
# features = data_set.features_to_list(target_features)
# features = data_set.get_average_by_perform(features)
# data_set.save_features_as_csv(features, target_features, path='features_by_piecewise_average.csv')
#
#
# measure_level_features = data_set.get_average_feature_by_measure(target_features)
# data_set.save_features_by_features_as_csv(measure_level_features, target_features)
#
# features = data_set.features_to_list(target_features)
# data_set.save_features_by_features_as_csv(features, target_features, path='note.csv')
