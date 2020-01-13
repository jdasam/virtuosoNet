'''
split data classes to an individual file
'''
import os
import pickle
import pandas
import math
import ntpath
import shutil

from .musicxml_parser import MusicXMLDocument
from .midi_utils import midi_utils
from . import score_as_graph as score_graph, xml_midi_matching as matching

# total data class
class DataSet:
    def __init__(self, path):
        self.path = path
        self.pieces = []
        self.performances = []
        self.performs_by_tag = {}
        self.flattened_performs_by_tag = []

        self.num_pieces = 0
        self.num_performances = 0
        self.num_score_notes = 0
        self.num_performance_notes = 0

        self.default_performances = self.performances

        self._load_all_scores()

    def _load_all_scores(self):
        musicxml_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.path) for f in filenames if
                     f.endswith('xml')]
        for xml in musicxml_list:
            piece = PieceData(xml)
            self.pieces.append(piece)
        self.num_pieces = len(self.pieces)

    def _load_all_performances(self):
        for piece in self.pieces:
            piece._load_performances()
            for perf in piece.performances:
                self.performances.append(perf)

        self.num_performances = len(self.performances)

    # want to extract features by using feature_extractin.py
    '''
    def _extract_all_features(self):
        for piece in self.pieces:
            piece._extract_score_features()
            piece._extract_all_perform_features()

    def _extract_selected_features(self, target_features):
        for piece in self.pieces:
            for perform in piece.performances:
                for feature_name in target_features:
                    getattr(piece, '_get_'+ feature_name)(perform)
    '''
    def _sort_performances(self):
        self.performances.sort(key=lambda x:x.midi_path)
        for tag in self.performs_by_tag:
            self.performs_by_tag[tag].sort(key=lambda x:x.midi_path)

        flattened_performs_by_tag = []
        for tag in self.performs_by_tag:
            for perform in self.performs_by_tag[tag]:
                flattened_performs_by_tag.append(perform)
                # self.perform_name_by_tag.append(perform.midi_path)
        self.flattened_performs_by_tag = flattened_performs_by_tag

    def save_dataset(self, filename='data_set.dat'):
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=2)

    def save_features_as_csv(self, features, path='features.csv'):
        feature_with_name = []
        perform_names = [x.midi_path for x in self.default_performances]
        for i,feature_by_perf in enumerate(features):
            feature_by_perf = [perform_names[i]] + feature_by_perf
            feature_with_name.append(feature_by_perf)
        dataframe = pandas.DataFrame(feature_with_name)
        dataframe.to_csv(path)

    def save_features_by_features_as_csv(self, feature_data, list_of_features, path='measure.csv'):
        for feature, feature_name in zip(feature_data, list_of_features):
            save_name = feature_name + '_' + path
            self.save_features_as_csv(feature, save_name)

    def features_to_list(self, list_of_feat):
        feature_data = []
        for perf in self.default_performances:
            perf_features = [[] for i in range(len(list_of_feat))]
            for feature in perf.perform_features:
                for i, feat_key in enumerate(list_of_feat):
                    value = getattr(feature, feat_key)
                    if value is not None:
                        perf_features[i].append(value)
            feature_data.append(perf_features)
        return feature_data

    def get_average_by_perform(self, feature_data):
        # axis 0: performance, axis 1: feature, axis 2: note
        average_data = []
        for perf in feature_data:
            avg_feature_by_perf = [[] for i in range(len(perf))]
            for i, feature in enumerate(perf):
                avg = sum(feature) / len(feature)
                avg_feature_by_perf[i] = avg
            average_data.append(avg_feature_by_perf)
        return average_data

    # def list_features_to_measure(self, feature_data):
    #     # axis 0: performance, axis 1: feature, axis 2: note
    #     for data_by_perf in feature_data:

    def get_average_feature_by_measure(self, list_of_features):
        measure_average_features = [[] for i in range(len(list_of_features))]
        for p_index, perf in enumerate(self.default_performances):
            features_in_performance = [ [] for i in range(len(list_of_features))]
            features_in_previous_measure = [ [] for i in range(len(list_of_features))]
            previous_measure = 0
            for i, pair in enumerate(perf.pairs):
                if pair == []:
                    continue
                if pair['xml'].measure_number != previous_measure:
                    previous_measure = pair['xml'].measure_number
                    if features_in_previous_measure[0] != []:
                        for j, data_of_selected_features in enumerate(features_in_previous_measure):
                            if len(data_of_selected_features) > 0:
                                average_of_selected_feature = sum(data_of_selected_features) / len(data_of_selected_features)
                            else:
                                average_of_selected_feature = 0
                            features_in_performance[j].append(average_of_selected_feature)
                        features_in_previous_measure = [[] for i in range(len(list_of_features))]
                for j, target_feature in enumerate(list_of_features):
                    feature_value = getattr(perf.perform_features[i], target_feature)
                    if feature_value is not None:
                        features_in_previous_measure[j].append(feature_value)
            for j, data_of_selected_features in enumerate(features_in_previous_measure):
                if len(data_of_selected_features) > 0:
                    average_of_selected_feature = sum(data_of_selected_features) / len(data_of_selected_features)
                else:
                    average_of_selected_feature = 0
                features_in_performance[j].append(average_of_selected_feature)
            for j, perform_data_of_feature in enumerate(measure_average_features):
                perform_data_of_feature.append(features_in_performance[j])
        return measure_average_features

    def _divide_by_tag(self, list_of_tag):
        # example of list_of_tag = ['professional', 'amateur']
        for tag in list_of_tag:
            self.performs_by_tag[tag] = []
        for piece in self.pieces:
            for perform in piece.performances:
                for tag in list_of_tag:
                    if tag in perform.midi_path:
                        self.performs_by_tag[tag].append(perform)
                        break

