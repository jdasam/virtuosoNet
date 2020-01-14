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
from . import xml_utils

ALIGN_DIR = '/home/jdasam/AlignmentTool_v190813'

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

# score data class
class PieceData:
    def __init__(self, xml_path, data_structure='folder'):
        self.meta = PieceMeta(xml_path, data_structure)
        self.xml_obj = None
        self.xml_notes = None
        self.performances = []
        self.score_performance_match = []

        self.score_features = []

        self._load_score_xml()
        self._load_score_midi()
        self._match_score_xml_to_midi()

        self.meta._load_list_of_performances()
        self.meta._check_perf_align()

    def _load_score_xml(self):
        self.xml_obj = MusicXMLDocument(self.meta.xml_path)
        self._get_direction_encoded_notes()
        self.notes_graph = score_graph.make_edge(self.xml_notes)
        self.measure_positions = self.xml_obj.get_measure_positions()
        self.beat_positions = self.xml_obj.get_beat_positions()
        self.section_positions = xml_utils.find_tempo_change(self.xml_notes)

    def _load_score_midi(self):
        if self.meta.data_structure == 'folder':
            midi_file_name = self.meta.folder_path + '/midi_cleaned.mid'
        else: # data_structure == 'name'
            midi_file_name = os.path.splitext(self.meta.xml_path)[0] + '.mid'
        if not os.path.isfile(midi_file_name):
            self.make_score_midi(midi_file_name)
        self.score_midi = midi_utils.to_midi_zero(midi_file_name)
        self.score_midi_notes = self.score_midi.instruments[0].notes
        self.score_midi_notes.sort(key=lambda x:x.start)

    def make_score_midi(self, midi_file_name):
        midi_notes, midi_pedals = xml_utils.xml_notes_to_midi(self.xml_notes)
        xml_utils.save_midi_notes_as_piano_midi(midi_notes, [], midi_file_name, bool_pedal=True)

    def _get_direction_encoded_notes(self):
        notes = self.xml_obj.get_notes()
        directions = self.xml_obj.get_directions()
        time_signatures = self.xml_obj.get_time_signatures()

        self.xml_notes = xml_utils.apply_directions_to_notes(notes, directions, time_signatures)

    def _match_score_xml_to_midi(self):
        self.score_match_list = matching.match_xml_to_midi(self.xml_notes, self.score_midi_notes)
        self.score_pairs = matching.make_xml_midi_pair(self.xml_notes, self.score_midi_notes, self.score_match_list)

    def _load_performances(self):
        for perf_midi_name in self.meta.perf_file_list:
            perform_data = PerformData(perf_midi_name, self.meta)
            self._align_perform_with_score(perform_data)
            self.performances.append(perform_data)

    def _align_perform_with_score(self, perform):
        perform.match_between_xml_perf = matching.match_score_pair2perform(self.score_pairs, perform.midi_notes, perform.corresp)
        perform.pairs = matching.make_xml_midi_pair(self.xml_notes, perform.midi_notes, perform.match_between_xml_perf)
        print('Performance path is ', perform.midi_path)
        perform._count_matched_notes()

    def __str__(self):
        text = 'Path name: {}, Composer Name: {}, Number of Performances: {}'.format(self.meta.xml_path, self.meta.composer, len(self.performances))
        return text


# score meta data class
class PieceMeta:
    def __init__(self, xml_path, data_structure='folder'):
        self.xml_path = xml_path
        self.folder_path = os.path.dirname(xml_path)
        self.composer = None
        self.data_structure = data_structure
        self.pedal_elongate = False
        self.perf_file_list = []

    def _load_list_of_performances(self):
        files_in_folder = os.listdir(self.folder_path)
        perf_file_list = []
        if self.data_structure == 'folder':
            for file in files_in_folder:
                if file.endswith('.mid') and not file in ('midi.mid', 'midi_cleaned.mid'):
                    perf_file_list.append(self.folder_path + '/' + file)
        else:
            for file in files_in_folder:
                pass

        self.perf_file_list = perf_file_list

    def _check_perf_align(self):
        aligned_perf = []
        for perf in self.perf_file_list:
            align_file_name = os.path.splitext(perf)[0] + '_infer_corresp.txt'
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)
                continue
            self.align_score_and_perf_with_nakamura(os.path.abspath(perf))
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)

        self.perf_file_list = aligned_perf

    def align_score_and_perf_with_nakamura(self, midi_file_path):
        file_folder, file_name = ntpath.split(midi_file_path)
        perform_midi = midi_file_path
        score_midi = os.path.join(file_folder, 'midi_cleaned.mid')
        if not os.path.isfile(score_midi):
            score_midi = os.path.join(file_folder, 'midi.mid')
        print(perform_midi)
        print(score_midi)

        shutil.copy(perform_midi, os.path.join(ALIGN_DIR, 'infer.mid'))
        shutil.copy(score_midi, os.path.join(ALIGN_DIR, 'score.mid'))
        current_dir = os.getcwd()
        try:
            os.chdir(ALIGN_DIR)
            subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
        except:
            print('Error to process {}'.format(midi_file_path))
            pass
        else:
            shutil.move('infer_corresp.txt', midi_file_path.replace('.mid', '_infer_corresp.txt'))
            shutil.move('infer_match.txt', midi_file_path.replace('.mid', '_infer_match.txt'))
            shutil.move('infer_spr.txt', midi_file_path.replace('.mid', '_infer_spr.txt'))
            shutil.move('score_spr.txt', os.path.join(ALIGN_DIR, '_score_spr.txt'))
            os.chdir(current_dir)

# performance data class
class PerformData:
    def __init__(self, midi_path, meta):
        self.midi_path = midi_path
        self.midi = midi_utils.to_midi_zero(self.midi_path)
        self.midi = midi_utils.add_pedal_inf_to_notes(self.midi)
        self.midi_notes = self.midi.instruments[0].notes
        self.corresp_path = os.path.splitext(self.midi_path)[0] + '_infer_corresp.txt'
        self.corresp = matching.read_corresp(self.corresp_path)
        self.perform_features = []
        self.match_between_xml_perf = None
        
        self.pairs = []

        self.num_matched_notes = 0
        self.num_unmatched_notes = 0
        self.tempos = []

        self.meta = meta

    def _count_matched_notes(self):
        self.num_matched_notes = 0
        self.num_unmatched_notes = 0
        for pair in self.pairs:
            if pair == []:
                self.num_unmatched_notes += 1
            else:
                self.num_matched_notes += 1
        print(
            'Number of Matched Notes: ' + str(self.num_matched_notes) + ', unmatched notes: ' + str(self.num_unmatched_notes))
