'''
split data classes to an individual file
'''
import os
import pickle
import pandas
import math
import ntpath
import shutil
import subprocess
from pathlib import Path
import warnings
import copy
from abc import abstractmethod
from tqdm import tqdm

from .musicxml_parser import MusicXMLDocument
from .midi_utils import midi_utils
from . import score_as_graph as score_graph, xml_midi_matching as matching
from . import xml_utils
from . import feature_extraction

align_dir = '/home/jdasam/AlignmentTool_v190813'
DEFAULT_SCORE_FEATURES = ['midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                          'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                          'slur_beam_vec',  'composer_vec', 'notation', 'tempo_primo', 'note_location']
DEFAULT_PERFORM_FEATURES = ['beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 'qpm_primo', 'align_matched', 'articulation_loss_weight',
                            'beat_dynamics', 'measure_tempo', 'measure_dynamics']


# total data class
class DataSet:
    def __init__(self, path):
        self.path = path

        self.pieces = []
        self.scores = []
        self.score_midis = []
        self.perform_midis = []
        self.composers = []

        self.performances = []
        
        self.performs_by_tag = {}
        self.flattened_performs_by_tag = []

        self.num_pieces = 0
        self.num_performances = 0
        self.num_score_notes = 0
        self.num_performance_notes = 0

        # self.default_perforddmances = self.performances
        # TGK: what for?

        # self._load_all_scores()

        scores, score_midis, perform_midis, composers = self.load_data()
        self.scores = scores
        self.score_midis = score_midis
        self.perform_midis = perform_midis
        self.composers = composers
        self.load_all_performances(scores, perform_midis, score_midis, composers)

    @classmethod
    @abstractmethod
    def load_data(self):
        '''return scores, score_midis, performances, composers'''
        raise NotImplementedError

    
    '''
    def _load_all_scores(self):
        musicxml_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.path) for f in filenames if
                     f.endswith('xml')]
        # path = Path(self.path)
        # musicxml_list = sorted(path.glob('**.xml'))
        for xml in musicxml_list:
            print('piece path:', xml)
            try:
                piece = PieceData(xml, data_structure=self.data_structure, dataset_name=self.dataset_name)
                self.pieces.append(piece)
            except Exception as ex:
                print('Error type :', ex)
        self.num_pieces = len(self.pieces)
    '''

    def load_all_performances(self, scores, perform_midis, score_midis, composers):
        for n in tqdm(range(len(scores))):
            try:
                piece = PieceData(scores[n], perform_midis[n], score_midis[n], composers[n])
                self.pieces.append(piece)
                for perf in piece.performances:
                    self.performances.append(perf)
            except Exception as ex:
                # TODO: TGK: this is ambiguous. Can we specify which file 
                # (score? performance? matching?) and in which function the error occur?
                print(f'Error while processing {scores[n]}. Error type :{ex}')
            
        self.num_performances = len(self.performances)

    '''
    # TGK : same as extract_selected_features(DEFAULT_SCORE_FEATURES) ...
    def extract_all_features(self):
        score_extractor = feature_extraction.ScoreExtractor(DEFAULT_SCORE_FEATURES)
        perform_extractor = feature_extraction.PerformExtractor(DEFAULT_PERFORM_FEATURES)
        for piece in self.pieces:
            piece.score_features = score_extractor.extract_score_features(piece)
            for perform in piece.performances:
                perform.perform_features = perform_extractor.extract_perform_features(piece, perform)
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

    def save_features_as_csv(self, features, feature_names, path='features.csv'):
        feature_type_name = ['MIDI Path'] + feature_names
        feature_with_name = [feature_type_name]
        perform_names = [x.midi_path for x in self.performances]
        for i,feature_by_perf in enumerate(features):
            feature_by_perf = [perform_names[i]] + feature_by_perf
            feature_with_name.append(feature_by_perf)
        dataframe = pandas.DataFrame(feature_with_name)
        dataframe.to_csv(path)

    def save_features_by_features_as_csv(self, feature_data, list_of_features, path='measure.csv'):
        for feature, feature_name in zip(feature_data, list_of_features):
            save_name = feature_name + '_' + path
            self.save_features_as_csv(feature, [feature_name], save_name)

    def features_to_list(self, list_of_feat):
        feature_data = [[] for i in range(len(list_of_feat))]
        for perf in self.performances:
            for i, feature_type in enumerate(list_of_feat):
                feature_data[i].append(perf.perform_features[feature_type])
        return feature_data

    def get_average_by_perform(self, feature_data):
        # axis 0: feature, axis 1: performance, axis 2: note
        average_data = [[] for i in range(len(feature_data[0]))]
        for feature in feature_data:
            for i, perf in enumerate(feature):
                valid_list = [x for x in perf if x is not None]
                avg = sum(valid_list) / len(valid_list)
                average_data[i].append(avg)

        return average_data

    # def list_features_to_measure(self, feature_data):
    #     # axis 0: performance, axis 1: feature, axis 2: note
    #     for data_by_perf in feature_data:

    def get_average_feature_by_measure(self, list_of_features):
        measure_average_features = [[] for i in range(len(list_of_features))]
        for p_index, perf in enumerate(self.performances):
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
                    feature_value = perf.perform_features[target_feature][i]
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

    def update_dataset(self):
        '''
        old_music_xml_list = [piece.meta.xml_path for piece in self.pieces]
        cur_musicxml_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.path) for f in filenames if
                         f.endswith('xml')]
        for xml in self.pieces:
            if xml not in old_music_xml_list:
                print('Updated piece:', xml)
                try:
                    piece = PieceData(xml, self.piece)
                    self.pieces.append(piece)
                except Exception as ex:
                    print('Error type :', ex)
        self.num_pieces = len(self.pieces)
        '''
        for piece in self.pieces:
            piece.update_performances()

# score data class
class PieceData:
    def __init__(self, xml_path, perform_lists, score_midi_path=None, composer=None):
        self.meta = PieceMeta(xml_path, perform_lists=perform_lists, score_midi_path=score_midi_path, composer=composer)
        self.xml_obj = None
        self.xml_notes = None
        self.num_notes = 0
        self.performances = []
        self.score_performance_match = []
        self.notes_graph = []

        self.score_match_list = []
        self.score_pairs = []

        self.score_features = {}

        self._load_score_xml(xml_path)
        self._load_score_midi(score_midi_path)
        self._match_score_xml_to_midi()

        self.meta._check_perf_align()

    def _load_score_xml(self, xml_path):
        self.xml_obj = MusicXMLDocument(xml_path)
        self._get_direction_encoded_notes()
        self.notes_graph = score_graph.make_edge(self.xml_notes)
        self.measure_positions = self.xml_obj.get_measure_positions()
        self.beat_positions = self.xml_obj.get_beat_positions()
        self.section_positions = xml_utils.find_tempo_change(self.xml_notes)

    def _load_score_midi(self, score_midi_path):
        '''
        if self.meta.data_structure == 'folder':
            midi_file_name = self.meta.folder_path + '/midi_cleaned.mid'
        else: # data_structure == 'name'
            midi_file_name = os.path.splitext(self.meta.xml_path)[0] + '.mid'
        ''' 
        if not os.path.isfile(score_midi_path):
            self.make_score_midi(score_midi_path)
        self.score_midi = midi_utils.to_midi_zero(score_midi_path)
        self.score_midi_notes = self.score_midi.instruments[0].notes
        self.score_midi_notes.sort(key=lambda x:x.start)
        
    def extract_perform_features(self, target_features):
        perform_extractor = feature_extraction.PerformExtractor(target_features)
        for perform in self.performances:
            print('Performance:', perform.midi_path)
            for feature_name in target_features:
                perform.perform_features[feature_name] = getattr(perform_extractor, 'get_'+ feature_name)(self, perform)

    def extract_score_features(self, target_features):
        score_extractor = feature_extraction.ScoreExtractor(target_features)
        self.score_features = score_extractor.extract_score_features(self)
    
    def make_score_midi(self, midi_file_name):
        midi_notes, midi_pedals = xml_utils.xml_notes_to_midi(self.xml_notes)
        xml_utils.save_midi_notes_as_piano_midi(midi_notes, [], midi_file_name, bool_pedal=True)
        
    def _get_direction_encoded_notes(self):
        notes, rests = self.xml_obj.get_notes()
        directions = self.xml_obj.get_directions()
        time_signatures = self.xml_obj.get_time_signatures()

        self.xml_notes = xml_utils.apply_directions_to_notes(notes, directions, time_signatures)
        self.num_notes = len(self.xml_notes)

    def _match_score_xml_to_midi(self):
        self.score_match_list = matching.match_xml_to_midi(self.xml_notes, self.score_midi_notes)
        self.score_pairs = matching.make_xml_midi_pair(self.xml_notes, self.score_midi_notes, self.score_match_list)

    def _load_performances(self):
        for perf_midi_name in self.meta.perform_lists:
            perform_data = PerformData(perf_midi_name, self.meta)
            self._align_perform_with_score(perform_data)
            self.performances.append(perform_data)

    def _align_perform_with_score(self, perform):
        perform.match_between_xml_perf = matching.match_score_pair2perform(self.score_pairs, perform.midi_notes, perform.corresp)
        perform.pairs = matching.make_xml_midi_pair(self.xml_notes, perform.midi_notes, perform.match_between_xml_perf)
        perform.pairs, perform.valid_position_pairs = matching.make_available_xml_midi_positions(perform.pairs)

        print('Performance path is ', perform.midi_path)
        perform._count_matched_notes()

    def update_performances(self):
        old_performances = copy.copy(self.meta.perform_lists)
        self.meta._check_perf_align()
        new_performances = self.meta.perform_lists
        for perf_path in new_performances:
            if perf_path not in old_performances:
                perform_data = PerformData(perf_path, self.meta)
                self._align_perform_with_score(perform_data)
                self.performances.append(perform_data)

    def __str__(self):
        text = 'Path name: {}, Composer Name: {}, Number of Performances: {}'.format(self.meta.xml_path, self.meta.composer, len(self.performances))
        return text


# score meta data class
class PieceMeta:
    def __init__(self, xml_path, perform_lists, score_midi_path, composer=None):
        self.xml_path = xml_path
        self.folder_path = os.path.dirname(xml_path)
        self.composer = composer
        self.pedal_elongate = False
        self.perform_lists = perform_lists
        self.score_midi_path = score_midi_path

    def _check_perf_align(self):
        # TODO Why these are here?
        aligned_perf = []
        for perf in self.perform_lists:
            align_file_name = os.path.splitext(perf)[0] + '_infer_corresp.txt'
            if os.path.isfile(align_file_name):
                aligned_perf.append(perf)
                continue
            self.align_score_and_perf_with_nakamura(os.path.abspath(perf), self.score_midi_path)
            if os.path.isfile(align_file_name): # check once again whether the alignment was successful
                aligned_perf.append(perf)

        self.perf_file_list = aligned_perf

    def align_score_and_perf_with_nakamura(self, midi_file_path, score_midi_path):
        file_folder, file_name = ntpath.split(midi_file_path)
        perform_midi = midi_file_path

        shutil.copy(perform_midi, os.path.join(align_dir, 'infer.mid'))
        shutil.copy(score_midi_path, os.path.join(align_dir, 'score.mid'))
        current_dir = os.getcwd()
        try:
            os.chdir(align_dir)
            subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
        except:
            print('Error to process {}'.format(midi_file_path))
            print('Trying to fix MIDI file {}'.format(midi_file_path))
            os.chdir(current_dir)
            shutil.copy(midi_file_path, midi_file_path+'old')
            midi_utils.to_midi_zero(midi_file_path, save_midi=True, save_name=midi_file_path)
            shutil.copy(midi_file_path, os.path.join(align_dir, 'infer.mid'))
            try:
                os.chdir(align_dir)
                subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
            except:
                align_success = False
                print('Fail to process {}'.format(midi_file_path))
                os.chdir(current_dir)
            else:
                align_success = True
        else:
            align_success = True

        if align_success:
            shutil.move('infer_corresp.txt', midi_file_path.replace('.mid', '_infer_corresp.txt'))
            shutil.move('infer_match.txt', midi_file_path.replace('.mid', '_infer_match.txt'))
            shutil.move('infer_spr.txt', midi_file_path.replace('.mid', '_infer_spr.txt'))
            shutil.move('score_spr.txt', os.path.join(align_dir, '_score_spr.txt'))
            os.chdir(current_dir)

    ''' 
    def _load_composer_name(self):
        if self.data_structure == 'folder':
            # self.folder_path = 'pyScoreParser/chopin_cleaned/{composer_name}/...'
            path_split = copy.copy(self.folder_path).split('/')
            if path_split[0] == self.dataset_name:
                composer_name = path_split[1]
            else:
                if self.dataset_name in path_split:
                    dataset_folder_name_index = path_split.index(self.dataset_name)
                    composer_name = path_split[dataset_folder_name_index+1]
                else:
                    composer_name = None
        else:
            # self.folder_path = '.../emotionDataset/{data_name.mid}'
            # consider data_name = '{composer_name}.{piece_name}.{performance_num}.mid'
            data_name = os.path.basename(self.xml_path)
            composer_name = data_name.split('.')[0]

        self.composer = composer_name
    '''


# performance data class
class PerformData:
    def __init__(self, midi_path, meta):
        self.midi_path = midi_path
        self.midi = midi_utils.to_midi_zero(self.midi_path)
        self.midi = midi_utils.add_pedal_inf_to_notes(self.midi)
        self.midi_notes = self.midi.instruments[0].notes
        self.corresp_path = os.path.splitext(self.midi_path)[0] + '_infer_corresp.txt'
        self.corresp = matching.read_corresp(self.corresp_path)
        self.perform_features = {}
        self.match_between_xml_perf = None
        
        self.pairs = []
        self.valid_position_pairs = []

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


class YamahaDataset(DataSet):
    def __init__(self, path):
        super().__init__(path)

    def load_data(self):
        path = Path(self.path)
        xml_list = sorted(path.glob('**/*.musicxml'))
        score_midis = [xml.parent / 'midi_cleaned.mid' for xml in xml_list]
        composers = [xml.relative_to(self.path).parts[0] for xml in xml_list]

        files_in_folder = [x for x in xml_list] 
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


class EmotionDataset(DataSet):
    def __init__(self, path):
        super().__init__(path)

    def load_data(self):
        path = Path(self.path)
        xml_list = sorted(path.glob('**/*.musicxml'))
        raise NotImplementedError
        score_midis = [xml.parent / 'midi_cleaned.mid' for xml in xml_list]
        composers = [xml.relative_to(self.path).parts[0] for xml in xml_list]

        files_in_folder = [x for x in xml_list] 
        perform_lists = []
        for xml in xml_list:
            midis = sorted(xml.parent.glob('*.mid'))
            midis = [str(midi) for midi in midis if midi.name not in ['midi.mid', 'midi_cleaned.mid']]
            midis = [midi for midi in midis if not 'XP' in midi]
            perform_lists.append(midis)

        # Path -> string wrapper
        xml_list = [str(xml) for xml in xml_list]
        score_midis = [str(midi) for midi in score_midis]
        return xml_list, score_midis, perform_lists, composers