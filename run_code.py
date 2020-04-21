import style_analysis
import pyScoreParser.xml_matching as xml_matching
import pyScoreParser.midi_utils.midi_utils as midi_utils
import pyScoreParser.data_class as data_class
import pickle
import _pickle as cPickle
import numpy as np
import pyScoreParser.data_for_training as dft
import pyScoreParser.feature_extraction as feat_ext


# style_analysis.load_tsne_and_plot("style_tsne_z.dat")
# style_analysis.load_z_filter_and_plot("style_z_encoded.dat")

# style_analysis.save_style_data('test_pieces/emotionNet')
# style_analysis.save_style_data('test_pieces/performScore_test/')

# style_analysis.save_emotion_perf_data('pyScoreParser/EmotionData')

# score_midi_name = 'pyScoreParser/EmotionData/Beethoven.sonata_op13_no8_mov3.mm_1-43.s007.E1.mid'
# output_name = 'convert.mid'
# score_midi = midi_utils.to_midi_zero(score_midi_name)
# midi_notes = score_midi.instruments[0].notes
# midi_pedals = score_midi.instruments[0].control_changes
#
# xml_matching.save_midi_notes_as_piano_midi(midi_notes, midi_pedals, output_name, bool_pedal=False, disklavier=False)

# data_set = data_class.DataSet('EmotionData/', 'name')
# data_set.load_all_performances()
# data_set.save_dataset('EmotionData.dat')


# test_piece = data_class.PieceData('pyScoreParser/chopin_cleaned/Schumann/Kreisleriana/2/musicxml_cleaned.musicxml', 'name')
# print(test_piece)



# with open('SarahData.dat', "rb") as f:
#     u = cPickle.Unpickler(f)
#     data_set = u.load()
#
# # data_set.extract_all_features()
# # data_set.save_dataset('SarahData.dat')
# # for perf in data_set.performances:
# #     perf.perform_features = {}
#
# # target_features = ['articulation']
# # data_set._divide_by_tag(['amateur', 'professional'])
# # data_set._sort_performances()
# # data_set.default_performances = data_set.flattened_performs_by_tag
# # data_set.extract_selected_features(target_features)
#
# values = dft.get_feature_from_entire_dataset(data_set, ['midi_pitch', 'pitch', 'duration'], ['beat_tempo', 'velocity'])
# values = dft.normalize_feature(values, ['midi_pitch', 'duration', 'velocity'])
#
# pair_data = dft.PairDataset(data_set)
# pair_data.update_dataset_split_type()
#
# print(values)



data_set = data_class.DataSet('pyScoreParser/chopin_cleaned/', 'folder')
data_set.save_dataset('vnet_data.dat')
data_set.load_all_performances()
# score_extractor = feat_ext.ScoreExtractor(['composer_vec'])
# for piece in data_set.pieces:
#     piece.meta.composer = 'Haydn'
#     piece.score_features['composer_vec'] = score_extractor.get_composer_vec(piece)
data_set.extract_all_features()
data_set.save_dataset('vnet_data.dat')

with open('vnet_data.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()

# perform_extractor = feat_ext.PerformExtractor(['beat_dynamics', 'measure_dynamics'])
# for piece in data_set.pieces:
#     for performance in piece.performances:
#         performance.perform_features = perform_extractor.extract_perform_features(piece, performance)
# data_set.save_dataset('HaydnTest.dat')


#
#

pair_data = dft.PairDataset(data_set)
pair_data.update_dataset_split_type()
pair_data.update_mean_stds_of_entire_dataset()
pair_data.save_features_for_virtuosoNet('vnet_data_feature')

