import style_analysis
import pyScoreParser.xml_matching as xml_matching
import pyScoreParser.midi_utils.midi_utils as midi_utils
import pyScoreParser.data_class as data_class
import pickle
import _pickle as cPickle
import numpy as np


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

# data_set = data_class.DataSet('MIDI_XML_sarah/')
# data_set.load_all_performances()
# data_set.extract_all_features()
# data_set.save_dataset()

with open('data_set.dat', "rb") as f:
    u = cPickle.Unpickler(f)
    data_set = u.load()


target_features = ['articulation', 'attack_deviation', 'abs_deviation', 'tempo_fluctuation', 'left_hand_velocity', 'right_hand_velocity',
                   'velocity', 'right_hand_attack_deviation', 'left_hand_attack_deviation', 'pedal_at_start', 'pedal_at_end', 'pedal_refresh', 'pedal_cut']
data_set._divide_by_tag(['amateur', 'professional'])
data_set._sort_performances()
data_set.default_performances = data_set.flattened_performs_by_tag
data_set.extract_selected_features(target_features)
features = data_set.features_to_list(target_features)
features = data_set.get_average_by_perform(features)
data_set.save_features_as_csv(features, target_features, path='features_by_piecewise_average.csv')


measure_level_features = data_set.get_average_feature_by_measure(target_features)
data_set.save_features_by_features_as_csv(measure_level_features, target_features)

features = data_set.features_to_list(target_features)
data_set.save_features_by_features_as_csv(features, target_features, path='note.csv')
