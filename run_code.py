import style_analysis
import pyScoreParser.xml_matching as xml_matching
import pyScoreParser.midi_utils.midi_utils as midi_utils

# style_analysis.load_tsne_and_plot("style_tsne_z.dat")
# style_analysis.load_z_filter_and_plot("style_z_encoded.dat")

# style_analysis.save_style_data('test_pieces/emotionNet')
style_analysis.save_style_data('test_pieces/performScore_test/')

# style_analysis.save_emotion_perf_data('pyScoreParser/EmotionData')

# score_midi_name = 'pyScoreParser/EmotionData/Beethoven.sonata_op13_no8_mov3.mm_1-43.s007.E1.mid'
# output_name = 'convert.mid'
# score_midi = midi_utils.to_midi_zero(score_midi_name)
# midi_notes = score_midi.instruments[0].notes
# midi_pedals = score_midi.instruments[0].control_changes
#
# xml_matching.save_midi_notes_as_piano_midi(midi_notes, midi_pedals, output_name, bool_pedal=False, disklavier=False)
