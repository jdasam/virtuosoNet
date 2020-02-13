from .data_class import PieceData
from pathlib import Path
from .feature_extraction import ScoreExtractor, PerformExtractor
DEFAULT_SCORE_FEATURES = ['midi_pitch', 'duration', 'beat_importance', 'measure_length', 'qpm_primo',
                          'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                          'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                          'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                          'slur_beam_vec',  'composer_vec', 'notation', 'tempo_primo']
DEFAULT_PERFORM_FEATURES = ['beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut', 'qpm_primo', 'align_matched', 'articulation_loss_weight']

target = 'refactory/test_examples/Beethoven/32-1'
performance = 'refactory/test_examples/Beethoven/32-1/DupreeF03.mid'
xml_name = 'musicxml_cleaned.musicxml'

piece = PieceData(target + '/' + xml_name, [performance], target + '/' + 'midi.mid', composer='Beethoven')
first_note = dict(duration= 60, midi_ticks= 27.5, seconds= 0.20833333333333334, pitch= 'Eb4', MIDI_pitch= 63, voice= 5, velocity= 64)
assert (piece.score.xml_notes[0].note_duration.duration == 60 and
        piece.score.xml_notes[0].note_duration.midi_ticks == 27.5 and
        piece.score.xml_notes[0].pitch[0] == 'Eb4' and
        piece.score.xml_notes[0].pitch[1] == 63 and
        piece.score.xml_notes[0].voice == 5 and
        piece.score.xml_notes[0].velocity == 64), \
            f"first note not matched. \nans:{first_note}, \ngot:{piece.score.xml_notes[0]}"

piece._load_performances()
perform_ext = PerformExtractor(DEFAULT_PERFORM_FEATURES)
perform_f = perform_ext.extract_perform_features(piece, piece.performances[0])
for key in perform_f:
    try:
        print(f'{key}: len={len(perform_f[key])}, \n ex:{perform_f[key][:5]}')
    except:
        print(key, perform_f[key])

score_ext = ScoreExtractor(DEFAULT_SCORE_FEATURES)
score_f = score_ext.extract_score_features(piece)
for key in score_f:
    try:
        print(key, len(score_f[key]), score_f[key][:5])
    except:
        print(key, score_f[key])

print('Test success')
