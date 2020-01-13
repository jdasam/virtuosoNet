from . import xml_direction_encoding as dir_enc
from . import xml_utils, utils

class NoteExtractorState:
    def __init__(self):
        self.cur_note = None
        self.cur_position = None
        self.cur_idx = None

    def _update_position(self, note, idx):
        self.cur_note = note
        self.cur_position = self.cur_note.note_duration.xml_position
        self.cur_idx = idx

class ScoreExtractor:
    def __init__(self, feature_keys):
        self.state = NoteExtractorState()
        self.selected_feature_keys = feature_keys

    def _update_global_feature(self, piece_data):
        self.beat_positions = piece_data.beat_positions
        self.measure_positions = piece_data.measure_positions4#

    def extract_and_update_score_features(self, piece_data):
        features = []
        DYN_EMB_TAB = dir_enc.define_dyanmic_embedding_table()
        TEM_EMB_TAB = dir_enc.define_tempo_embedding_table()

        piece_data.qpm_primo = piece_data.xml_notes[0].state_fixed.qpm
        tempo_primo_word = dir_enc.direction_words_flatten(piece_data.xml_notes[0].tempo)
        if tempo_primo_word:
            piece_data.tempo_primo = dir_enc.dynamic_embedding(tempo_primo_word, TEM_EMB_TAB, 5)
            piece_data.tempo_primo = piece_data.tempo_primo[0:2]
        else:
            piece_data.tempo_primo = [0, 0]

        total_length = xml_utils.cal_total_xml_length(piece_data.xml_notes)

        for i, note in enumerate(piece_data.xml_notes):
            feature = {}
            self.state._update_position(note, i)
            feature['note_location'] = self.get_note_location(piece_data)
            measure_index = note.measure_number - 1
            note_position = note.note_duration.xml_position

            if measure_index + 1 < len(piece_data.measure_positions):
                measure_length = piece_data.measure_positions[measure_index + 1] - piece_data.measure_positions[measure_index]
                # measure_sec_length = measure_seocnds[measure_index+1] - measure_seocnds[measure_index]
            else:
                measure_length = piece_data.measure_positions[measure_index] - piece_data.measure_positions[measure_index - 1]
                # measure_sec_length = measure_seocnds[measure_index] - measure_seocnds[measure_index-1]
            feature.midi_pitch = note.pitch[1]
            feature.pitch = pitch_into_vector(note.pitch[1])
            feature.duration = note.note_duration.duration / note.state_fixed.divisions

            beat_position = (note_position - piece_data.measure_positions[measure_index]) / measure_length
            feature.beat_importance = cal_beat_importance(beat_position, note.tempo.time_numerator)
            feature.measure_length = measure_length / note.state_fixed.divisions
            feature.xml_position = note.note_duration.xml_position / total_length

            feature._update_score_features_from_note(note)
            feature._update_tempo_dynamics(note)
            self.crescendo_to_continuous_value(note, feature)

            for key in self.selected_feature_keys:
                feature[key] = getattr(self, 'get_' + key)()

            features.append(feature)

        ###
        features = make_index_continuous(features, score=False)

        return features

    def crescendo_to_continuous_value(self, note, feature):
        cresc_words = ['cresc', 'decresc', 'dim']
        if feature.dynamic[1] != 0:
            for rel in note.dynamic.relative:
                for word in cresc_words:
                    if word in rel.type['type'] or word in rel.type['content']:
                        rel_length = rel.end_xml_position - rel.xml_position
                        if rel_length == float("inf") or rel_length == 0:
                            rel_length = note.state_fixed.divisions * 10
                        ratio = (note.note_duration.xml_position - rel.xml_position) / rel_length
                        feature.dynamic[1] *= (ratio + 0.05)
                        break

    def get_note_location(self, piece_data):
        note = self.state.cur_note
        measure_index = note.measure_number - 1

        return NoteLocation(NoteLocation(beat=utils.binary_index(piece_data.beat_positions, note.note_duration.xml_position),
                                                    measure = measure_index,
                                                    voice = note.voice,
                                                    section = utils.binary_index(piece_data.section_positions, note.note_duration.xml_position)))

    def get_midi_pitch(self):
        return self.state.cur_note.pitch[1]

    def get_pitch_vector(self):
        pitch = self.state.cur_note.pitch[1]
        pitch_vec = [0] * 13  # octave + pitch class
        octave = (pitch // 12) - 1
        octave = (octave - 4) / 4  # normalization
        pitch_class = pitch % 12

        pitch_vec[0] = octave
        pitch_vec[pitch_class + 1] = 1

        return pitch_vec

    def get_duration(self):
        return self.state.cur_note.note_duration.duration / state.cur_note.state_fixed.divisions

    def get_beat_importance(self):
        beat_position = (note_position - piece_data.measure_positions[measure_index]) / measure_length
        numerator = self.state.cur_note.tempo.time_numerator
        if beat_position == 0:
            beat_importance = 4
        elif beat_position == 0.5 and numerator in [2, 4, 6, 12]:
            beat_importance = 3
        elif abs(beat_position - (1 / 3)) < 0.001 and numerator in [3, 9]:
            beat_importance = 2
        elif (beat_position * 4) % 1 == 0 and numerator in [2, 4]:
            beat_importance = 1
        elif (beat_position * 5) % 1 == 0 and numerator in [5]:
            beat_importance = 2
        elif (beat_position * 6) % 1 == 0 and numerator in [3, 6, 12]:
            beat_importance = 1
        elif (beat_position * 8) % 1 == 0 and numerator in [2, 4]:
            beat_importance = 0.5
        elif (beat_position * 9) % 1 == 0 and numerator in [9]:
            beat_importance = 1
        elif (beat_position * 12) % 1 == 0 and numerator in [3, 6, 12]:
            beat_importance = 0.5
        elif numerator == 7:
            if abs((beat_position * 7) - 2) < 0.001:
                beat_importance = 2
            elif abs((beat_position * 5) - 2) < 0.001:
                beat_importance = 2
            else:
                beat_importance = 0
        else:
            beat_importance = 0
        return beat_importance
