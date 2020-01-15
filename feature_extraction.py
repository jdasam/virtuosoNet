from . import xml_direction_encoding as dir_enc
from . import xml_utils, utils, feature_utils
class NoteLocation:
    def __init__(self, beat, measure, voice, section):
        self.beat = beat
        self.measure = measure
        self.voice = voice
        self.section = section

class ScoreExtractor:
    def __init__(self, feature_keys):
        self.selected_feature_keys = feature_keys

    '''
    def _update_global_feature(self, piece_data):
        self.beat_positions = piece_data.beat_positions
        self.measure_positions = piece_data.measure_positions
    '''

    def extract_and_update_score_features(self, piece_data):
        def _get_beat_position(piece_data):
            beat_positions = []
            for _, note in enumerate(piece_data.xml_notes):
                measure_index = note.measure_number - 1
                note_position = note.note_duration.xml_position

                if measure_index + 1 < len(piece_data.measure_positions):
                    measure_length = piece_data.measure_positions[measure_index +
                                                                  1] - piece_data.measure_positions[measure_index]
                else:
                    measure_length = piece_data.measure_positions[measure_index] - \
                        piece_data.measure_positions[measure_index - 1]
                beat_position = (
                    note_position - piece_data.measure_positions[measure_index]) / measure_length
                beat_positions.append(beat_position)
            return beat_positions

        def _get_beat_importance(piece_data):
            beat_positions = _get_beat_position(piece_data)
            beat_importances = []
            for i, note in enumerate(piece_data.xml_notes):
                importance = feature_utils.cal_beat_importance(
                    beat_positions[i], note.tempo.time_numerator)
                beat_importances.append(importance)
            return beat_importances

        def _get_measure_length(piece_data):
            measure_lengthes = []
            for _, note in enumerate(piece_data.xml_notes):
                measure_index = note.measure_number - 1

                if measure_index + 1 < len(piece_data.measure_positions):
                    measure_length = piece_data.measure_positions[measure_index +
                                                                  1] - piece_data.measure_positions[measure_index]
                else:
                    measure_length = piece_data.measure_positions[measure_index] - \
                        piece_data.measure_positions[measure_index - 1]
                measure_lengthes.append(
                    measure_length / note.state_fixed.divisions)
            return measure_lengthes

        def _get_cresciuto(piece_data):
            # TODO: what is this?
            cresciutos = []
            for note in piece_data.xml_notes:
                if note.dynamic.cresciuto:
                    cresciuto = (note.dynamic.cresciuto.overlapped + 1) / 2
                    if note.dynamic.cresciuto.type == 'diminuendo':
                        cresciuto *= -1
                else:
                    cresciuto = 0
                cresciutos.append(cresciuto)
            return cresciutos

        DYN_EMB_TAB = dir_enc.define_dyanmic_embedding_table()
        TEM_EMB_TAB = dir_enc.define_tempo_embedding_table()

        '''
        # TODO: unused.. separate from here?
        '''
        piece_data.qpm_primo = piece_data.xml_notes[0].state_fixed.qpm
        tempo_primo_word = dir_enc.direction_words_flatten(
            piece_data.xml_notes[0].tempo)
        if tempo_primo_word:
            piece_data.tempo_primo = dir_enc.dynamic_embedding(
                tempo_primo_word, TEM_EMB_TAB, 5)
            piece_data.tempo_primo = piece_data.tempo_primo[0:2]
        else:
            piece_data.tempo_primo = [0, 0]

        total_length = xml_utils.cal_total_xml_length(piece_data.xml_notes)

        features = dict()
        features['note_location'] = self.get_note_location(piece_data)
        features['midi_pitch'] = [note.pitch[1]
                                  for note in piece_data.xml_notes]
        features['pitch'] = [feature_utils.pitch_into_vector(
            note.pitch[1]) for note in piece_data.xml_notes]
        features['duration'] = [note.note_duration.duration / note.state_fixed.divisions
                                for note in piece_data.xml_notes]
        features['beat_position'] = _get_beat_position(piece_data)
        features['beat_importance'] = _get_beat_importance(piece_data)
        features['measure_length'] = _get_measure_length(piece_data)
        features['xml_position'] = [note.note_duration.xml_position /
                                    total_length for note in piece_data.xml_notes]

        features['grace_order'] = [
            note.note_duration.grace_order for note in piece_data.xml_notes]
        features['is_grace_note'] = [
            int(note.note_duration.is_grace_note) for note in piece_data.xml_notes]
        features['preceded_by_grace_note'] = [
            int(note.note_duration.preceded_by_grace_note) for note in piece_data.xml_notes]
        features['time_sig_vec'] = [feature_utils.time_signature_to_vector(
            note.tempo.time_signature) for note in piece_data.xml_notes]
        features['following_rest'] = [note.following_rest_duration /
                                      note.state_fixed.divisions for note in piece_data.xml_notes]
        features['followed_by_fermata_rest'] = [
            int(note.followed_by_fermata_rest) for note in piece_data.xml_notes]
        features['notation'] = [feature_utils.note_notation_to_vector(
            note) for note in piece_data.xml_notes]

        # TODO: better to save it as dict
        features['slur_beam_vec'] = \
            [[int(note.note_notations.is_slur_start),
              int(note.note_notations.is_slur_continue),
              int(note.note_notations.is_slur_stop),
              int(note.note_notations.is_beam_start),
              int(note.note_notations.is_beam_continue),
              int(note.note_notations.is_beam_stop)]
             for note in piece_data.xml_notes]

        features['dynamic'] = [
            dir_enc.dynamic_embedding(
                dir_enc.direction_words_flatten(note.dynamic), DYN_EMB_TAB, len_vec=4)
            for note in piece_data.xml_notes]

        features['tempo'] = [
            dir_enc.dynamic_embedding(
                dir_enc.direction_words_flatten(note.tempo), TEM_EMB_TAB, len_vec=5)
            for note in piece_data.xml_notes]

        features['cresciuto'] = _get_cresciuto(piece_data)
        # TODO: maybe its redundant?
        # self.dynamic.append(self.cresciuto)

        features['distance_from_abs_dynamic'] = \
            [(note.note_duration.xml_position - note.dynamic.absolute_position)
             / note.state_fixed.divisions for note in piece_data.xml_notes]

        features['distance_from_recent_tempo'] = \
            [(note.note_duration.xml_position - note.tempo.recently_changed_position)
             / note.state_fixed.divisions for note in piece_data.xml_notes]

        '''
        # TODO: ok...
        for key in self.selected_feature_keys:
            feature[key] = getattr(self, 'get_' + key)()

            features.append(feature)

        '''
        features = feature_utils.make_index_continuous(features, score=False)

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
        # TODO: need check up
        locations = []
        for _, note in enumerate(piece_data.xml_notes):
            measure_index = note.measure_number - 1
            locations.append(
                NoteLocation(beat=utils.binary_index(piece_data.beat_positions, note.note_duration.xml_position),
                             measure=measure_index,
                             voice=note.voice,
                             section=utils.binary_index(piece_data.section_positions, note.note_duration.xml_position)))
        return locations