from . import xml_direction_encoding as dir_enc
from . import xml_utils, utils


class ScoreExtractor:
    def __init__(self, feature_keys):
        self.selected_feature_keys = feature_keys

    def _update_global_feature(self, piece_data):
        self.beat_positions = piece_data.beat_positions
        self.measure_positions = piece_data.measure_positions

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
                importance = cal_beat_importance(
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
            # This function converts cresciuto class information into single numeric value
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
        features['pitch'] = [pitch_into_vector(
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
        features['time_sig_vec'] = [time_signature_to_vector(
            note.tempo.time_signature) for note in piece_data.xml_notes]
        features['following_rest'] = [note.following_rest_duration /
                                      note.state_fixed.divisions for note in piece_data.xml_notes]
        features['followed_by_fermata_rest'] = [
            int(note.followed_by_fermata_rest) for note in piece_data.xml_notes]
        features['notation'] = [note_notation_to_vector(
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
        # Cresciuto should be concatenated with dynamic vector (features['dynamic'])
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

def cal_beat_importance(beat_position, numerator):
    # beat_position : [0-1), note's relative position in measure
    if beat_position == 0:
        beat_importance = 4
    elif beat_position == 0.5 and numerator in [2, 4, 6, 12]:
        beat_importance = 3
    elif abs(beat_position - (1/3)) < 0.001 and numerator in [3, 9]:
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


def pitch_into_vector(pitch):
    # TODO: should be located in general file. maybe utils?
    pitch_vec = [0] * 13  # octave + pitch class
    octave = (pitch // 12) - 1
    octave = (octave - 4) / 4  # normalization
    pitch_class = pitch % 12

    pitch_vec[0] = octave
    pitch_vec[pitch_class+1] = 1

    return pitch_vec


def time_signature_to_vector(time_signature):
    numerator = time_signature.numerator
    denominator = time_signature.denominator

    denominator_list = [2, 4, 8, 16]
    numerator_vec = [0] * 5
    denominator_vec = [0] * 4

    if denominator == 32:
        denominator_vec[-1] = 1
    else:
        denominator_type = denominator_list.index(denominator)
        denominator_vec[denominator_type] = 1

    if numerator == 2:
        numerator_vec[0] = 1
    elif numerator == 3:
        numerator_vec[1] = 1
    elif numerator == 4:
        numerator_vec[0] = 1
        numerator_vec[2] = 1
    elif numerator == 6:
        numerator_vec[0] = 1
        numerator_vec[3] = 1
    elif numerator == 9:
        numerator_vec[1] = 1
        numerator_vec[3] = 1
    elif numerator == 12:
        numerator_vec[0] = 1
        numerator_vec[2] = 1
        numerator_vec[3] = 1
    elif numerator == 24:
        numerator_vec[0] = 1
        numerator_vec[2] = 1
        numerator_vec[3] = 1
    else:
        print('Unclassified numerator: ', numerator)
        numerator_vec[4] = 1

    return numerator_vec + denominator_vec


def note_notation_to_vector(note):
    # trill, tenuto, accent, staccato, fermata
    keywords = ['is_trill', 'is_tenuto', 'is_accent', 'is_staccato',
                'is_fermata', 'is_arpeggiate', 'is_strong_accent', 'is_cue', 'is_slash']
    # keywords = ['is_trill', 'is_tenuto', 'is_accent', 'is_staccato', 'is_fermata']

    notation_vec = [0] * len(keywords)

    for i in range(len(keywords)):
        key = keywords[i]
        if getattr(note.note_notations, key) == True:
            notation_vec[i] = 1

    return notation_vec


def make_index_continuous(features, score=False):
    # Sometimes a beat or a measure can contain no notes at all.
    # In this case, the sequence of beat index or measure indices of notes are not continuous,
    # e.g. 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4 ...
    # This function ommits the beat or measure without any notes so that entire sequence of indices become continuous
    prev_beat = 0
    prev_measure = 0

    beat_compensate = 0
    measure_compensate = 0

    for feat in features:
        if feat.qpm is not None or score:
            if feat.note_location.beat - prev_beat > 1:
                beat_compensate -= (feat.note_location.beat - prev_beat) - 1
            if feat.note_location.measure - prev_measure > 1:
                measure_compensate -= (feat.note_location.measure -
                                       prev_measure) - 1

            prev_beat = feat.note_location.beat
            prev_measure = feat.note_location.measure

            feat.note_location.beat += beat_compensate
            feat.note_location.measure += measure_compensate
        else:
            continue
    return features


class NoteLocation:
    def __init__(self, beat, measure, voice, section):
        self.beat = beat
        self.measure = measure
        self.voice = voice
        self.section = section
