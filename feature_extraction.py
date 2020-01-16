from . import xml_direction_encoding as dir_enc
from . import xml_utils, utils
import copy
import math


class ScoreExtractor:
    def __init__(self, feature_keys):
        self.selected_feature_keys = feature_keys

        self.dyn_emb_tab = dir_enc.define_dyanmic_embedding_table()
        self.tem_emb_tab = dir_enc.define_tempo_embedding_table()

    def extract_score_features(self, piece_data):
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

        piece_data.qpm_primo = piece_data.xml_notes[0].state_fixed.qpm
        tempo_primo_word = dir_enc.direction_words_flatten(
            piece_data.xml_notes[0].tempo)
        if tempo_primo_word:
            piece_data.tempo_primo = dir_enc.dynamic_embedding(
                tempo_primo_word, self.tem_emb_tab, 5)
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
                dir_enc.direction_words_flatten(note.dynamic), self.dyn_emb_tab, len_vec=4)
            for note in piece_data.xml_notes]

        features['tempo'] = [
            dir_enc.dynamic_embedding(
                dir_enc.direction_words_flatten(note.tempo), self.tem_emb_tab, len_vec=5)
            for note in piece_data.xml_notes]

        features['cresciuto'] = _get_cresciuto(piece_data)
        # TODO: maybe its redundant?
        # Cresciuto was always concatenated with dynamic vector (features['dynamic'])
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
        features['note_location'] = make_index_continuous(features['note_location'])

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


class PerformExtractor:
    def __init__(self, selected_feature_keys):
        self.selected_feature_keys = selected_feature_keys

    def extract_perform_features(self, piece_data, perform_data):
        perform_data.perform_features = {}
        for feature_key in self.selected_feature_keys:
            perform_data.perform_features[feature_key] = getattr(self, 'get_' + feature_key)(piece_data, perform_data)

        return perform_data.perform_features

    def get_beat_tempo(self, piece_data, perform_data):
        tempos = cal_tempo_by_positions(piece_data.beat_positions, perform_data.pairs)
        # update tempos for perform data
        perform_data.tempos = tempos
        return [math.log(utils.get_item_by_xml_position(tempos, note).qpm, 10) for note in piece_data.xml_notes]

    def get_measure_tempo(self, piece_data, perform_data):
        tempos = cal_tempo_by_positions(piece_data.measure_positions, perform_data.pairs)
        return [math.log(utils.get_item_by_xml_position(tempos, note).qpm, 10) for note in piece_data.xml_notes]

    def get_section_tempo(self, piece_data, perform_data):
        tempos = cal_tempo_by_positions(piece_data.section_positions, perform_data.pairs)
        return [math.log(utils.get_item_by_xml_position(tempos, note).qpm, 10) for note in piece_data.xml_notes]

    def get_qpm_primo(self, piece_data, perform_data, view_range=10):
        if 'beat_tempo' not in perform_data.perform_features:
            perform_data.perform_features['beat_tempo'] = self.get_beat_tempo(piece_data, perform_data)
        qpm_primo = 0
        for i in range(view_range):
            tempo = perform_data.perform_features['beat_tempo'][i]
            qpm_primo += tempo.qpm

        return qpm_primo / view_range

    def get_articulation(self, piece_data, perform_data, trill_length):
        features = []
        if 'beat_tempo' not in perform_data.perform_features:
            perform_data.perform_features['beat_tempo'] = self.get_beat_tempo(piece_data, perform_data)
        for pair, tempo in zip(perform_data.pairs, perform_data.perform_features['beat_tempo']):
            if pair == []:
                articulation = 0
            else:
                note = pair['xml']
                midi = pair['midi']
                xml_duration = note.note_duration.duration
                if xml_duration == 0:
                    return 0
                duration_as_quarter = xml_duration / note.state_fixed.divisions
                second_in_tempo = duration_as_quarter / tempo * 60
                if trill_length:
                    actual_second = trill_length
                else:
                    actual_second = midi.end - midi.start

                articulation = actual_second / second_in_tempo
                if articulation > 6:
                    print('check: articulation is ' + str(articulation))
                articulation = math.log(articulation, 10)
            features.append(articulation)
        return features

    def get_onset_deviation(self, piece_data, perform_data):
        features = []
        if 'beat_tempo' not in perform_data.perform_features:
            perform_data.perform_features['beat_tempo'] = self.get_beat_tempo(piece_data, perform_data)
        for pair, tempo in zip(perform_data.pairs, perform_data.perform_features['beat_tempo']):
            if pair == []:
                deviation = 0
            else:
                note = pair['xml']
                midi = pair['midi']

                tempo_start = tempo.time_position

                passed_duration = note.note_duration.xml_position - tempo.xml_position
                actual_passed_second = midi.start - tempo_start
                actual_passed_duration = actual_passed_second / 60 * tempo.qpm * note.state_fixed.divisions

                xml_pos_difference = actual_passed_duration - passed_duration
                pos_diff_in_quarter_note = xml_pos_difference / note.state_fixed.divisions
                # deviation_time = xml_pos_difference / note.state_fixed.divisions / tempo_obj.qpm * 60
                # if pos_diff_in_quarter_note >= 0:
                #     pos_diff_sqrt = math.sqrt(pos_diff_in_quarter_note)
                # else:
                #     pos_diff_sqrt = -math.sqrt(-pos_diff_in_quarter_note)
                # pos_diff_cube_root = float(pos_diff_
                deviation = pos_diff_in_quarter_note
            features.append(deviation)
        return features

    def get_align_matched(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                matched = 0
            else:
                matched = 1
            features.append(matched)
        return features

    def get_velocity(self, piece_data, perform_data):
        features = []
        prev_velocity = 64
        for pair in perform_data.pairs:
            if pair == []:
                velocity = prev_velocity
            else:
                velocity = pair['midi'].velocity
                prev_velocity = velocity
            features.append(velocity)
        return features

    # TODO: get pedal _ can be simplified

    def get_pedal_at_start(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = pedal_sigmoid(pair['midi'].pedal_at_start)
                prev_pedal = pedal
            features.append(pedal)
        return features

    def get_pedal_at_end(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = pedal_sigmoid(pair['midi'].pedal_at_end)
                prev_pedal = pedal
            features.append(pedal)
        return features

    def get_pedal_refresh_time(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                pedal = 0
            else:
                pedal = pedal_sigmoid(pair['midi'].pedal_refresh_time)
            features.append(pedal)
        return features

    def get_pedal_cut(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = pedal_sigmoid(pair['midi'].pedal_cut)
                prev_pedal = pedal
            features.append(pedal)
        return features

    def get_pedal_cut_time(self, piece_data, perform_data):
        features = []
        for pair in perform_data.pairs:
            if pair == []:
                pedal = 0
            else:
                pedal = pedal_sigmoid(pair['midi'].pedal_cut_time)
            features.append(pedal)
        return features

    def get_pedal_soft_pedal(self, piece_data, perform_data):
        features = []
        prev_pedal = 0
        for pair in perform_data.pairs:
            if pair == []:
                pedal = prev_pedal
            else:
                pedal = pedal_sigmoid(pair['midi'].soft_pedal)
                prev_pedal = pedal
            features.append(pedal)
        return features

    def get_articulation_loss_weight(self, piece_data, perform_data):
        if 'pedal_at_end' not in perform_data.perform_features:
            perform_data.perform_features['pedal_at_end'] = get_pedal_at_end(piece_data, perform_data)
        if 'pedal_refresh' not in perform_data.perform_features:
            perform_data.perform_features['pedal_refresh'] = get_pedal_at_end(piece_data, perform_data)
        features = []
        for pair, pedal, pedal_refresh in zip(perform_data.pairs,
                                              perform_data.perform_features['pedal_at_end'],
                                              perform_data.perform_features['pedal_refresh']):
            if pair == []:
                articulation_loss_weight = 0
            elif pedal > 70:
                articulation_loss_weight = 0.05
            elif pedal > 60:
                articulation_loss_weight = 0.5
            else:
                articulation_loss_weight = 1

            if pedal > 64 and pedal_refresh < 64:
                # pedal refresh occurs in the note
                articulation_loss_weight = 1

            features.append(articulation_loss_weight)
        return features


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


def make_index_continuous(note_locations):
    # Sometimes a beat or a measure can contain no notes at all.
    # In this case, the sequence of beat index or measure indices of notes are not continuous,
    # e.g. 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4 ...
    # This function ommits the beat or measure without any notes so that entire sequence of indices become continuous
    prev_beat = 0
    prev_measure = 0

    beat_compensate = 0
    measure_compensate = 0

    for loc_data in note_locations:
        if loc_data.beat - prev_beat > 1:
            beat_compensate -= (loc_data.beat - prev_beat) - 1
        if loc_data.measure - prev_measure > 1:
            measure_compensate -= (loc_data.measure -
                                   prev_measure) - 1

        prev_beat = loc_data.beat
        prev_measure = loc_data.measure

        loc_data.beat += beat_compensate
        loc_data.measure += measure_compensate
    return note_locations


class NoteLocation:
    def __init__(self, beat, measure, voice, section):
        self.beat = beat
        self.measure = measure
        self.voice = voice
        self.section = section

class Tempo:
    def __init__(self, xml_position, qpm, time_position, end_xml, end_time):
        self.qpm = qpm
        self.xml_position = xml_position
        self.time_position = time_position
        self.end_time = end_time
        self.end_xml = end_xml

    def __str__(self):
        string = '{From ' + str(self.xml_position)
        string += ' to ' + str(self.end_xml)
        return string

def cal_tempo_by_positions(beats, position_pairs):
    #
    tempos = []
    num_beats = len(beats)
    previous_end = 0

    for i in range(num_beats-1):
        beat = beats[i]
        current_pos_pair = utils.get_item_by_xml_position(position_pairs, beat)
        if current_pos_pair.xml_position < previous_end:
            continue

        next_beat = beats[i+1]
        next_pos_pair = utils.get_item_by_xml_position(position_pairs, next_beat)

        if next_pos_pair.xml_position == previous_end:
            continue

        if current_pos_pair == next_pos_pair:
            continue

        cur_xml = current_pos_pair.xml_position
        cur_time = current_pos_pair.time_position
        cur_divisions = current_pos_pair.divisions
        next_xml = next_pos_pair.xml_position
        next_time = next_pos_pair.time_position

        qpm = (next_xml - cur_xml) / (next_time - cur_time) / cur_divisions * 60

        if qpm > 1000:
            print('need check: qpm is ' + str(qpm) +', current xml_position is ' + str(cur_xml))
        tempo = Tempo(cur_xml, qpm, cur_time, next_xml, next_time)
        tempos.append(tempo)        #
        previous_end = next_pos_pair.xml_position

    return tempos


def make_available_xml_midi_positions(pairs):
    global NUM_EXCLUDED_NOTES
    class PositionPair:
        def __init__(self, xml_pos, time, pitch, index, divisions):
            self.xml_position = xml_pos
            self.time_position = time
            self.pitch = pitch
            self.index = index
            self.divisions = divisions
            self.is_arpeggiate = False

    available_pairs = []
    num_pairs = len(pairs)
    for i in range(num_pairs):
        pair = pairs[i]
        if not pair == []:
            xml_note = pair['xml']
            midi_note = pair['midi']
            xml_pos = xml_note.note_duration.xml_position
            time = midi_note.start
            divisions = xml_note.state_fixed.divisions
            if not xml_note.note_duration.is_grace_note:
                pos_pair = PositionPair(xml_pos, time, xml_note.pitch[1], i, divisions)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair.is_arpeggiate = True
                available_pairs.append(pos_pair)

    # available_pairs = save_lowest_note_on_same_position(available_pairs)
    available_pairs, mismatched_indexes = make_average_onset_cleaned_pair(available_pairs)
    print('Number of mismatched notes: ', len(mismatched_indexes))
    NUM_EXCLUDED_NOTES += len(mismatched_indexes)
    for index in mismatched_indexes:
        pairs[index] = []

    return pairs, available_pairs



def make_available_note_feature_list(notes, features, predicted):
    class PosTempoPair:
        def __init__(self, xml_pos, pitch, qpm, index, divisions, time_pos):
            self.xml_position = xml_pos
            self.qpm = qpm
            self.index = index
            self.divisions = divisions
            self.pitch = pitch
            self.time_position = time_pos
            self.is_arpeggiate = False

    if not predicted:
        available_notes = []
        num_features = len(features)
        for i in range(num_features):
            feature = features[i]
            if not feature.qpm == None:
                xml_note = notes[i]
                xml_pos = xml_note.note_duration.xml_position
                time_pos = feature.midi_start
                divisions = xml_note.state_fixed.divisions
                qpm = feature.qpm
                pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair.is_arpeggiate = True
                available_notes.append(pos_pair)

    else:
        available_notes = []
        num_features = len(features)
        for i in range(num_features):
            feature = features[i]
            xml_note = notes[i]
            xml_pos = xml_note.note_duration.xml_position
            time_pos = xml_note.note_duration.time_position
            divisions = xml_note.state_fixed.divisions
            qpm = feature.qpm
            pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
            available_notes.append(pos_pair)

    # minimum_time_interval = 0.05
    # available_notes = save_lowest_note_on_same_position(available_notes, minimum_time_interval)
    available_notes, _ = make_average_onset_cleaned_pair(available_notes)
    return available_notes


def make_average_onset_cleaned_pair(position_pairs, maximum_qpm=600):
    length = len(position_pairs)
    previous_position = -float("Inf")
    previous_time = -float("Inf")
    previous_index = 0
    # position_pairs.sort(key=lambda x: (x.xml_position, x.pitch))
    cleaned_list = list()
    notes_in_chord = list()
    mismatched_indexes = list()
    for i in range(length):
        pos_pair = position_pairs[i]
        current_position = pos_pair.xml_position
        current_time = pos_pair.time_position
        if current_position > previous_position >= 0:
            minimum_time_interval = (current_position - previous_position) / pos_pair.divisions / maximum_qpm * 60 + 0.01
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair.time_position = get_average_onset_time(notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair.time_position > cleaned_list[-1].time_position + (
                        (average_pos_pair.xml_position - cleaned_list[-1].xml_position) /
                        average_pos_pair.divisions / maximum_qpm * 60 + 0.01):
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            # print('the note is far from other notes in the chord')
                            mismatched_indexes.append(note.index)
                else:
                    # print('the onset is too close to the previous onset', average_pos_pair.xml_position, cleaned_list[-1].xml_position, average_pos_pair.time_position, cleaned_list[-1].time_position)
                    for note in notes_in_chord:
                        mismatched_indexes.append(note.index)
            notes_in_chord = list()
            notes_in_chord.append(pos_pair)
            previous_position = current_position
            previous_time = current_time
            previous_index = i
        elif current_position == previous_position:
            notes_in_chord.append(pos_pair)
        else:
            # print('the note is too close to the previous note', current_position - previous_position, current_time - previous_time)
            # print(previous_position, current_position, previous_time, current_time)
            mismatched_indexes.append(position_pairs[previous_index].index)
            mismatched_indexes.append(pos_pair.index)

    return cleaned_list, mismatched_indexes


def get_average_onset_time(notes_in_chord_saved, threshold=0.2):
    # notes_in_chord: list of PosTempoPair, len > 0
    notes_in_chord = copy.copy(notes_in_chord_saved)
    average_onset_time = 0
    for pos_pair in notes_in_chord:
        average_onset_time += pos_pair.time_position
        if pos_pair.is_arpeggiate:
            threshold = 1
    average_onset_time /= len(notes_in_chord)

    # check whether there is mis-matched note
    deviations = list()
    for pos_pair in notes_in_chord:
        dev = abs(pos_pair.time_position - average_onset_time)
        deviations.append(dev)
    if max(deviations) > threshold:
        # print(deviations)
        if len(notes_in_chord) == 2:
            del notes_in_chord[0:2]
        else:
            index = deviations.index(max(deviations))
            del notes_in_chord[index]
            notes_in_chord, average_onset_time = get_average_onset_time(notes_in_chord, threshold)

    return notes_in_chord, average_onset_time


def pedal_sigmoid(pedal_value, k=8):
    sigmoid_pedal = 127 / (1 + math.exp(-(pedal_value-64)/k))
    return int(sigmoid_pedal)




def get_trill_parameters():
    return