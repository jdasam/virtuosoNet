""" Utilities for feature extraction

Interface summary:

        import feature_utils

        pitch_vector = feature_utils.pitch_into_vector(pitch)

call in feature_extraction.py

get feature information to generate or modify feature
"""

import math
from . import utils

def cal_beat_importance(beat_position, numerator):
    """ Returns beat importance in integer

    Args:
        beat_position (integer): [0-1), note's relative position in measure
        numerator (integer): note tempo's time numerator 
    
    Returns:
        beat_importance (integer): importance of each beat in integer format

    Example:
        (in feature_extraction.py -> ScoreExtractor().extract_score_features()._get_beat_importance())
        >>> beat_positions = _get_beat_position(piece_data)
        >>> for i, note in enumerate(piece_data.xml_notes):
        >>>    importance = feature_utils.cal_beat_importance(
                    beat_positions[i], note.tempo.time_numerator)

    """
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
    """ Returns pitch vector from midi pitch value
    octave value is normalized

    Args:
        pitch (integer) : pitch value in midi number

    Returns:
        pitch_vec (1-D list) : vector with [octave value, 12-class 1-hot vector] in shape (13, )
    
    Example:
        (in feature_extraction.py -> ScoreExtractor().extract_score_features())
        >>> features['pitch'] = [feature_utils.pitch_into_vector(
            note.pitch[1]) for note in piece_data.xml_notes]
    """
    # TODO: should be located in general file. maybe utils?
    pitch_vec = [0] * 13  # octave + pitch class
    octave = (pitch // 12) - 1
    octave = (octave - 4) / 4  # normalization
    pitch_class = pitch % 12

    pitch_vec[0] = octave
    pitch_vec[pitch_class+1] = 1

    return pitch_vec


def time_signature_to_vector(time_signature):
    """ Returns

    Args:
        time_signature
    
    Returns:
        time signature vector (1-D list)
            : appended list of numerator_vec and denominator_vec in shape (9, )
              numerator_vec (1-D list) : multi-hot vector correspond to each numerator value (integer) in shape (5, )
              denominator_vec (1-D list) : one-hot vector correspond to each denominator value (integer) in shape (4, )
    
    Example:
        (in feature_extraction.py -> ScoreExtractor().extract_score_features())
        >>> features['time_sig_vec'] = [feature_utils.time_signature_to_vector(
            note.tempo.time_signature) for note in piece_data.xml_notes]
    """
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
    """ Returns note notation vector

    Args:
        note: Note() object in xml_notes

    Returns:
        notation_vec (1-D list): multi-hot vector represents note notation in shape (num_keywords, )
    
    Example:
        (in feature_extraction.py -> ScoreExtractor().extract_score_features())
        >>> features['notation'] = [feature_utils.note_notation_to_vector(
                                        note) for note in piece_data.xml_notes]

    """
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
    """ Returns continuous note location list

    Sometimes a beat or a measure can contain no notes at all.
    In this case, the sequence of beat index or measure indices of notes are not continuous,
    e.g. 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4 ...
    This function ommits the beat or measure without any notes so that entire sequence of indices become continuous
    
    Args:
        note_locations (1-D list) : list of .NoteLocation object
    
    Returns:
        note_locations (1-D list) : list of .NoteLocation object
    
    Example:
        (in feature_extraction.py -> ScoreExtractor().extract_score_features())
        >>> features['note_location'] = feature_utils.make_index_continuous(features['note_location'])
    """
    prev_beat = 0
    beat_compensate = 0

    for beat_idx in note_locations:
        if beat_idx - prev_beat > 1:
            beat_compensate -= (beat_idx - prev_beat) - 1
        prev_beat = beat_idx
        beat_idx += beat_compensate

    # prev_measure = 0
    # measure_compensate = 0
    # for loc_data in note_locations:
    #     if loc_data.beat - prev_beat > 1:
    #         beat_compensate -= (loc_data.beat - prev_beat) - 1
    #     if loc_data.measure - prev_measure > 1:
    #         measure_compensate -= (loc_data.measure -
    #                                prev_measure) - 1

    #     prev_beat = loc_data.beat
    #     prev_measure = loc_data.measure

    #     loc_data.beat += beat_compensate
    #     loc_data.measure += measure_compensate
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
    """ Returns list of Tempo objects

    Args:
        beats (1-D list): list of beats in piece
        position_pairs (1-D list): list of valid pair dictionaries {xml_note, midi_note} 
    
    Returns:
        tempos (1-D list): list of .Tempo object
    
    Example:
        (in feature_extraction.py -> PerformExtractor().get_beat_tempo())
        >>> tempos = feature_utils.cal_tempo_by_positions(piece_data.beat_positions, perform_data.valid_position_pairs)
        
    """
    tempos = []
    num_beats = len(beats)
    previous_end = 0

    for i in range(num_beats-1):
        beat = beats[i]
        current_pos_pair = utils.get_item_by_xml_position(position_pairs, beat)
        if current_pos_pair['xml_position'] < previous_end:
            continue

        next_beat = beats[i+1]
        next_pos_pair = utils.get_item_by_xml_position(position_pairs, next_beat)

        if next_pos_pair['xml_position'] == previous_end:
            continue

        if current_pos_pair == next_pos_pair:
            continue

        cur_xml = current_pos_pair['xml_position']
        cur_time = current_pos_pair['time_position']
        cur_divisions = current_pos_pair['divisions']
        next_xml = next_pos_pair['xml_position']
        next_time = next_pos_pair['time_position']
        qpm = (next_xml - cur_xml) / (next_time - cur_time) / cur_divisions * 60

        if qpm > 1000:
            print('need check: qpm is ' + str(qpm) +', current xml_position is ' + str(cur_xml))
        tempo = Tempo(cur_xml, qpm, cur_time, next_xml, next_time)
        tempos.append(tempo)        #
        previous_end = next_pos_pair['xml_position']

    return tempos



def pedal_sigmoid(pedal_value, k=8):
    """ Returns

    Args:
        pedal_value (integer) : pedal value in midi number
        k (integer)
    
    Returns:
        sigmoid_pedal (integer) : sigmoid pedal value

    Example:
        (in feature_extraction.py -> PerformExtractor().get_pedal_at_start())
        >>> pedal = feature_utils.pedal_sigmoid(pair['midi'].pedal_at_start)
    """
    sigmoid_pedal = 127 / (1 + math.exp(-(pedal_value-64)/k))
    return int(sigmoid_pedal)


def get_trill_parameters():
    return


def composer_name_to_vec(composer_name):
    composer_name_list = ['Bach','Balakirev', 'Beethoven', 'Brahms', 'Chopin', 'Debussy', 'Glinka', 'Haydn',
                          'Liszt', 'Mozart', 'Prokofiev', 'Rachmaninoff', 'Ravel', 'Schubert', 'Schumann', 'Scriabin']
    one_hot_vec = [0] * (len(composer_name_list)  + 1)
    if composer_name in composer_name_list:
        index = composer_name_list.index(composer_name)
    else:
        index = len(composer_name_list)
        print('The given composer name {} is not in the list'.format(composer_name))
    one_hot_vec[index] = 1

    return one_hot_vec


def get_longer_level_dynamics(features, note_locations, length='beat'):
    num_notes = len(note_locations)

    prev_beat = 0
    prev_beat_index = 0
    temp_beat_dynamic = []

    longer_dynamics = [0] * num_notes

    for i in range(num_notes):
        if not features['align_matched'][i]:
            continue
        current_beat = getattr(note_locations[i], length)

        if current_beat > prev_beat and temp_beat_dynamic != []:
            prev_beat_dynamic = (sum(temp_beat_dynamic) / len(temp_beat_dynamic) + max(temp_beat_dynamic)) / 2
            for j in range(prev_beat_index, i):
                longer_dynamics[j] = prev_beat_dynamic
            temp_beat_dynamic = []
            prev_beat = current_beat
            prev_beat_index = i

        temp_beat_dynamic.append(features['velocity'][i])

    if temp_beat_dynamic != []:
        prev_beat_dynamic = (sum(temp_beat_dynamic) + max(temp_beat_dynamic)) / 2 / len(temp_beat_dynamic)

        for j in range(prev_beat_index, num_notes):
            longer_dynamics[j] = prev_beat_dynamic

    return longer_dynamics