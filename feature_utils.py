import math
import utils

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
    sigmoid_pedal = 127 / (1 + math.exp(-(pedal_value-64)/k))
    return int(sigmoid_pedal)




def get_trill_parameters():
    return