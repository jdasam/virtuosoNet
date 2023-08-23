from .utils import binary_index

import pretty_midi
THRESHOLD = 64
OVERLAP_THR = 0.03
BARELY_OFF = 20
BARELY_ON = 68
PEDAL_OFF_TIME_MARGIN = 0.15

class XML_Pedal:
    def __init__(self, number, value, time, xml_position):
        self.number = number
        self.value = value
        self.time = time
        self.xml_position = xml_position
    def __str__(self):

        return 'Value: {}, time: {}, position: {}'.format(self.value, self.time, self.xml_position)


def predicted_pedals_to_midi_pedals(xml_notes, eps=0.03):
    note_end_pedals = make_note_pedal_to_CC(xml_notes, 'at_end', eps=eps)
    note_start_pedals = make_note_pedal_to_CC(xml_notes, 'at_start', eps=eps)
    refresh_pedals = make_note_pedal_to_CC(xml_notes, 'refresh', eps=eps)
    cut_pedals = make_note_pedal_to_CC(xml_notes, 'cut', eps=eps)
    soft_pedals = make_note_pedal_to_CC(xml_notes, 'soft', eps=eps)

    note_end_pedals = clean_overlapped_pedals(note_end_pedals)
    note_start_pedals = clean_overlapped_pedals(note_start_pedals)
    soft_pedals = clean_overlapped_pedals(soft_pedals)

    refresh_pedals = clean_refresh_or_cut_pedals(refresh_pedals)
    cut_pedals = clean_refresh_or_cut_pedals(cut_pedals)

    total_pedals = add_two_pedals_without_collision(note_end_pedals, note_start_pedals)
    total_pedals = add_pedal_offset_between_pedals(total_pedals, refresh_pedals)
    total_pedals = add_pedal_offset_between_pedals(total_pedals, cut_pedals)

    total_pedals += soft_pedals

    total_pedals = xml_pedals_to_midi_pedals(total_pedals)

    return total_pedals


def xml_pedals_to_midi_pedals(xml_pedals):
    midi_pedals = []
    for pedal in xml_pedals:
        midi_pedal = pretty_midi.ControlChange(number=pedal.number, value=pedal.value, time=pedal.time)
        midi_pedals.append(midi_pedal)
    return midi_pedals


def make_note_pedal_to_CC(xml_notes, pedal_type='at_end', eps=0.03):
    num_notes = len(xml_notes)
    pedals = []
    if pedal_type == 'soft':
        pedal_CC_number = 67
    else:
        pedal_CC_number = 64
    for i in range(num_notes):
        note = xml_notes[i]
        feat = note.pedal
        if pedal_type == 'refresh' and not check_refresh_valid(feat):
            continue
        if pedal_type == 'cut' and not check_cut_valid(xml_notes, i):
            continue
        pedal_time = set_pedal_time(note, feat, pedal_type, eps=eps)
        if pedal_type in ['cut', 'at_end']:
            xml_position = note.note_duration.xml_position + note.note_duration.duration
        else:
            xml_position = note.note_duration.xml_position
        end_pedal = XML_Pedal(number=pedal_CC_number, value=to_8(getattr(feat, pedal_type)), time=pedal_time,
                              xml_position=xml_position)
        pedals.append(end_pedal)

    pedals.sort(key=lambda x: (x.xml_position, x.time))
    return pedals


def set_pedal_time(xml_note, feat, pedal_type, eps=0.03):
    if pedal_type == 'at_end':
        pedal_time = xml_note.note_duration.time_position + xml_note.note_duration.seconds
        pedal_time -= eps
    elif pedal_type == 'at_start' or pedal_type == 'soft':
        pedal_time = xml_note.note_duration.time_position
        pedal_time -= eps
    elif pedal_type == 'refresh':
        pedal_time = xml_note.note_duration.time_position + feat.refresh_time
    elif pedal_type == 'cut':
        pedal_time = xml_note.note_duration.time_position + xml_note.note_duration.seconds + feat.cut_time
    return pedal_time


def check_refresh_valid(feat):
    if feat.refresh < THRESHOLD < feat.at_start:
        return True
    else:
        return False


def check_cut_valid(xml_notes, note_index):
    note = xml_notes[note_index]
    feat = note.pedal
    if feat.at_end < THRESHOLD:
        return False
    if feat.cut > THRESHOLD:
        return False
    note_start = note.note_duration.xml_position
    note_end = note_start + note.note_duration.duration
    num_notes = len(xml_notes)
    front_search_range = min(20, note_index+1)
    for i in range(1, front_search_range):
        previous_note = xml_notes[note_index - i]
        previous_note_end = previous_note.note_duration.xml_position + previous_note.note_duration.duration
        if previous_note_end > note_end:
            return False

    for i in range(note_index+1, num_notes):
        next_note = xml_notes[i]
        next_note_start = next_note.note_duration.xml_position
        next_note_end = next_note_start + next_note.note_duration.duration

        if next_note_start < note_start and next_note_end > note_end:
            return False
        if next_note_start == note_end:
            return False
        elif next_note_start > note_end:
            return True


def clean_refresh_or_cut_pedals(pedals):
    num_pedals = len(pedals)
    cleaned_pedals = []
    i = 0
    while i < num_pedals:
        pedal = pedals[i]
        same_onset_pedal = [pedal]
        for j in range(i+1, num_pedals-i):
            next_pedal = pedals[j]
            if pedal.xml_position < next_pedal.xml_position:
                i = j-1
                break
            elif pedal.xml_position == next_pedal.xml_position:
                same_onset_pedal.append(next_pedal)
        earliest_pedal = find_earliest_pedal(same_onset_pedal)
        cleaned_pedals.append(earliest_pedal)
        i += 1
    return cleaned_pedals


def clean_overlapped_pedals(pedals):
    num_pedals = len(pedals)
    cleaned_pedals = []
    i = 0
    while i < num_pedals:
        pedal = pedals[i]
        same_onset_pedal = [pedal]
        for j in range(i + 1, num_pedals - i):
            next_pedal = pedals[j]
            if pedal.xml_position < next_pedal.xml_position:
                break
            elif pedal.xml_position == next_pedal.xml_position and abs(pedal.time - next_pedal.time) < OVERLAP_THR:
                same_onset_pedal.append(next_pedal)

        selected_pedal = make_representative_pedal(same_onset_pedal)
        cleaned_pedals.append(selected_pedal)
        i += len(same_onset_pedal)
    return cleaned_pedals


def find_earliest_pedal(pedals):
    num_pedals = len(pedals)
    earliest_time = float("inf")
    earliest_index = 0
    for i in range(num_pedals):
        pedal = pedals[i]
        if pedal.time < earliest_time:
            earliest_index = i
            earliest_time = pedal.time

    return pedals[earliest_index]


def make_representative_pedal(pedals):
    num_pedals = len(pedals)
    earliest_time = float("inf")
    mean_pedal_value = 0
    for i in range(num_pedals):
        pedal = pedals[i]
        if pedal.time < earliest_time:
            earliest_time = pedal.time
        mean_pedal_value += pedal.value
    mean_pedal_value /= num_pedals

    return XML_Pedal(pedal.number, int(round(mean_pedal_value)), earliest_time, pedal.xml_position)


def add_two_pedals_without_collision(pedals_a, pedals_b):
    combined_pedals = []
    pedals_a_positions = [x.xml_position for x in pedals_a]

    for pedal in pedals_b:
        nearby_pedal_index = binary_index(pedals_a_positions, pedal.xml_position)
        nearby_pedal_a = pedals_a[nearby_pedal_index]
        if abs(nearby_pedal_a.time - pedal.time) < OVERLAP_THR:
            continue
        else:
            combined_pedals.append(pedal)

    combined_pedals += pedals_a
    combined_pedals.sort(key=lambda x: (x.xml_position, x.time))

    return combined_pedals


def add_pedal_offset_between_pedals(pedals, offsets):
    combined_pedal = []
    pedals_position = [x.xml_position for x in pedals]
    for off_pedal in offsets:
        nearby_pedal_index = binary_index(pedals_position, off_pedal.xml_position)
        prev_pedal = pedals[nearby_pedal_index]
        if prev_pedal.value < THRESHOLD:
            continue
        next_pedal_on = find_next_pedal_on(pedals, nearby_pedal_index+1)

        if next_pedal_on is None:
            combined_pedal.append(off_pedal)
        elif off_pedal.time + OVERLAP_THR > next_pedal_on.time:
            off_pedal.time = prev_pedal.time + OVERLAP_THR
            off_pedal.value = BARELY_OFF
            prev_pedal.value = BARELY_ON
            combined_pedal.append(off_pedal)
        elif off_pedal.time + PEDAL_OFF_TIME_MARGIN > next_pedal_on.time:
            off_pedal.time = min(prev_pedal.time + OVERLAP_THR * 2, off_pedal.time)
            off_pedal.value = BARELY_OFF
            prev_pedal.value = BARELY_ON
            combined_pedal.append(off_pedal)

    combined_pedal += pedals
    combined_pedal.sort(key=lambda x: (x.xml_position, x.time))
    return combined_pedal


def find_next_pedal_on(pedals, start_index):
    num_pedals = len(pedals)
    for i in range(start_index, num_pedals):
        pedal = pedals[i]
        if pedal.value > THRESHOLD:
            return pedal
    return None


def to_8(value):
    return int(min(max(round(value), 0), 127))
