from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pretty_midi
import warnings
import numpy as np
import copy


ONSET_DURATION = 0.032


class SustainPedal:
    """A sustain_pedal event.
    Parameters
    ----------
    number : int
        control number. {64, 127}
    value : int
        The value of the control change, in [0, 127].
    start, end : float or None
        Time where the control change occurs.
    """

    def __init__(self, start, end, value, number):
        self.number = number
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):
        return ('Sustain_Pedal (start={:f}, end={:f}, value={:d},  number={:d}'
                .format(self.start, self.end, self.value, self.number))

    def is_valid(self):
        return self.end is not None and self.end > self.start


def elongate_offset_by_pedal(midi_obj):
    """elongate off set of notes in midi_object, according to sustain pedal length.

    :param
        midi_obj: pretty_midi.PrettyMIDI object
    :return:
        pretty_midi.PrettyMIDI object
    """

    assert len(midi_obj.instruments) == 1
    pedals = read_sustain_pedal(midi_obj)
    for pedal in pedals:
        instrument = midi_obj.instruments[0]
        for note in instrument.notes:
            if pedal.start < note.end <= pedal.end:
                note.end = pedal.end

    return midi_obj


def add_pedal_inf_to_notes(midi_obj):
    assert len(midi_obj.instruments) == 1
    sustain_pedals = read_sustain_pedal(midi_obj)
    sustain_pedals_positions = [pedal.start for pedal in sustain_pedals]
    soft_pedals = read_sustain_pedal(midi_obj,search_target=(67,)) # soft pedal CC == 67
    soft_pedals_positions = [pedal.start for pedal in soft_pedals]
    sostenuto_pedals = read_sustain_pedal(midi_obj, search_target=(66,)) #sostenuto pedal CC == 66
    sostenuto_pedals_positions = [pedal.start for pedal in sostenuto_pedals]

    notes = midi_obj.instruments[0].notes
    saved_notes = copy.copy(notes)
    notes.sort(key=lambda note:note.start)
    threshold = 30
    soft_threshold = 30
    # notes_offset_sorted = notes
    # notes_offset_sorted.sort(key = lambda note: note.end)

    for note in notes:
        pedal_index_at_note_start = binaryIndex(sustain_pedals_positions, note.start)
        pedal_index_at_note_end = binaryIndex(sustain_pedals_positions, note.end)
        soft_pedal_index = binaryIndex(soft_pedals_positions, note.start)
        sostenuto_index_at_start = binaryIndex(sostenuto_pedals_positions, note.start)
        sostenuto_index_at_end = binaryIndex(sostenuto_pedals_positions, note.end)

        note.pedal_at_start = sustain_pedals[pedal_index_at_note_start].value
        note.pedal_at_end = sustain_pedals[pedal_index_at_note_end].value
        note.soft_pedal = soft_pedals[soft_pedal_index].value
        # note.sostenuto_at_start = sostenuto_pedals[sostenuto_index_at_start].value
        # note.sostenuto_at_end = sostenuto_pedals[sostenuto_index_at_end].value

        note.pedal_refresh, note.pedal_refresh_time = \
            cal_pedal_refresh_in_note(note, notes, sustain_pedals,sustain_pedals_positions,
                                        pedal_index_at_note_start, pedal_index_at_note_end)
        # note.sostenuto_refresh = check_pedal_refresh_in_note(note, notes, sostenuto_pedals, sostenuto_pedals_positions,
        #                                                  sostenuto_index_at_start, sostenuto_index_at_end)

        note.pedal_cut, note.pedal_cut_time = \
            cal_pedal_cut_after(note, notes, sustain_pedals, sustain_pedals_positions)
        # note.sostenuto_cut = check_pedal_cut(note, notes, sostenuto_pedals,
        #                                                                 sostenuto_pedals_positions)

    # new_notes = [0] * len(notes)
    # for note in notes:
    #     old_index = saved_notes.index(note)
    #     new_notes[old_index] = note
    # midi_obj.instruments[0].notes = new_notes
    return midi_obj

def cal_pedal_refresh_in_note(note, notes, pedals, pedals_positions, pd_ind1, pd_ind2):
    note_index = notes.index(note)
    search_time_end = note.end
    lowest_pedal_value = note.pedal_at_start
    lowest_pedal = None
    # counts only when pedal is pressed at start
    # if note.pedal_at_start == False:
    #     return False, 0

    # if note_index < len(notes) - 1:
    #     next_note = notes[note_index+1]
    #     if next_note.start < note.end:
    #         pd_ind2 = binaryIndex(pedals_positions, next_note.start)
    #         search_time_end = next_note.start

    # check only the pedal between note start and end
    for i in range(pd_ind1, pd_ind2):
        pedal = pedals[i]
        if pedal.value < lowest_pedal_value:
            lowest_pedal_value = pedal.value
            lowest_pedal = pedal
    if lowest_pedal:
        time_ratio = (lowest_pedal.start - note.start) #/ (note.end - note.start)
        return lowest_pedal_value, time_ratio
    else:
        return lowest_pedal_value, 0


def cal_pedal_cut(note, notes, pedals, pedals_positions, threshold=30):
    note_index = notes.index(note)
    lowest_pedal_value = note.pedal_at_start
    lowest_pedal = None
    if note_index == 0:
        return 0, 0
    #check that there is no activated notes when the note starts
    prev_notes = notes[0:note_index]
    prev_notes.sort(key=lambda note:note.end)
    index = -1
    while index-1 >= -note_index and prev_notes[index].end >= note.start:
        index += -1
    # if last_note.end > note.start:
    #     return False, 0

    last_note = prev_notes[index]

    pd1 = binaryIndex(pedals_positions, last_note.end)
    pd2 = binaryIndex(pedals_positions, note.start)
    for i in range(pd1, pd2):
        pedal = pedals[i]
        if pedal.value < lowest_pedal_value:
            lowest_pedal_value = pedal.value
            lowest_pedal = pedal
    notes.sort(key=lambda x:x.start)
    if lowest_pedal:
        time_ratio = (note.start - lowest_pedal.start) / (note.end - note.start)
        return lowest_pedal_value, time_ratio
    else:
        return lowest_pedal_value, 0


def cal_pedal_cut_after(note, notes, pedals, pedals_positions, threshold=30):
    note_index = notes.index(note)
    note_end = note.end
    lowest_pedal_value = note.pedal_at_end
    lowest_pedal = None
    if note_index == 0:
        return 0, 0
    #check that there is no activated notes when the note starts
    next_notes = notes[note_index+1:]
    next_onset = float('Inf')
    for nxt_nt in next_notes:
        if nxt_nt.start > note_end:
            next_onset = nxt_nt.start
            break
    # if last_note.end > note.start:
    #     return False, 0
    pd1 = binaryIndex(pedals_positions, note.end)
    pd2 = binaryIndex(pedals_positions, next_onset)

    for i in range(pd1, pd2):
        pedal = pedals[i]
        if pedal.value < lowest_pedal_value:
            lowest_pedal_value = pedal.value
            lowest_pedal = pedal
    notes.sort(key=lambda x:x.start)
    if lowest_pedal:
        time_ratio = (lowest_pedal.start - note_end)
        return lowest_pedal_value, time_ratio
    else:
        return lowest_pedal_value, 0

def to_midi_zero(midi_path, midi_min=21, midi_max=108, save_midi=False, save_name=None):
    """Convert midi files to midi-0 format (1 track). set resolution = 1000, tempo=120.

    :param
        midi_path: path to .mid file
        midi_min: minimum midi number to convert. belows will be ignored.
        midi_max: maximum midi number to convert. highers will be ignored.
        save_midi: if true, save midi with name *.midi0.mid
        save_name: full path of save file. if given, save midi file with given name.
    :return:
        0-type pretty_midi.Pretty_Midi object.
    """
    pretty_midi.pretty_midi.MAX_TICK = 1e10
    if save_name is None:
        save_name = midi_path.replace('.mid', '_midi0.mid')

    midi = pretty_midi.PrettyMIDI(midi_path)
    midi_new = pretty_midi.PrettyMIDI(resolution=1000, initial_tempo=120)
    instrument = pretty_midi.Instrument(0)
    for instruments in midi.instruments:
        for midi_note in instruments.notes:
            note_pitch = midi_note.pitch
            if midi_min <= note_pitch <= midi_max:
                instrument.notes.append(midi_note)
            else:
                print('note with pitch : {:d} detected. Omitted because note not in [{:d}, {:d}]'
                      .format(note_pitch, midi_min, midi_max))
        for controls in instruments.control_changes:
            instrument.control_changes.append(controls)
    midi_new.instruments.append(instrument)
    midi_new.remove_invalid_notes()
    if save_midi:
        midi_new.write(save_name)
    return midi_new


def read_sustain_pedal(midi_obj, threshold=0, search_target=(64, 127)):
    """Read sustain pedal in midi.

    :param
        midi_obj: pretty_midi.Pretty_Midi object.
        threshold: threshold of velocity to activate/deactivate pedal
    :return:
        list of SustainPedal objects
    """
    assert len(midi_obj.instruments) == 1
    instrument = midi_obj.instruments[0]
    pedals = []
    default_pedal = SustainPedal(0,0.0001,0,search_target[0])
    pedals.append(default_pedal) # always starts with zero pedal
    current_pedal = None
    for control in instrument.control_changes:
        # 64 is allocated for sustain pedal, but MAPS uses 127 as pedal
        if control.number in search_target:
            if control.value > threshold:
                if isinstance(current_pedal, SustainPedal):
                    current_pedal.end = control.time
                    pedals.append(current_pedal)
                current_pedal = SustainPedal(control.time, None, control.value, control.number)
            elif control.value <= threshold:
                if isinstance(current_pedal, SustainPedal):
                    current_pedal.end = control.time
                    pedals.append(current_pedal)
                    current_pedal = None
                else:
                    warnings.warn('Sustain pedal offset detected without onset. Omitted')
    if isinstance(current_pedal, SustainPedal):
        warnings.warn('Last Sustain pedal detected without offset. Add offset at end')
        current_pedal.end = midi_obj.get_end_time()
        pedals.append(current_pedal)
    return pedals


def mid2piano_roll(midi_path, pedal=False, onset=False, midi_min=21, midi_max=108, clean_midi=True, fps=50.0):
    """Convert midi into piano-roll like array

    :param
        midi_path: midi path
        pedal: if True, elongate offset according to pedal
        onset: if True, mark only onset frame
        midi_min: minimum midi number to convert. belows will be ignored.
        midi_max: maximum midi number to convert. highers will be ignored.
        clean_midi: if True, clean up midi file before process.
        fps: frame rate per second. accept float values(ex: 36.6)
    :return:
        numpy array of piano roll, (midi_num, time_frames)
    """
    assert (pedal and onset) is not True, 'pedal + onset is not reasonable'

    if clean_midi:
        mid = to_midi_zero(midi_path, midi_min, midi_max)
    else:
        mid = pretty_midi.PrettyMIDI(midi_path)
    if pedal:
        mid = elongate_offset_by_pedal(mid)

    max_step = int(np.ceil(mid.get_end_time() * fps))
    dim = midi_max - midi_min + 1

    roll = np.zeros((max_step, dim))

    def time_to_frame(start, end):
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        return start_frame, end_frame

    if onset:
        for note in mid.instruments[0].notes:
            start_time = note.start
            end_time = np.min([start_time + ONSET_DURATION, note.end])
            start_frame, end_frame = time_to_frame(start_time, end_time)
            roll[start_frame: end_frame, note.pitch - midi_min] = 1
    else:
        for note in mid.instruments[0].notes:
            start_time = note.start
            end_time = note.end
            start_frame, end_frame = time_to_frame(start_time, end_time)
            roll[start_frame: end_frame, note.pitch - midi_min] = 1

    return roll


def piano_roll2chroma_roll(piano_roll):
    """Convert piano roll into chroma roll
    # TODO: fixed indexing, according to midi_min
    :param
        piano_roll: numpy array of shape (midi_num, time_frames)
    :return:
        chroma roll, numpy array of shape (12, time_frames)

    """

    chroma_roll = np.zeros((piano_roll.shape[0], 12))  # (time, class)
    for n in range(piano_roll.shape[1]):
        chroma_roll[:, n % 12] += piano_roll[:, n]
    chroma_roll = (chroma_roll >= 1).astype(np.int)
    return chroma_roll


def mid2chroma_roll(midi_path, pedal=False, onset=False):
    piano_roll = mid2piano_roll(midi_path, pedal=pedal, onset=onset)
    chroma_roll = piano_roll2chroma_roll(piano_roll)
    return chroma_roll


def binaryIndex(alist, item):
    first = 0
    last = len(alist)-1
    midpoint = 0

    if(item< alist[first]):
        return 0

    while first<last:
        midpoint = (first + last)//2
        currentElement = alist[midpoint]

        if currentElement < item:
            if alist[midpoint+1] > item:
                return midpoint
            else: first = midpoint +1;
            if first == last and alist[last] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint -1
        else:
            if midpoint +1 ==len(alist):
                return midpoint
            while alist[midpoint+1] == item:
                midpoint += 1
                if midpoint + 1 == len(alist):
                    return midpoint
            return midpoint
    return last


def save_note_pedal_to_CC(midi_obj, bool_pedal=False, disklavier=False):
    # input = pretty midi object with pedal inf embedded in note (e.g. note.pedal_at_start etc.)
    assert len(midi_obj.instruments) == 1
    instrument = midi_obj.instruments[0]
    notes = instrument.notes
    notes.sort(key=lambda note:note.start)
    num_notes = len(notes)
    eps = 0.03  # hyper-parameter

    def to_8(value):
        # if value == True:
        #     return 127
        # else:
        #     return 0
        return int(min(max(value,0),127))

    primary_pedal = []
    secondary_pedal = []
    for i in range(num_notes):
        note = notes[i]
        next_note = notes[min(i+1, num_notes-1)]
        # print(note.start, note.end, note.pitch, note.pedal_refresh, note.pedal_cut)
        pedal1 = pretty_midi.ControlChange(number=64, value=to_8(note.pedal_at_start), time=note.start-eps)
        pedal2 = pretty_midi.ControlChange(number=64, value=to_8(note.pedal_at_end), time=note.end-eps)
        soft_pedal = pretty_midi.ControlChange(number=67, value=to_8(note.soft_pedal), time=note.start-eps)

        instrument.control_changes.append(pedal1)
        instrument.control_changes.append(pedal2)
        instrument.control_changes.append(soft_pedal)
        primary_pedal.append(pedal1)
        primary_pedal.append(pedal2)

        # if note.pedal_refresh:
        refresh_time = note.start + note.pedal_refresh_time #(note.end - note.start) * note.pedal_refresh_time
        pedal3 = pretty_midi.ControlChange(number=64, value=to_8(note.pedal_refresh), time=refresh_time)
        if pedal3.time < pedal2.time:
            secondary_pedal.append(pedal3)
            instrument.control_changes.append(pedal3)
        #
        # if note.pedal_cut:
        # cut_time = note.start - (note.end - note.start) * note.pedal_cut_time
        cut_time = note.end + note.pedal_cut_time
        pedal4 = pretty_midi.ControlChange(number=64, value=to_8(note.pedal_cut), time=cut_time)
        instrument.control_changes.append(pedal4)
        secondary_pedal.append(pedal4)
    primary_pedal.sort(key=lambda x: x.time)



    last_note_end = notes[-1].end
    # end pedal 3 seconds after the last note
    last_pedal = pretty_midi.ControlChange(number=64, value=0, time=last_note_end + 3)
    instrument.control_changes.append(last_pedal)

    return midi_obj