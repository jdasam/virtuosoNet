""" Utilities for xml usage

Interface summary:

        import xml_utils

        notes, pedals = xml_utils.xml_notes_to_midi(xml_notes)

call in feature_extraction.py and data_class.py

get xml information like note information to generate or modify xml information
"""
import pretty_midi
import copy

from . import xml_direction_encoding as dir_enc
from . import pedal_cleaning, utils

def xml_notes_to_midi(xml_notes):
    """ Returns midi-transformed xml notes in pretty_midi.Note() format

    Args:
        xml_notes (1-D list): list of Note() object in xml of shape (num_notes, )
    
    Returns:
        midi_notes (1-D list): list of pretty_midi.Note() of shape (num_notes, )
        midi_pedals (1-D list): list of pretty_midi pedal value of shape (num_pedals, )
    
    Example:
        (in data_class.py -> make_score_midi())
        >>> midi_notes, midi_pedals = xml_utils.xml_notes_to_midi(self.xml_notes)
        
    """
    midi_notes = []
    for note in xml_notes:
        if note.is_overlapped:  # ignore overlapped notes.
            continue

        pitch = note.pitch[1]
        start = note.note_duration.time_position
        end = start + note.note_duration.seconds
        if note.note_duration.seconds < 0.005:
            end = start + 0.005
        elif note.note_duration.seconds > 10:
            end = start + 10
        velocity = int(min(max(note.velocity,0),127))
        midi_note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)

        midi_notes.append(midi_note)
    midi_pedals = pedal_cleaning.predicted_pedals_to_midi_pedals(xml_notes)

    return midi_notes, midi_pedals


def find_tempo_change(xml_notes):
    """ Returns position of note where the tempo changes

    Args:
        xml_notes (1-D list): list of Note() object in xml of shape (num_notes, )
    
    Returns:
        tempo_change_positions (1-D list): list of xml_position shape (num of tempo_change_position, )
    
    Example:
        (in data_class.py -> _load_score_xml()
        >>> self.section_positions = xml_utils.find_tempo_change(self.xml_notes)
        
    """
    # TODO: This function can be simplified if it takes xml_obj or directions as the input
    tempo_change_positions = []
    previous_abs_tempo = None
    previous_rel_tempo = None
    for note in xml_notes:
        current_abs_tempo = note.tempo.absolute
        current_rel_tempo = note.tempo.relative
        if previous_abs_tempo != current_abs_tempo or previous_rel_tempo != current_rel_tempo:
            tempo_change_positions.append(note.note_duration.xml_position)
            previous_abs_tempo = current_abs_tempo
            previous_rel_tempo = current_rel_tempo

    tempo_change_positions.append(xml_notes[-1].note_duration.xml_position+0.1)
    return tempo_change_positions


'''
want to move to midi_utils
'''
def save_midi_notes_as_piano_midi(midi_notes, midi_pedals, output_name, bool_pedal=False, disklavier=False):
    """ Generate midi file by using received midi notes and midi pedals

    Args:
        midi_notes (1-D list) : list of pretty_midi.Note() of shape (num_notes, )
        midi_pedals (1-D list): list of pretty_midi pedal value of shape (num_pedals, ) 
        output_name (string) : output midi file name
        bool_pedal (boolean) : check whether the method needs to handle meaningless pedal values
        disklavier (boolean) : unused 

    Returns:
        -

    Example:
        (in data_class.py -> make_score_midi()
        >>> midi_notes, midi_pedals = xml_utils.xml_notes_to_midi(self.xml_notes)
        >>> xml_utils.save_midi_notes_as_piano_midi(midi_notes, [], midi_file_name, bool_pedal=True)

    """
    piano_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    # pedal_threhsold = 60
    # pedal_time_margin = 0.2

    for note in midi_notes:
        piano.notes.append(note)

    piano_midi.instruments.append(piano)

    # piano_midi = midi_utils.save_note_pedal_to_CC(piano_midi)

    if bool_pedal:
        for pedal in midi_pedals:
            if pedal.value < pedal_cleaning.THRESHOLD:
                pedal.value = 0

    last_note_end = midi_notes[-1].end
    # end pedal 3 seconds after the last note
    last_pedal = pretty_midi.ControlChange(number=64, value=0, time=last_note_end + 3)
    midi_pedals.append(last_pedal)

    piano_midi.instruments[0].control_changes = midi_pedals

    #
    # if disklavier:
    #     pedals = piano_midi.instruments[0].control_changes
    #     pedals.sort(key=lambda x:x.time)
    #     previous_off_time = 0
    #
    #     prev_high_time = 0
    #     prev_low_time = 0
    #
    #     pedal_remove_candidate = []
    #     for pedal in pedals:
    #         if pedal.number == 67:
    #             continue
    #         if pedal.time < 0.2:
    #             continue
    #         if pedal.value < pedal_threhsold:
    #             previous_off_time = pedal.time
    #         else:
    #             time_passed = pedal.time - previous_off_time
    #             if time_passed < pedal_time_margin:  #hyperparameter
    #                 # pedal.time = previous_off_time + pedal_time_margin
    #                 pedal.value = 30
    #
    #         if pedal.value > 75:
    #             if pedal.time - prev_high_time < 0.25:
    #                 pedal_remove_candidate.append(pedal)
    #             else:
    #                 prev_high_time = pedal.time
    #         if pedal.value < 55:
    #             if pedal.time - prev_low_time < 0.25:
    #                 pedal_remove_candidate.append(pedal)
    #             else:
    #                 prev_low_time = pedal.time
    #
    #     for pedal in pedal_remove_candidate:
    #         pedals.remove(pedal)

    piano_midi.write(output_name)


def apply_directions_to_notes(xml_notes, directions, time_signatures):
    """ apply xml directions to each xml_notes

    Args:
        xml_notes (1-D list): list of Note() object in xml of shape (num_notes, )
        directions (1-D list): list of Direction() object in xml of shape (num_direction, )
        time_signatures (1-D list): list of TimeSignature() object in xml of shape (num_time_signature, )

    Returns:
        xml_notes (1-D list): list of direction-encoded Note() object in xml of shape (num_notes, )

    Example:
        (in data_class.py -> _get_direction_encoded_notes())
        >>> notes, rests = self.xml_obj.get_notes()
        >>> directions = self.xml_obj.get_directions()
        >>> time_signatures = self.xml_obj.get_time_signatures()
        >>> self.xml_notes = xml_utils.apply_directions_to_notes(notes, directions, time_signatures)
        >>> self.num_notes = len(self.xml_notes)
    """
    absolute_dynamics, relative_dynamics, cresciutos = dir_enc.get_dynamics(directions)
    absolute_dynamics_position = [dyn.xml_position for dyn in absolute_dynamics]
    absolute_tempos, relative_tempos = dir_enc.get_tempos(directions)
    absolute_tempos_position = [tmp.xml_position for tmp in absolute_tempos]
    time_signatures_position = [time.xml_position for time in time_signatures]

    num_dynamics = len(absolute_dynamics)
    num_tempos = len(absolute_tempos)

    for note in xml_notes:
        note_position = note.note_duration.xml_position

        if num_dynamics > 0:
            index = utils.binary_index(absolute_dynamics_position, note_position)
            note.dynamic.absolute = absolute_dynamics[index].type['content']
            note.dynamic.absolute_position = absolute_dynamics[index].xml_position

        if num_tempos > 0:
            tempo_index = utils.binary_index(absolute_tempos_position, note_position)
        # note.tempo.absolute = absolute_tempos[tempo_index].type[absolute_tempos[tempo_index].type.keys()[0]]
            note.tempo.absolute = absolute_tempos[tempo_index].type['content']
            note.tempo.recently_changed_position = absolute_tempos[tempo_index].xml_position
        time_index = utils.binary_index(time_signatures_position, note_position)
        note.tempo.time_numerator = time_signatures[time_index].numerator
        note.tempo.time_denominator = time_signatures[time_index].denominator
        note.tempo.time_signature = time_signatures[time_index]

        # have to improve algorithm
        for rel in relative_dynamics:
            if rel.xml_position > note_position:
                continue
            if note_position < rel.end_xml_position:
                note.dynamic.relative.append(rel)
                if rel.xml_position > note.tempo.recently_changed_position:
                    note.tempo.recently_changed_position = rel.xml_position

        for cresc in cresciutos:
            if cresc.xml_position > note_position:
                break
            if note_position < cresc.end_xml_position:
                note_cresciuto = note.dynamic.cresciuto
                if note_cresciuto is None:
                    note.dynamic.cresciuto = copy.copy(cresc)
                else:
                    prev_type = note.dynamic.cresciuto.type
                    if cresc.type == prev_type:
                        note.dynamic.cresciuto.overlapped += 1
                    else:
                        if note_cresciuto.overlapped == 0:
                            note.dynamic.cresciuto = None
                        else:
                            note.dynamic.cresciuto.overlapped -= 1

        if len(note.dynamic.relative) >1:
            note = divide_cresc_staff(note)

        for rel in relative_tempos:
            if rel.xml_position > note_position:
                continue
            if note_position < rel.end_xml_position:
                note.tempo.relative.append(rel)

    return xml_notes

def divide_cresc_staff(note):
    """ update note.dynamic

    Args:
        note: Note() object in xml_notes

    Returns:
        note: dynamic updated Note() object in xml_notes

    Example:
        (in apply_directions_to_notes())
        >>> note = divide_cresc_staff(note)
    """
    #check the note has both crescendo and diminuendo (only wedge type)
    cresc = False
    dim = False
    for rel in note.dynamic.relative:
        if rel.type['type'] == 'crescendo':
            cresc = True
        elif rel.type['type'] == 'diminuendo':
            dim = True

    if cresc and dim:
        delete_list = []
        for i in range(len(note.dynamic.relative)):
            rel = note.dynamic.relative[i]
            if rel.type['type'] in ['crescendo', 'diminuendo']:
                if (rel.placement == 'above' and note.staff ==2) or (rel.placement == 'below' and note.staff ==1):
                    delete_list.append(i)
        for i in sorted(delete_list, reverse=True):
            del note.dynamic.relative[i]

    return note

def cal_total_xml_length(xml_notes):
    """ Return proper length of total xml notes

    Args:
        xml_notes (1-D list): list of Note() object in xml of shape (num_notes, )

    Returns:
        lates_end (integer) : proper length of total xml notes

    Example:
        (in feature_extraction.py -> _get_cresciuto())
        >>> total_length = xml_utils.cal_total_xml_length(piece_data.xml_notes)

    """
    latest_end = 0
    latest_start =0
    xml_len = len(xml_notes)
    for i in range(1,xml_len):
        note = xml_notes[-i]
        current_end = note.note_duration.xml_position + note.note_duration.duration
        if current_end > latest_end:
            latest_end = current_end
            latest_start = note.note_duration.xml_position
        elif current_end < latest_start - note.note_duration.duration * 4:
            break
    return latest_end


def check_corresponding_accidental(note, accidentals):
    """ Return a key correspond to the type of accidentals on note position

    Args:
        note (Note) : Note() object in xml_notes
        accidentals (1-D list) : list of accidentals in Note() objects

    Returns:
        final_key (integer) : final key corresponds to accidental type

    Example:
        (in find_corresp_trill_notes_from_midi())
        >>> accidentals = piece_data.xml_obj.get_accidentals()
        >>> accidental_on_trill = check_corresponding_accidental(note, accidentals)
    """
    final_key = None
    for acc in accidentals:
        if acc.xml_position == note.note_duration.xml_position:
            if acc.type['content'] == '#':
                final_key = 1
                break
            elif acc.type['content'] == '♭':
                final_key = -1
                break
            elif acc.type['content'] == '♮':
                final_key = 0
                break
    return final_key


def find_corresp_trill_notes_from_midi(piece_data, perform_data, index):
    """ Return information about trill(length of trill and trill parameter) which starts from current note

    Args: 
        piece_data (PieceData object) : PieceData object
        perform_data (PerformData object) : PerformData object
        index (integer) : index of current note pair

    Returns:
        trill_vec (1-D list) : list of [num_trills, last_note_velocity, first_note_ratio, last_note_ratio, up_trill] shape (5, )
        trill_length (integer) : length of current trill

    Example1:
        (in feature_extraction.py -> get_articulation())
        >>> if note.note_notations.is_trill:
        >>>     _, actual_second = xml_utils.find_corresp_trill_notes_from_midi(piece_data, perform_data, i)
        >>> else:
        >>>     actual_second = midi.end - midi.start
                
    Example2:
        (in feature_extraction.py -> get_trill_parameters())
        >>> if note.note_notations.is_trill:
        >>>     trill_parameter, _ = xml_utils.find_corresp_trill_notes_from_midi(piece_data, perform_data, i)
        >>> else:
        >>>     trill_parameter = [0, 0, 0, 0, 0]
    """
    #start of trill, end of trill
    key_signatures = piece_data.xml_obj.get_key_signatures()
    note = piece_data.xml_notes[index]
    num_note = piece_data.num_notes
    accidentals = piece_data.xml_obj.get_accidentals()

    key = utils.get_item_by_xml_position(key_signatures, note)
    key = key.key

    accidental_on_trill = check_corresponding_accidental(note, accidentals)
    measure_accidentals = get_measure_accidentals(piece_data.xml_notes, index)
    trill_pitch = note.pitch[1]

    up_pitch, _ = cal_up_trill_pitch(note.pitch, key, accidental_on_trill, measure_accidentals)

    prev_search = 1
    prev_idx = index
    next_idx = index
    while index - prev_search >= 0:
        prev_idx = index - prev_search
        prev_note = piece_data.xml_notes[prev_idx]
        # if prev_note.voice == note.voice: #and not pairs[prev_idx] == []:
        if prev_note.note_duration.xml_position < note.note_duration.xml_position:
            break
        elif prev_note.note_duration.xml_position == note.note_duration.xml_position and \
            prev_note.note_duration.grace_order < note.note_duration.grace_order:
            break
        else:
            prev_search += 1

    next_search = 1
    trill_end_position = note.note_duration.xml_position + note.note_duration.duration
    while index + next_search < num_note:
        next_idx = index + next_search
        next_note = piece_data.xml_notes[next_idx]
        if next_note.note_duration.xml_position > trill_end_position:
            break
        else:
            next_search += 1

    skipped_pitches_start = []
    skipped_pitches_end = []
    while perform_data.pairs[prev_idx] == [] and prev_idx >0:
        skipped_note = piece_data.xml_notes[prev_idx]
        skipped_pitches_start.append(skipped_note.pitch[1])
        prev_idx -= 1
    while perform_data.pairs[next_idx] == [] and next_idx < num_note -1:
        skipped_note = piece_data.xml_notes[next_idx]
        skipped_pitches_end.append(skipped_note.pitch[1])
        next_idx += 1

    if perform_data.pairs[next_idx] == [] or perform_data.pairs[prev_idx] == []:
        print("Error: Cannot find trill start or end note")
        return [0] * 5, 0
    prev_midi_note = perform_data.pairs[prev_idx]['midi']
    next_midi_note = perform_data.pairs[next_idx]['midi']
    search_range_start = perform_data.midi_notes.index(prev_midi_note)
    search_range_end = perform_data.midi_notes.index(next_midi_note)
    trills = []
    prev_pitch = None
    if len(skipped_pitches_end) > 0:
        end_cue = skipped_pitches_end[0]
    else:
        end_cue = None
    for i in range(search_range_start+1, search_range_end):
        midi_note = perform_data.midi_notes[i]
        cur_pitch = midi_note.pitch
        if cur_pitch in skipped_pitches_start:
            skipped_pitches_start.remove(cur_pitch)
            continue
        elif cur_pitch == trill_pitch or cur_pitch == up_pitch:
            # if cur_pitch == prev_pitch:
            #     next_midi_note = midi_note
            #     break
            # else:
            trills.append(midi_note)
            prev_pitch = cur_pitch
        elif cur_pitch == end_cue:
            next_midi_note = midi_note
            break
        elif 0 < midi_note.pitch - trill_pitch < 4:
            print('check trill pitch - detected pitch: ', midi_note.pitch, ' trill note pitch: ', trill_pitch,
                  'expected up trill pitch: ', up_pitch, 'time:', midi_note.start, 'measure: ', note.measure_number)

    # while len(skipped_pitches_start) > 0:
    #     skipped_pitch = skipped_pitches_start[0]
    #     if trills[0].pitch == skipped_pitch:
    #         dup_note = trills[0]
    #         trills.remove(dup_note)
    #     skipped_pitches_start.remove(skipped_pitch)
    #
    # while len(skipped_pitches_end) > 0:
    #     skipped_pitch = skipped_pitches_end[0]
    #     if trills[-1].pitch == skipped_pitch:
    #         dup_note = trills[-1]
    #         trills.remove(dup_note)
    #     skipped_pitches_end.remove(skipped_pitch)


    trills_vec = [0] * 5 # num_trills, last_note_velocity, first_note_ratio, last_note_ratio, up_trill
    num_trills = len(trills)

    if num_trills == 0:
        return trills_vec, 0
    elif num_trills == 1:
        return trills_vec, 1
    elif num_trills > 2 and trills[-1].pitch == trills[-2].pitch:
        del trills[-1]
        num_trills -= 1

    if trills[0].pitch == up_pitch:
        up_trill = True
    else:
        up_trill = False

    ioi_seconds = []
    prev_start = trills[0].start
    next_note_start = next_midi_note.start
    trill_length = next_note_start - prev_start
    for i in range(1, num_trills):
        ioi = trills[i].start - prev_start
        ioi *= (num_trills / trill_length)
        ioi_seconds.append(ioi)
        prev_start = trills[i].start
    ioi_seconds.append( (next_note_start - trills[-1].start) *  num_trills / trill_length )
    trill_density = num_trills / trill_length

    if trill_density < 1:
        return trills_vec, trill_length


    trills_vec[0] = num_trills / trill_length
    trills_vec[1] = trills[-1].velocity / trills[0].velocity
    trills_vec[2] = ioi_seconds[0]
    trills_vec[3] = ioi_seconds[-1]
    trills_vec[4] = int(up_trill)


    if perform_data.pairs[index] == []:
        for pair in perform_data.pairs:
            if not pair == [] and pair['midi'] == trills[0]:
                pair = []
        perform_data.pairs[index] = {'xml': note, 'midi': trills[0]}

    # for num in trills_vec:
    #     if math.isnan(num):
    #         print('trill vec is nan')
    #         num = 0

    return trills_vec, trill_length


def get_measure_accidentals(xml_notes, index):
    """ Return pitch-accidental pairs list for current measure

    Args: 
        xml_notes (1-D list): list of Note() object in xml of shape (num_notes, )
        index (integer) : index of current note pair

    Returns:
        measure_accidentals (1-D list): list of {pitch, accident} dictionary pair

    Example1:
        (in find_corresp_trill_notes_from_midi())
        >>> measure_accidentals = get_measure_accidentals(piece_data.xml_notes, index)
    """
    accs = ['bb', 'b', '♮', '#', 'x']
    note = xml_notes[index]
    num_note = len(xml_notes)
    measure_accidentals=[]
    for i in range(1,num_note):
        prev_note = xml_notes[index - i]
        if prev_note.measure_number != note.measure_number:
            break
        else:
            for acc in accs:
                if acc in prev_note.pitch[0]:
                    pitch = prev_note.pitch[0][0] + prev_note.pitch[0][-1]
                    for prev_acc in measure_accidentals:
                        if prev_acc['pitch'] == pitch:
                            break
                    else:
                        accident = accs.index(acc) - 2
                        temp_pair = {'pitch': pitch, 'accident': accident}
                        measure_accidentals.append(temp_pair)
                        break
    return measure_accidentals


def cal_up_trill_pitch(pitch_tuple, key, final_key, measure_accidentals):
    """ Return 

    Args: 
        pitch_tuple (tuple) : pitch information in (Pitch Name, MIDI number) format
        key (integer) : MIDI and MusicXML identify key by using "fifths"
              -1 = F, 0 = C, 1 = G etc.
        final_key (integer) : final key delta by check_corresponding_accidental()
        measure_accidentals (1-D list): list of {pitch, accident} dictionary pair

    Returns:
        up_pitch (integer) : final pitch value
        final_pitch_string (string) : final pitch value in string format

    Example1:
        (in find_corresp_trill_notes_from_midi())
        >>> up_pitch, _ = cal_up_trill_pitch(note.pitch, key, accidental_on_trill, measure_accidentals)

    """
    pitches = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    corresp_midi_pitch = [0, 2, 4, 5, 7, 9, 11]
    pitch_name = pitch_tuple[0][0:1].lower()
    octave = int(pitch_tuple[0][-1])
    next_pitch_name = pitches[(pitches.index(pitch_name)+1)%7]
    if next_pitch_name == 'c':
        octave += 1

    if final_key:
        acc = final_key
    else:
        accidentals = ['f', 'c', 'g', 'd', 'a', 'e', 'b']
        if key > 0 and next_pitch_name in accidentals[:key]:
            acc = +1
        elif key < 0 and next_pitch_name in accidentals[key:]:
            acc = -1
        else:
            acc = 0

    pitch_string = next_pitch_name + str(octave)
    for pitch_acc_pair in measure_accidentals:
        if pitch_string == pitch_acc_pair['pitch'].lower():
            acc = pitch_acc_pair['accident']

    if not final_key == None:
        acc= final_key

    if acc == 0:
        acc_in_string = ''
    elif acc ==1:
        acc_in_string = '#'
    elif acc ==-1:
        acc_in_string = '♭'
    else:
        acc_in_string = ''
    final_pitch_string = next_pitch_name.capitalize() + acc_in_string + str(octave)
    up_pitch = 12 * (octave + 1) + corresp_midi_pitch[pitches.index(next_pitch_name)] + acc

    return up_pitch, final_pitch_string
