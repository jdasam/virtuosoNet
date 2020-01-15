import pretty_midi
import copy

from . import xml_direction_encoding as dir_enc
from . import pedal_cleaning, utils

def xml_notes_to_midi(xml_notes):
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