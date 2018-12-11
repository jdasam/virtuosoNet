import xml_matching
import musicxml_parser.mxp
import numpy as np



def make_edge(xml_notes):
    num_notes = len(xml_notes)
    edge_list =[]
    forward_edge_matrix = [ [] for i in range(num_notes) ]
    backward_edge_matrix =[ [] for i in range(num_notes) ]
    voice_forward_matrix = [ [] for i in range(num_notes) ]
    voice_backward_matrix = [ [] for i in range(num_notes) ]
    same_onset_matrix = [ [] for i in range(num_notes) ]
    melisma_note_matrix = [ [] for i in range(num_notes) ]
    pedal_tone_matrix = [ [] for i in range(num_notes) ]
    rest_forward_matrix = [ [] for i in range(num_notes) ]
    rest_backward_matrix = [ [] for i in range(num_notes) ]

    closest_pitch_forward = [ [] for i in range(num_notes) ]
    closest_pitch_backward = [ [] for i in range(num_notes) ]

    boundary_pitch_forward = [ [] for i in range(num_notes) ]
    boundary_pitch_backward = [ [] for i in range(num_notes) ]



    for i in range(num_notes):
        note = xml_notes[i]
        note_position = note.note_duration.xml_position
        note_end_position = note_position + note.note_duration.duration
        note_end_include_rest = note_end_position + note.following_rest_duration
        current_voice = note.voice
        current_pitch = note.pitch[1]
        for j in range(1,num_notes-i):
            next_note = xml_notes[i+j]
            next_note_start = next_note.note_duration.xml_position
            next_voice = next_note.voice
            if next_note_start == note_position:  #same onset
                edge_list.append((i, i + j, 'onset'))
                # same_onset_matrix[i].append(i+j)
                # same_onset_matrix[i+j].append(i)
            elif next_note_start < note_end_position:
                edge_list.append((i, i + j, 'melisma'))
                # melisma_note_matrix[i].append(i+j)
                # pedal_tone_matrix[i+j].append(i)
            elif next_note_start == note_end_position and note_end_position == note_end_include_rest:
                if next_voice == current_voice:
                    edge_list.append((i, i + j, 'voice'))
                    # voice_forward_matrix[i].append(i+j)
                    # voice_backward_matrix[i+j].append(i)
                else:
                    edge_list.append((i, i + j, 'forward'))
                    # forward_edge_matrix[i].append(i+j)
                    # backward_edge_matrix[i+j].append(i)
            elif next_note_start < note_end_include_rest:
                continue
            elif next_note_start == note_end_include_rest:
                edge_list.append((i, i + j, 'rest'))
                # rest_forward_matrix[i].append(i+j)
                # rest_backward_matrix[i+j].append(i)
            else:
                break

        num_onset = len(same_onset_matrix[i])
        onset_pitch_list = [xml_notes[same_onset_matrix[i][k]].pitch[1]for k in range(num_onset)]
        num_next = len(forward_edge_matrix[i])
        next_pitch_list = [{'pitch': xml_notes[forward_edge_matrix[i][k]].pitch[1], 'index':forward_edge_matrix[i][k]} for k in range(num_next)]
        num_voice_next = len(voice_forward_matrix[i])
        next_pitch_voice_list = [{'pitch': xml_notes[voice_forward_matrix[i][k]].pitch[1], 'index':voice_forward_matrix[i][k]} for k in range(num_voice_next)]
        num_pedal_tone = len(pedal_tone_matrix[i])
        onset_pitch_list.append(current_pitch)

        onset_pitch_list += [xml_notes[pedal_tone_matrix[i][k]].pitch[1] for k in range(num_pedal_tone)]
        next_pitch_list += next_pitch_voice_list

        onset_pitch_list.sort()

        if len(next_pitch_list) == 0:
            num_notes_after_rest = len(rest_forward_matrix[i])
            next_pitch_list = [{'pitch': xml_notes[rest_forward_matrix[i][k]].pitch[1], 'index':rest_forward_matrix[i][k]} for k in range(num_notes_after_rest)]

        if len(next_pitch_list) != 0:
            next_pitch_list.sort(key=lambda x: x['pitch'])
            num_next_pitch = len(next_pitch_list)
            current_pitch_index = onset_pitch_list.index(current_pitch)

            if current_pitch_index == 0: #lowest note
                next_pitch_index = next_pitch_list[0]['index']
                edge_list.append((i, next_pitch_index, 'boundary'))
                # boundary_pitch_forward[i].append(next_pitch_index)
                # boundary_pitch_backward[next_pitch_index].append(i)
            elif current_pitch_index == len(next_pitch_list)-1: #higest note
                next_pitch_index = next_pitch_list[-1]['index']
                edge_list.append((i, next_pitch_index, 'boundary'))
                # boundary_pitch_forward[i].append(next_pitch_index)
                # boundary_pitch_backward[next_pitch_index].append(i)

            pitch_diff = [abs(current_pitch-next_pitch_list[k]['pitch']) for k in range(num_next_pitch)]
            min_pitch_diff = min(pitch_diff)
            min_diff_index = pitch_diff.index(min_pitch_diff)
            closest_pitch_forward[i].append(next_pitch_list[min_diff_index]['index'])
            closest_pitch_backward[next_pitch_list[min_diff_index]['index']].append(i)
            search_index = min_diff_index
            while search_index > 0:
                search_index -= 1
                if pitch_diff[search_index] == min_pitch_diff:
                    edge_list.append((i, min_diff_index, 'closest'))
                    # closest_pitch_forward[i].append(next_pitch_list[min_diff_index]['index'])
                    # closest_pitch_backward[next_pitch_list[min_diff_index]['index']].append(i)
                else:
                    break
            search_index = min_diff_index
            while search_index < num_next_pitch -1:
                search_index += 1
                if pitch_diff[search_index] == min_pitch_diff:
                    edge_list.append((i, min_diff_index, 'closest'))
                    # closest_pitch_forward[i].append(next_pitch_list[min_diff_index]['index'])
                    # closest_pitch_backward[next_pitch_list[min_diff_index]['index']].append(i)
                else:
                    break


    # total_edges = {'forward': forward_edge_matrix, 'backward': backward_edge_matrix, 'onset': same_onset_matrix,
    #                'melisma': melisma_note_matrix, 'pedal_tone': pedal_tone_matrix, 'rest_backward': rest_backward_matrix,
    #                'rest_forward': rest_forward_matrix, 'closest_forward': closest_pitch_backward, 'closest_backward': closest_pitch_backward,
    #                'boundary_forward': boundary_pitch_forward, 'boundary_backward': boundary_pitch_backward,
    #                'voice_forward': voice_forward_matrix, 'voice_backward':voice_backward_matrix}

    return edge_list

