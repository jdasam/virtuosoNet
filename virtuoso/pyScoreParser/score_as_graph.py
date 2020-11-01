def make_edge(xml_notes):
    num_notes = len(xml_notes)
    edge_list =[]

    for i in range(num_notes):
        note = xml_notes[i]
        note_position = note.note_duration.xml_position
        note_end_position = note_position + note.note_duration.duration
        note_end_include_rest = note_end_position + note.following_rest_duration
        current_voice = note.voice
        # current_pitch = note.pitch[1]
        slurs = note.note_notations.slurs
        current_slur_indexes = [slur.index for slur in slurs]
        if note.note_duration.duration == 0:  # grace note without tie
            current_grace_order = note.note_duration.grace_order
            for j in range(1, num_notes - i):
                next_note = xml_notes[i + j]
                next_note_start = next_note.note_duration.xml_position
                next_voice = next_note.voice
                next_note_slur_indexes = [slur.index for slur in next_note.note_notations.slurs]
                next_note_grace_order = next_note.note_duration.grace_order

                if next_note_start == note_position:
                    if current_grace_order == next_note_grace_order:  #same onset grace notes
                        edge_list.append((i, i + j, 'onset'))
                    elif current_voice == next_voice and current_grace_order+1 == next_note_grace_order:
                        in_same_slur = check_in_same_slur(current_slur_indexes, next_note_slur_indexes)
                        if in_same_slur:
                            edge_list.append((i, i + j, 'slur'))
                            edge_list.append((i, i + j, 'voice'))
                            edge_list.append((i, i + j, 'forward'))
                        else:
                            edge_list.append((i, i + j, 'voice'))
                            edge_list.append((i, i + j, 'forward'))
                if next_note_start > note_position:
                    break
        else: # ordinary note
            for j in range(1,num_notes-i):
                next_note = xml_notes[i+j]
                next_note_start = next_note.note_duration.xml_position
                # next_note_end = next_note_start + next_note.note_duration.duration
                next_voice = next_note.voice
                next_note_slur_indexes = [slur.index for slur in next_note.note_notations.slurs]
                if next_note_start == note_position and not next_note.note_duration.duration == 0:  #same onset
                    edge_list.append((i, i + j, 'onset'))
                elif next_note_start < note_end_position:
                    edge_list.append((i, i + j, 'melisma'))
                elif next_note_start == note_end_position and note_end_position == note_end_include_rest:
                    if next_note.note_duration.duration == 0:
                        edge_list.append((i, i + j, 'melisma'))
                    elif next_voice == current_voice:
                        in_same_slur = check_in_same_slur(current_slur_indexes, next_note_slur_indexes)
                        if in_same_slur:
                            edge_list.append((i, i + j, 'slur'))
                            edge_list.append((i, i + j, 'forward'))
                            edge_list.append((i, i + j, 'voice'))
                        else:
                            edge_list.append((i, i + j, 'voice'))
                            edge_list.append((i, i + j, 'forward'))
                    else:
                        edge_list.append((i, i + j, 'forward'))
                elif next_note_start < note_end_include_rest:
                    continue
                elif next_note_start == note_end_include_rest:
                    edge_list.append((i, i + j, 'rest'))
                else:
                    break

        # num_onset = len(same_onset_matrix[i])
        # onset_pitch_list = [xml_notes[same_onset_matrix[i][k]].pitch[1]for k in range(num_onset)]
        # num_next = len(forward_edge_matrix[i])
        # next_pitch_list = [{'pitch': xml_notes[forward_edge_matrix[i][k]].pitch[1], 'index':forward_edge_matrix[i][k]} for k in range(num_next)]
        # num_voice_next = len(voice_forward_matrix[i])
        # next_pitch_voice_list = [{'pitch': xml_notes[voice_forward_matrix[i][k]].pitch[1], 'index':voice_forward_matrix[i][k]} for k in range(num_voice_next)]
        # num_pedal_tone = len(pedal_tone_matrix[i])
        # onset_pitch_list.append(current_pitch)
        #
        # onset_pitch_list += [xml_notes[pedal_tone_matrix[i][k]].pitch[1] for k in range(num_pedal_tone)]
        # next_pitch_list += next_pitch_voice_list
        #
        # onset_pitch_list.sort()
        #
        # if len(next_pitch_list) == 0:
        #     num_notes_after_rest = len(rest_forward_matrix[i])
        #     next_pitch_list = [{'pitch': xml_notes[rest_forward_matrix[i][k]].pitch[1], 'index':rest_forward_matrix[i][k]} for k in range(num_notes_after_rest)]
        #
        # if len(next_pitch_list) != 0:
        #     next_pitch_list.sort(key=lambda x: x['pitch'])
        #     num_next_pitch = len(next_pitch_list)
        #     current_pitch_index = onset_pitch_list.index(current_pitch)
        #
        #     if current_pitch_index == 0: #lowest note
        #         next_pitch_index = next_pitch_list[0]['index']
        #         edge_list.append((i, next_pitch_index, 'boundary'))
        #         # boundary_pitch_forward[i].append(next_pitch_index)
        #         # boundary_pitch_backward[next_pitch_index].append(i)
        #     elif current_pitch_index == len(next_pitch_list)-1: #higest note
        #         next_pitch_index = next_pitch_list[-1]['index']
        #         edge_list.append((i, next_pitch_index, 'boundary'))
        #         # boundary_pitch_forward[i].append(next_pitch_index)
        #         # boundary_pitch_backward[next_pitch_index].append(i)
        #
        #     pitch_diff = [abs(current_pitch-next_pitch_list[k]['pitch']) for k in range(num_next_pitch)]
        #     min_pitch_diff = min(pitch_diff)
        #     min_diff_index = pitch_diff.index(min_pitch_diff)
        #     closest_pitch_forward[i].append(next_pitch_list[min_diff_index]['index'])
        #     closest_pitch_backward[next_pitch_list[min_diff_index]['index']].append(i)
        #     edge_list.append((i, next_pitch_list[min_diff_index]['index'], 'closest'))
        #     search_index = min_diff_index
        #     while search_index > 0:
        #         search_index -= 1
        #         if pitch_diff[search_index] == min_pitch_diff:
        #             edge_list.append((i, next_pitch_list[search_index]['index'], 'closest'))
        #             # closest_pitch_forward[i].append(next_pitch_list[min_diff_index]['index'])
        #             # closest_pitch_backward[next_pitch_list[min_diff_index]['index']].append(i)
        #         else:
        #             break
        #     search_index = min_diff_index
        #     while search_index < num_next_pitch -1:
        #         search_index += 1
        #         if pitch_diff[search_index] == min_pitch_diff:
        #             edge_list.append((i, next_pitch_list[search_index]['index'], 'closest'))
        #             # closest_pitch_forward[i].append(next_pitch_list[min_diff_index]['index'])
        #             # closest_pitch_backward[next_pitch_list[min_diff_index]['index']].append(i)
        #         else:
        #             break


    # total_edges = {'forward': forward_edge_matrix, 'backward': backward_edge_matrix, 'onset': same_onset_matrix,
    #                'melisma': melisma_note_matrix, 'pedal_tone': pedal_tone_matrix, 'rest_backward': rest_backward_matrix,
    #                'rest_forward': rest_forward_matrix, 'closest_forward': closest_pitch_backward, 'closest_backward': closest_pitch_backward,
    #                'boundary_forward': boundary_pitch_forward, 'boundary_backward': boundary_pitch_backward,
    #                'voice_forward': voice_forward_matrix, 'voice_backward':voice_backward_matrix}

    return edge_list


def check_in_same_slur(slursA, slursB):
    for slur in slursA:
        if slur in slursB:
            return True

    return False
