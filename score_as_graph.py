import xml_matching
import musicxml_parser.mxp
import numpy as np



def make_edge(xml_notes):
    num_notes = len(xml_notes)
    forward_edge_matrix = [ [] for i in range(num_notes) ]
    backward_edge_matrix =[ [] for i in range(num_notes) ]
    same_onset_matrix = [ [] for i in range(num_notes) ]
    melisma_note_matrix = [ [] for i in range(num_notes) ]
    pedal_tone_matrix = [ [] for i in range(num_notes) ]
    rest_forward_matrix = [ [] for i in range(num_notes) ]
    rest_backward_matrix = [ [] for i in range(num_notes) ]


    for i in range(num_notes):
        note = xml_notes[i]
        note_position = note.note_duration.xml_position
        note_end_position = note_position + note.note_duration.duration
        note_end_include_rest = note_end_position + note.following_rest_duration
        for j in range(1,num_notes-i):
            next_note = xml_notes[i+j]
            next_note_start = next_note.note_duration.xml_position
            if next_note_start == note_position:  #same onset
                same_onset_matrix[i].append(i+j)
                same_onset_matrix[i+j].append(i)
            elif next_note_start < note_end_position:
                melisma_note_matrix[i].append(i+j)
                pedal_tone_matrix[i+j].append(i)
            elif next_note_start == note_end_position:
                forward_edge_matrix[i].append(i+j)
                backward_edge_matrix[i+j].append(i)
            elif next_note_start < note_end_include_rest:
                continue
            elif next_note_start == note_end_include_rest:
                rest_forward_matrix[i].append(i+j)
                rest_backward_matrix[i+j].append(i)
            else:
                break

    total_edges = {'forward': forward_edge_matrix, 'backward': backward_edge_matrix, 'onset': same_onset_matrix,
                   'melisma': melisma_note_matrix, 'pedal_tone': pedal_tone_matrix, 'rest_backward': rest_backward_matrix,
                   'rest_forward': rest_forward_matrix}
    return total_edges