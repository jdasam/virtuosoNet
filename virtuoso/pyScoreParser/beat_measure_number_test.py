import xml_matching
import os
from .musicxml_parser import MusicXMLDocument
from .midi_utils.midi_utils import midi_utils
import pretty_midi
import copy
import math
import numpy as np


def load_entire_subfolder(path):
    entire_pairs = []
    num_train_pairs = 0
    midi_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              f == 'midi_cleaned.mid']
    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        xml_name = foldername + 'musicxml_cleaned.musicxml'

        if os.path.isfile(xml_name):
            print(foldername)
            piece_pairs = load_pairs_from_folder(foldername)
            if piece_pairs is not None:
                entire_pairs.append(piece_pairs)

    return entire_pairs


def load_pairs_from_folder(path):
    xml_name = path+'musicxml_cleaned.musicxml'
    score_midi_name = path+'midi_cleaned.mid'
    composer_name = copy.copy(path).split('/')[1]
    composer_name_vec = xml_matching.composer_name_to_vec(composer_name)

    XMLDocument = MusicXMLDocument(xml_name)
    xml_notes = xml_matching.extract_notes(XMLDocument, melody_only=False, grace_note=True)
    score_midi = midi_utils.to_midi_zero(score_midi_name)
    score_midi_notes = score_midi.instruments[0].notes
    score_midi_notes.sort(key=lambda x:x.start)
    match_list = xml_matching.matchXMLtoMIDI(xml_notes, score_midi_notes)
    score_pairs = xml_matching.make_xml_midi_pair(xml_notes, score_midi_notes, match_list)
    num_non_matched = xml_matching.check_pairs(score_pairs)
    if num_non_matched > 100:
        print("There are too many xml-midi matching errors")
        return None

    measure_positions = xml_matching.extract_measure_position(XMLDocument)
    filenames = os.listdir(path)
    perform_features_piece = []
    directions, time_signatures = xml_matching.extract_directions(XMLDocument)
    xml_notes = xml_matching.apply_directions_to_notes(xml_notes, directions, time_signatures)
    measure_positions = xml_matching.extract_measure_position(XMLDocument)
    beats = xml_matching.cal_beat_positions_of_piece(XMLDocument)
    features = xml_matching.extract_score_features(xml_notes, measure_positions, beats)
    features = xml_matching.make_index_continuous(features, score=True)


    num_notes = len(features)
    num_total_batch = int(math.ceil(num_notes / 300))
    onset_numbers = [feat.note_location.onset for feat in features]
    beat_numbers = [feat.note_location.beat for feat in features]
    measure_numbers = [feat.note_location.measure for feat in features]

    for i in range(num_total_batch):
        if i < num_total_batch -1:
            start_index = i * 300
        else:
            start_index = num_notes - 300

        dummy_lower_nodes = np.arange(300)
        onset_nodes = mimic_making_onset_node(dummy_lower_nodes, onset_numbers, start_index)
        beat_nodes = mimic_making_higher_node(onset_nodes, onset_numbers, beat_numbers, start_index)
        measure_nodes = mimic_making_higher_node(beat_nodes, beat_numbers, measure_numbers, start_index)

        num_generated_onset = len(onset_nodes)
        num_generated_beat = len(beat_nodes)
        num_generated_measure = len(measure_nodes)

        num_required_onset = onset_numbers[start_index+300-1] - onset_numbers[start_index] + 1
        num_required_beat = beat_numbers[start_index+300-1] - beat_numbers[start_index] + 1
        num_required_measure = measure_numbers[start_index+300-1] - measure_numbers[start_index] + 1

        if num_generated_onset != num_required_onset:
            print('num gen onset', num_generated_onset, num_required_onset)
        if num_generated_beat != num_required_beat:
            print('num gen beat', num_generated_beat, num_required_beat)
        if num_generated_measure != num_required_measure:
            print('num gen measure', num_generated_measure, num_required_measure, start_index, measure_numbers[start_index], measure_numbers[start_index+299])


    return perform_features_piece


def mimic_making_higher_node(lower_out, lower_indexes, higher_indexes, start_index):
    higher_nodes = []
    prev_higher_index = higher_indexes[start_index]
    lower_node_start = 0
    lower_node_end = 0
    num_lower_nodes = len(lower_out)
    start_lower_index = lower_indexes[start_index]
    for low_index in range(num_lower_nodes):
        absolute_low_index = start_lower_index + low_index
        current_note_index = lower_indexes.index(absolute_low_index)

        if higher_indexes[current_note_index] > prev_higher_index:
            # new beat start
            lower_node_end = low_index
            corresp_lower_out = lower_out[lower_node_start:low_index]
            higher = np.mean(corresp_lower_out)
            higher_nodes.append(higher)

            lower_node_start = low_index
            prev_higher_index = higher_indexes[current_note_index]

    corresp_lower_out = lower_out[lower_node_start:]
    higher = np.mean(corresp_lower_out)
    higher_nodes.append(higher)

    return higher_nodes


def mimic_making_onset_node(input_notes, onset_numbers, start_index):
    num_notes = len(input_notes)
    onset_nodes = []
    onset_notes_start = 0
    prev_onset = onset_numbers[start_index]
    for note_index in range(num_notes):
        abs_index = start_index + note_index
        if onset_numbers[abs_index] > prev_onset:
            # new beat start or sequence ends
            onset_notes_end = note_index
            corresp_notes = input_notes[onset_notes_start:onset_notes_end]
            onset = np.mean(corresp_notes)
            onset_nodes.append(onset)

            onset_notes_start = note_index
            prev_onset = onset_numbers[abs_index]

    corresp_notes = input_notes[onset_notes_start:]
    onset = np.mean(corresp_notes)
    onset_nodes.append(onset)

    return onset_nodes

# load_entire_subfolder('chopin_cleaned/')
load_entire_subfolder('chopin_cleaned/Schumann/Kreisleriana/')