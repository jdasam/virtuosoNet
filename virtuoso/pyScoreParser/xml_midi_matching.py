import csv
import copy


def match_xml_to_midi(xml_notes, midi_notes):
    candidates_list = []
    match_list = []
    # midi_positions = [note.start for note in midi_notes]
    # def find_candidate_list(xml_note, midi_notes, midi_positions):
    #     num_midi = len(midi_notes)
    #     temp_list =[]
    #     match_threshold = 0.1
    #     if note.is_rest:
    #         return([])
    #     note_start = xml_note.note_duration.time_position
    #     if note.note_duration.preceded_by_grace_note:
    #         note_start += 0.5
    #         match_threshold = 0.6
    #     elif note.note_notations.is_arpeggiate:
    #         note_start += 0.3
    #         match_threshold = 0.4
    #
    #     nearby_index = binaryIndex(midi_positions, note_start)
    #
    #     for i in range(-10, 10):
    #         index = nearby_index+i
    #         if index < 0:
    #             index = 0
    #         elif index >= num_midi:
    #             break
    #         midi_note = midi_notes[index]
    #         if midi_note.pitch == note.pitch[1] or abs(midi_note.start - note_start) < match_threshold:
    #             temp_list.append({'index': index, 'midi_note':midi_note})
    #
    #         if midi_note.start > note_start + match_threshold:
    #             break
    #
    #     return temp_list

    # for each note in xml, make candidates of the matching midi note
    for note in xml_notes:
        match_threshold = 0.1
        if note.is_rest:
            candidates_list.append([])
            continue
        note_start = note.note_duration.time_position
        if note.note_duration.preceded_by_grace_note:
            note_start += 0.5
            match_threshold = 0.6
        elif note.note_notations.is_arpeggiate:
            match_threshold = 0.5
        # check grace note and adjust time_position
        note_pitch = note.pitch[1]
        temp_list = [{'index': index, 'midi_note': midi_note} for index, midi_note in enumerate(midi_notes)
                     if abs(midi_note.start - note_start) < match_threshold and midi_note.pitch == note_pitch]
        # temp_list = find_candidate_list(note, midi_notes, midi_positions)
        candidates_list.append(temp_list)


    for candidates in candidates_list:
        if len(candidates) ==1:
            matched_index = candidates[0]['index']
            match_list.append(matched_index)
        elif len(candidates) > 1:
            added = False
            for cand in candidates:
                if cand['index'] not in match_list:
                    match_list.append(cand['index'])
                    added = True
                    break
            if not added:
                match_list.append([])
        else:
            match_list.append([])
    return match_list


def make_xml_midi_pair(xml_notes, midi_notes, match_list):
    pairs = [{'xml': xml_notes[i], 'midi': midi_notes[match_list[i]]} if match_list[i] != [] else [] for i in range(len(match_list))  ]
    # for i in range(len(match_list)):
    #     if not match_list[i] ==[]:
    #         temp_pair = {'xml': xml_notes[i], 'midi': midi_notes[match_list[i]]}
    #         pairs.append(temp_pair)
    #     else:
    #         pairs.append([])
    return pairs


def read_corresp(txtpath):
    file = open(txtpath, 'r')
    reader = csv.reader(file, dialect='excel', delimiter='\t')
    corresp_list = []
    for row in reader:
        if len(row) == 1:
            continue
        temp_dic = {'alignID': row[0], 'alignOntime': row[1], 'alignSitch': row[2], 'alignPitch': row[3], 'alignOnvel': row[4], 'refID':row[5], 'refOntime':row[6], 'refSitch':row[7], 'refPitch':row[8], 'refOnvel':row[9] }
        corresp_list.append(temp_dic)

    return corresp_list


def match_score_pair2perform(pairs, perform_midi, corresp_list):
    match_list = []
    for pair in pairs:
        if pair == []:
            match_list.append([])
            continue
        ref_midi = pair['midi']
        index_in_corresp = find_by_key(corresp_list, 'refOntime', ref_midi.start, 'refPitch', ref_midi.pitch)
        if index_in_corresp == -1:
            match_list.append([])
        else:
            corresp_pair = corresp_list[index_in_corresp]
            index_in_perform_midi = find_by_attr(perform_midi, float(corresp_pair['alignOntime']),  int(corresp_pair['alignPitch']))
            # if index_in_perform_midi == []:
            #     print('perf midi missing: ', corresp_pair, ref_midi.start, ref_midi.pitch)
            match_list.append(index_in_perform_midi)
    return match_list


def match_xml_midi_perform(xml_notes, midi_notes, perform_notes, corresp):
    # xml_notes = apply_tied_notes(xml_notes)
    match_list = match_xml_to_midi(xml_notes, midi_notes)
    score_pairs = make_xml_midi_pair(xml_notes, midi_notes, match_list)
    xml_perform_match = match_score_pair2perform(score_pairs, perform_notes, corresp)
    perform_pairs = make_xml_midi_pair(xml_notes, perform_notes, xml_perform_match)

    return score_pairs, perform_pairs


def find_by_key(alist, key1, value1, key2, value2):
    for i, dic in enumerate(alist):
        if abs(float(dic[key1]) - value1) < 0.02 and int(dic[key2]) == value2:
            return i
    return -1


def find_by_attr(alist, value1, value2):
    for i, obj in enumerate(alist):
        if abs(obj.start - value1) < 0.02 and obj.pitch == value2:
            return i
    return []


def make_available_xml_midi_positions(pairs):
    # global NUM_EXCLUDED_NOTES
    available_pairs = []
    for i, pair in enumerate(pairs):
        if not pair == []:
            xml_note = pair['xml']
            midi_note = pair['midi']
            xml_pos = xml_note.note_duration.xml_position
            time = midi_note.start
            divisions = xml_note.state_fixed.divisions
            if not xml_note.note_duration.is_grace_note:
                pos_pair = {'xml_position': xml_pos, 'time_position': time, 'pitch': xml_note.pitch[1], 'index':i, 'divisions':divisions}
                # pos_pair = PositionPair(xml_pos, time, xml_note.pitch[1], i, divisions)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair['is_arpeggiate'] = True
                else:
                    pos_pair['is_arpeggiate'] = False
                available_pairs.append(pos_pair)

    # available_pairs = save_lowest_note_on_same_position(available_pairs)
    available_pairs, mismatched_indexes = make_average_onset_cleaned_pair(available_pairs)
    print('Number of mismatched notes: ', len(mismatched_indexes))
    # NUM_EXCLUDED_NOTES += len(mismatched_indexes)
    # for index in mismatched_indexes:
    #     pairs[index] = []

    return available_pairs, mismatched_indexes


def make_average_onset_cleaned_pair(position_pairs, maximum_qpm=600):
    length = len(position_pairs)
    previous_position = -float("Inf")
    previous_time = -float("Inf")
    previous_index = 0
    # position_pairs.sort(key=lambda x: (x.xml_position, x.pitch))
    cleaned_list = list()
    notes_in_chord = list()
    mismatched_indexes = list()
    for i in range(length):
        pos_pair = position_pairs[i]
        current_position = pos_pair['xml_position']
        current_time = pos_pair['time_position']
        if current_position > previous_position >= 0:
            minimum_time_interval = (current_position - previous_position) / pos_pair['divisions'] / maximum_qpm * 60 + 0.01
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair['time_position'] = get_average_onset_time(notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair['time_position'] > cleaned_list[-1]['time_position'] + (
                        (average_pos_pair['xml_position'] - cleaned_list[-1]['xml_position']) /
                        average_pos_pair['divisions'] / maximum_qpm * 60 + 0.01):
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            # print('the note is far from other notes in the chord')
                            mismatched_indexes.append(note['index'])
                else:
                    # print('the onset is too close to the previous onset', average_pos_pair.xml_position, cleaned_list[-1].xml_position, average_pos_pair.time_position, cleaned_list[-1].time_position)
                    for note in notes_in_chord:
                        mismatched_indexes.append(note['index'])
            notes_in_chord = list()
            notes_in_chord.append(pos_pair)
            previous_position = current_position
            previous_time = current_time
            previous_index = i
        elif current_position == previous_position:
            notes_in_chord.append(pos_pair)
        else:
            # print('the note is too close to the previous note', current_position - previous_position, current_time - previous_time)
            # print(previous_position, current_position, previous_time, current_time)
            mismatched_indexes.append(position_pairs[previous_index]['index'])
            mismatched_indexes.append(pos_pair['index'])

    return cleaned_list, mismatched_indexes


def make_available_note_feature_list(notes, features, predicted):
    # class PosTempoPair:
    #     def __init__(self, xml_pos, pitch, qpm, index, divisions, time_pos):
    #         self.xml_position = xml_pos
    #         self.qpm = qpm
    #         self.index = index
    #         self.divisions = divisions
    #         self.pitch = pitch
    #         self.time_position = time_pos
    #         self.is_arpeggiate = False

    if not predicted:
        available_notes = []
        num_features = len(features['beat_tempo'])
        for i in range(num_features):
            qpm = features['beat_tempo'][i]

            if qpm is not None:
                xml_note = notes[i]
                xml_pos = xml_note.note_duration.xml_position
                time_pos = xml_note.note_duration.time_position
                divisions = xml_note.state_fixed.divisions
                qpm = features['beat_tempo'][i]
                pos_pair = {'xml_position': xml_pos, 
                            'pitch':xml_note.pitch[1], 
                            'beat_tempo':qpm, 
                            'index':i, 
                            'divisions':divisions, 
                            'time_position': time_pos,
                            'is_arpeggiate': False}
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair["is_arpeggiate"] = True
                available_notes.append(pos_pair)

    else:
        available_notes = []
        num_features = len(features['beat_tempo'])
        for i in range(num_features):
            xml_note = notes[i]
            xml_pos = xml_note.note_duration.xml_position
            time_pos = xml_note.note_duration.time_position
            divisions = xml_note.state_fixed.divisions
            qpm = features['beat_tempo'][i]
            pos_pair = {'xml_position': xml_pos, 
                        'pitch':xml_note.pitch[1], 
                        'beat_tempo':qpm, 
                        'index':i, 
                        'divisions':divisions, 
                        'time_position': time_pos,
                        'is_arpeggiate': False
                        }
            # pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
            available_notes.append(pos_pair)

    # minimum_time_interval = 0.05
    # available_notes = save_lowest_note_on_same_position(available_notes, minimum_time_interval)
    available_notes, _ = make_average_onset_cleaned_pair(available_notes)
    return available_notes


def get_average_onset_time(notes_in_chord_saved, threshold=0.2):
    # notes_in_chord: list of PosTempoPair Dictionary, len > 0
    notes_in_chord = copy.copy(notes_in_chord_saved)
    average_onset_time = 0
    for pos_pair in notes_in_chord:
        average_onset_time += pos_pair['time_position']
        if pos_pair['is_arpeggiate']:
            threshold = 1
    average_onset_time /= len(notes_in_chord)

    # check whether there is mis-matched note
    deviations = list()
    for pos_pair in notes_in_chord:
        dev = abs(pos_pair['time_position'] - average_onset_time)
        deviations.append(dev)
    if max(deviations) > threshold:
        # print(deviations)
        if len(notes_in_chord) == 2:
            del notes_in_chord[0:2]
        else:
            index = deviations.index(max(deviations))
            del notes_in_chord[index]
            notes_in_chord, average_onset_time = get_average_onset_time(notes_in_chord, threshold)

    return notes_in_chord, average_onset_time