import csv


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
    pairs = []
    for i in range(len(match_list)):
        if not match_list[i] ==[]:
            temp_pair = {'xml': xml_notes[i], 'midi': midi_notes[match_list[i]]}
            pairs.append(temp_pair)
        else:
            pairs.append([])
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