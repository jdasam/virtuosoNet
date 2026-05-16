def save_lowest_note_on_same_position(alist, minimum_time_interval = 0.08):
    length = len(alist)
    previous_position = -float("Inf")
    previous_time = -float("Inf")
    alist.sort(key=lambda x: (x.xml_position, x.pitch))
    new_list = []
    for i in range(length):
        item = alist[i]
        current_position = item.xml_position
        current_time = item.time_position
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            new_list.append(item)
            previous_position = current_position
            previous_time = current_time

    return new_list


def apply_time_position_features(xml_notes, features, start_time=0, include_unmatched=True):
    num_notes = len(xml_notes)
    tempos = []
    prev_vel = 64
    current_sec = start_time

    valid_notes = []
    previous_position = start_time
    for i in range(num_notes):
        note = xml_notes[i]
        feat = features[i]
        if feat.passed_second == None:
            continue
        note.note_duration.time_position = current_sec + feat.passed_second
        current_sec += feat.passed_second

        note.note_duration.seconds = feat.duration_second
        feat.articulation = 1
        note, prev_vel = apply_feat_to_a_note(note, feat, prev_vel)

        valid_notes.append(note)

    return valid_notes

def extract_score_features(xml_notes, measure_positions, beats=None, qpm_primo=0, vel_standard=False):
    xml_length = len(xml_notes)
    # melody_notes = extract_melody_only_from_notes(xml_notes)
    features = []

    if qpm_primo == 0:
        qpm_primo = xml_notes[0].state_fixed.qpm
    tempo_primo_word = dir_enc.direction_words_flatten(xml_notes[0].tempo)
    if tempo_primo_word:
        tempo_primo = dir_enc.dynamic_embedding(tempo_primo_word, TEM_EMB_TAB, 5)
        tempo_primo = tempo_primo[0:2]
    else:
        tempo_primo = [0, 0]

    cresc_words = ['cresc', 'decresc', 'dim']

    onset_positions = list(set([note.note_duration.xml_position for note in xml_notes]))
    onset_positions.sort()
    section_positions = find_tempo_change(xml_notes)
    total_length = cal_total_xml_length(xml_notes)

    for i in range(xml_length):
        note = xml_notes[i]
        feature = MusicFeature()
        note_position = note.note_duration.xml_position
        measure_index = binary_index(measure_positions, note_position)
        if measure_index+1 < len(measure_positions):
            measure_length = measure_positions[measure_index+1] - measure_positions[measure_index]
            # measure_sec_length = measure_seocnds[measure_index+1] - measure_seocnds[measure_index]
        else:
            measure_length = measure_positions[measure_index] - measure_positions[measure_index-1]
            # measure_sec_length = measure_seocnds[measure_index] - measure_seocnds[measure_index-1]
        feature.midi_pitch = note.pitch[1]
        feature.pitch = pitch_into_vector(note.pitch[1])
        feature.duration = note.note_duration.duration / note.state_fixed.divisions

        beat_position = (note_position - measure_positions[measure_index]) / measure_length
        feature.beat_position = beat_position
        feature.beat_importance = cal_beat_importance(beat_position, note.tempo.time_numerator)
        feature.measure_length = measure_length / note.state_fixed.divisions
        feature.note_location.voice = note.voice
        feature.note_location.onset = binary_index(onset_positions, note_position)
        feature.xml_position = note.note_duration.xml_position / total_length
        feature.grace_order = note.note_duration.grace_order
        feature.is_grace_note = int(note.note_duration.is_grace_note)
        feature.preceded_by_grace_note = int(note.note_duration.preceded_by_grace_note)
        # feature.melody = int(note in melody_notes)

        feature.slur_beam_vec = [int(note.note_notations.is_slur_start), int(note.note_notations.is_slur_continue),
                                 int(note.note_notations.is_slur_stop), int(note.note_notations.is_beam_start),
                                 int(note.note_notations.is_beam_continue), int(note.note_notations.is_beam_stop)]

        # feature.time_sig_num = 1/note.tempo.time_numerator
        # feature.time_sig_den = 1/note.tempo.time_denominator
        feature.time_sig_vec = time_signature_to_vector(note.tempo.time_signature)
        feature.following_rest = note.following_rest_duration / note.state_fixed.divisions
        feature.followed_by_fermata_rest = int(note.followed_by_fermata_rest)

        dynamic_words = dir_enc.direction_words_flatten(note.dynamic)
        tempo_words = dir_enc.direction_words_flatten(note.tempo)

        # feature.dynamic = keyword_into_onehot(dynamic_words, dynamics_merged_keys)
        feature.dynamic = dir_enc.dynamic_embedding(dynamic_words, DYN_EMB_TAB, len_vec=4)
        # if dynamic_words and feature.dynamic[0] == 0:
        #     print('dynamic vector zero index value is zero:',dynamic_words.encode('utf-8'))
        if feature.dynamic[1] != 0:
            for rel in note.dynamic.relative:
                for word in cresc_words:
                    if word in rel.type['type'] or word in rel.type['content']:
                        rel_length = rel.end_xml_position - rel.xml_position
                        if rel_length == float("inf") or rel_length == 0:
                            rel_length = note.state_fixed.divisions * 10
                        ratio = (note_position - rel.xml_position) / rel_length
                        feature.dynamic[1] *= (ratio+0.05)
                        break
        if note.dynamic.cresciuto:
            feature.cresciuto = (note.dynamic.cresciuto.overlapped +1) / 2
            if note.dynamic.cresciuto.type == 'diminuendo':
                feature.cresciuto *= -1
        else:
            feature.cresciuto = 0
        feature.dynamic.append(feature.cresciuto)
        feature.tempo = dir_enc.dynamic_embedding(tempo_words, TEM_EMB_TAB, len_vec=5)
        # TEMP CODE for debugging
        # if dynamic_words:
        #     tempo_pair = (dynamic_words.encode('utf-8'), [feature.dynamic[0]] + feature.dynamic[2:])
        #     if tempo_pair not in TEMP_WORDS:
        #         TEMP_WORDS.append(tempo_pair)
        #         print('tempo pair: ', tempo_pair)

        # feature.tempo = keyword_into_onehot(note.tempo.absolute, tempos_merged_key)
        feature.notation = note_notation_to_vector(note)
        feature.qpm_primo = math.log(qpm_primo, 10)
        feature.tempo_primo = tempo_primo
        feature.note_location.measure = note.measure_number-1
        feature.note_location.section = binary_index(section_positions, note_position)
        feature.distance_from_abs_dynamic = (note.note_duration.xml_position - note.dynamic.absolute_position) / note.state_fixed.divisions
        feature.distance_from_recent_tempo = (note_position - note.tempo.recently_changed_position) / note.state_fixed.divisions
        # print(feature.dynamic + feature.tempo)
        features.append(feature)

    _, piano_mark, _, forte_mark = cal_mean_velocity_by_dynamics_marking(features)
    if beats:
        for beat in beats:
            num = 0
            note = get_item_by_xml_position(xml_notes, beat)
            note_index = xml_notes.index(note)
            while note_index-num >=0 and xml_notes[note_index-num].note_duration.xml_position == note.note_duration.xml_position:
                feat = features[note_index-num]
                feat.is_beat = True
                num += 1

        for i in range(xml_length):
            note = xml_notes[i]
            beat = binary_index(beats, note.note_duration.xml_position)
            features[i].note_location.beat = beat
            if vel_standard:
                features[i].note_location.beat = beat
                features[i].mean_piano_mark = piano_mark
                features[i].mean_forte_mark = forte_mark
                features[i].mean_piano_vel = vel_standard[0]
                features[i].mean_forte_vel = vel_standard[1]

    return features



def extract_perform_features(xml_doc, xml_notes, pairs, perf_midi, measure_positions):
    beats = xml_doc.get_beat_positions()
    accidentals_in_words = xml_doc.get_accidentals
    score_features = extract_score_features(xml_notes, measure_positions, beats=beats)
    feat_len = len(score_features)

    tempos, measure_tempos, section_tempos = cal_tempo(xml_doc, xml_notes, pairs)
    previous_qpm = 1
    save_qpm = xml_notes[0].state_fixed.qpm
    previous_second = None

    def cal_qpm_primo(tempos, view_range=10):
        qpm_primo = 0
        for i in range(view_range):
            tempo = tempos[i]
            qpm_primo += tempo.qpm

        return qpm_primo / view_range

    qpm_primo = cal_qpm_primo(tempos)
    qpm_primo = math.log(qpm_primo, 10)

    prev_articulation = 0
    prev_vel = 64
    prev_pedal = 0
    prev_soft_pedal = 0
    prev_start = 0

    num_matched_notes = 0
    num_unmatched_notes = 0

    for i in range(feat_len):
        feature= score_features[i]
        # tempo = find_corresp_tempo(pairs, i, tempos)
        tempo = get_item_by_xml_position(tempos, xml_notes[i])
        measure_tempo = get_item_by_xml_position(measure_tempos, xml_notes[i])
        section_tempo = get_item_by_xml_position(section_tempos, xml_notes[i])
        if tempo.qpm > 0:
            feature.qpm = math.log(tempo.qpm, 10)
            feature.measure_tempo = math.log(measure_tempo.qpm, 10)
            feature.section_tempo = math.log(section_tempo.qpm, 10)
        else:
            print('Error: qpm is zero')
        if tempo.qpm > 1000:
            print('Need Check: qpm is ' + str(tempo.qpm))
        if tempo.qpm != save_qpm:
            # feature.previous_tempo = math.log(previous_qpm, 10)
            previous_qpm = save_qpm
            save_qpm = tempo.qpm
        if xml_notes[i].note_notations.is_trill:
            feature.trill_param, trill_length = find_corresp_trill_notes_from_midi(xml_doc, xml_notes, pairs, perf_midi, accidentals_in_words, i)
        else:
            feature.trill_param = [0] * 5
            trill_length = None

        if not pairs[i] == []:
            feature.align_matched = 1
            num_matched_notes += 1
            feature.articulation = cal_articulation_with_tempo(pairs, i, tempo.qpm, trill_length)
            feature.xml_deviation = cal_onset_deviation_with_tempo(pairs, i, tempo)
            # feature['IOI_ratio'], feature['articulation']  = calculate_IOI_articulation(pairs,i, total_length_tuple)
            # feature['loudness'] = math.log( pairs[i]['midi'].velocity / velocity_mean, 10)
            feature.velocity = pairs[i]['midi'].velocity
            # feature['xml_deviation'] = cal_onset_deviation(xml_notes, melody_notes, melody_onset_positions, pairs, i)
            feature.pedal_at_start = pedal_sigmoid(pairs[i]['midi'].pedal_at_start)
            feature.pedal_at_end = pedal_sigmoid(pairs[i]['midi'].pedal_at_end)
            feature.pedal_refresh = pedal_sigmoid(pairs[i]['midi'].pedal_refresh)
            feature.pedal_refresh_time = pairs[i]['midi'].pedal_refresh_time
            feature.pedal_cut = pedal_sigmoid(pairs[i]['midi'].pedal_cut)
            feature.pedal_cut_time = pairs[i]['midi'].pedal_cut_time
            feature.soft_pedal = pedal_sigmoid(pairs[i]['midi'].soft_pedal)

            if feature.pedal_at_end > 70:
                feature.articulation_loss_weight = 0.05
            elif feature.pedal_at_end > 60:
                feature.articulation_loss_weight = 0.5
            else:
                feature.articulation_loss_weight = 1

            if feature.pedal_at_end > 64 and feature.pedal_refresh < 64:
                # pedal refresh occurs in the note
                feature.articulation_loss_weight = 1

            feature.midi_start = pairs[i]['midi'].start # just for reproducing and testing perform features

            if previous_second is None:
                feature.passed_second = 0
            else:
                feature.passed_second = pairs[i]['midi'].start - previous_second
            feature.duration_second = pairs[i]['midi'].end - pairs[i]['midi'].start
            previous_second = pairs[i]['midi'].start
            # if not feature['melody'] and not feature['IOI_ratio'] == None :
            #     feature['IOI_ratio'] = 0

            prev_articulation = feature.articulation
            prev_vel = feature.velocity
            prev_pedal = feature.pedal_at_start
            prev_soft_pedal = feature.soft_pedal
            prev_start = feature.midi_start
        else:
            feature.align_matched = 0
            num_unmatched_notes += 1
            feature.articulation = prev_articulation
            feature.xml_deviation = 0
            feature.velocity = prev_vel
            feature.pedal_at_start = prev_pedal
            feature.pedal_at_end = prev_pedal
            feature.pedal_refresh = prev_pedal
            feature.pedal_cut = prev_pedal
            feature.pedal_refresh_time = 0
            feature.pedal_cut_time = 0
            feature.soft_pedal = prev_soft_pedal
            feature.midi_start = prev_start
            feature.articulation_loss_weight = 0

        feature.previous_tempo = math.log(previous_qpm, 10)
        feature.qpm_primo = qpm_primo

    piano_vel, piano_vec, forte_vel, forte_vec = cal_mean_velocity_by_dynamics_marking(score_features)
    for feature in score_features:
        feature.mean_piano_vel = piano_vel
        feature.mean_piano_mark = piano_vec
        feature.mean_forte_vel = forte_vel
        feature.mean_forte_mark = forte_vec

    score_features = make_index_continuous(score_features, score=False)
    score_features = cal_and_add_dynamic_by_position(score_features)
    print('Number of Matched Notes: ' + str(num_matched_notes) + ', unmatched notes: ' + str(num_unmatched_notes))

    return score_features


def extract_melody_only_from_notes(xml_notes):
    melody_notes = []
    for note in xml_notes:
        if note.voice == 1 and not note.note_duration.is_grace_note:
            melody_notes.append(note)
    melody_notes = MusicXMLDocument.delete_chord_notes_for_melody(MusicXMLDocument, melody_notes=melody_notes)

    return melody_notes

def calculate_IOI_articulation(pairs, index, total_length):
    margin = len(pairs) - index
    if margin < 1:
        return None, None
    if pairs[index]['xml'].note_duration.is_grace_note:
        return 0, 0
    for i in range(1, margin-1):
        if not pairs[index+i] ==[] and pairs[index]['xml'].voice ==  pairs[index+i]['xml'].voice \
                and pairs[index]['xml'].chord_index ==  pairs[index+i]['xml'].chord_index\
                and not pairs[index+i]['xml'].note_duration.is_grace_note:
            xml_ioi = pairs[index + i]['xml'].note_duration.xml_position - pairs[index][
                'xml'].note_duration.xml_position
            midi_ioi = pairs[index + i]['midi'].start - pairs[index]['midi'].start
            if midi_ioi <= 0 or xml_ioi <= 0 or total_length[1] <= 0 or total_length[0] <= 0:
                return None, None
            xml_length = pairs[index]['xml'].note_duration.duration
            midi_length = pairs[index]['midi'].end - pairs[index]['midi'].start
            ioi = math.log(midi_ioi / total_length[1] / (xml_ioi / total_length[0]), 10)
            articulation = xml_ioi / xml_length / (midi_ioi / midi_length)
            return ioi, articulation
    return None, None



def calculate_total_length(pairs):
    first_pair_index = 0
    for i in range(len(pairs)):
        if not pairs[i]  == []:
            first_pair_index = i
            break
    for i in range(len(pairs)):
        if not pairs[-i-1] == []:
            xml_length =  pairs[-i-1]['xml'].note_duration.xml_position - pairs[first_pair_index]['xml'].note_duration.xml_position
            midi_length = pairs[-i-1]['midi'].start - pairs[first_pair_index]['midi'].start
            return (xml_length, midi_length)


def calculate_mean_velocity(pairs):
    sum = 0
    length =0
    for pair in pairs:
        if not pair == []:
            sum += pair['midi'].velocity
            length += 1

    return sum/float(length)


def cal_pitch_interval_and_duration_ratio(xml_notes, index):
    search_index = 1
    num_notes = len(xml_notes)
    note = xml_notes[index]
    candidate_notes = []
    next_position = note.note_duration.xml_position + note.note_duration.duration
    while index+search_index <num_notes:
        next_note = xml_notes[index+search_index]
        if next_note.note_duration.xml_position == next_position:
            if next_note.voice == note.voice and not next_note.note_duration.is_grace_note:
                if next_note.chord_index == note.chord_index:
                    pitch_interval = next_note.pitch[1] - note.pitch[1]
                    if note.note_duration.is_grace_note:
                        duration_ratio = 0
                    else:
                        duration_ratio = math.log(next_note.note_duration.duration / note.note_duration.duration, 10)
                    return pitch_interval, duration_ratio
                else:
                    candidate_notes.append(next_note)
        elif next_note.note_duration.xml_position > next_position:
            # there is no notes to search further

            if len(candidate_notes) > 0:
                closest_pitch = float('inf')
                closest_note = None
                for cand_note in candidate_notes:
                    temp_interval = cand_note.pitch[1] - note.pitch[1]
                    if abs(temp_interval)<closest_pitch:
                        closest_pitch = abs(temp_interval)
                        closest_note = cand_note
                if note.note_duration.is_grace_note:
                    duration_ratio = 0
                else:
                    duration_ratio = math.log(cand_note.note_duration.duration / note.note_duration.duration, 10)
                return temp_interval, duration_ratio
            else:
                # if next_note.voice == note.voice:
                #     rest_duration = (next_note.note_duration.xml_position - next_position) / note.state_fixed.divisions
                #     return 0, 0
                # else:
                break

        search_index += 1

    return None, 0



def calculate_pitch_interval(xml_notes, index):
    search_index = 1
    num_notes = len(xml_notes)
    note = xml_notes[index]
    while index+search_index <num_notes:
        next_note = xml_notes[index+search_index]
        if next_note.voice == note.voice and next_note.chord_index == note.chord_index \
                and not next_note.note_duration.is_grace_note:
            # check whether the next note is directly following the previous note
            if note.note_duration.xml_position + note.note_duration.duration == next_note.note_duration.xml_position:
                return xml_notes[index+search_index].pitch[1] - xml_notes[index].pitch[1]
            else:
                return 0
        search_index += 1
    # if index < len(xml_notes)-1:
    #     pitch_interval = xml_notes[index+1].pitch[1] - xml_notes[index].pitch[1]
    # else:
    #     pitch_interval = None
    # return pitch_interval
    return 0

def find_corresp_tempo(note, tempos):
    # note = pairs[index]['xml']
    tempo =  get_item_by_xml_position(tempos, note)
    # log_tempo = math.log(tempo, 10)
    # return log_tempo
    return tempo


def load_pairs_from_folder(path, pedal_elongate=False):
    global NUM_SCORE_NOTES
    global NUM_EXCLUDED_NOTES
    global NUM_NON_MATCHED_NOTES
    global NUM_PERF_NOTES

    score_midi_name = path+'midi_cleaned.mid'
    path_split = copy.copy(path).split('/')
    if path_split[0] == 'chopin_cleaned':
        composer_name = copy.copy(path).split('/')[1]
    else:
        dataset_folder_name_index = path_split.index('chopin_cleaned')
        composer_name = copy.copy(path).split('/')[dataset_folder_name_index+1]
    composer_name_vec = composer_name_to_vec(composer_name)

    XMLDocument, xml_notes = read_xml_to_notes(path)
    score_midi = midi_utils.to_midi_zero(score_midi_name)
    score_midi_notes = score_midi.instruments[0].notes
    score_midi_notes.sort(key=lambda x:x.start)
    match_list = matching.match_xml_to_midi(xml_notes, score_midi_notes)
    score_pairs = matching.make_xml_midi_pair(xml_notes, score_midi_notes, match_list)
    num_non_matched = check_pairs(score_pairs)
    if num_non_matched > 100:
        print("There are too many xml-midi matching errors")
        return None
    NUM_SCORE_NOTES += len(score_midi_notes) - num_non_matched

    measure_positions = XMLDocument.get_measure_positions()
    filenames = os.listdir(path)
    perform_features_piece = []
    notes_graph = score_graph.make_edge(xml_notes)

    for file in filenames:
        if file[-18:] == '_infer_corresp.txt':
            perf_name = file.split('_infer')[0]
            # perf_score = evaluation.cal_score(perf_name)
            perf_score = 0

            perf_midi_name = path + perf_name + '.mid'
            perf_midi = midi_utils.to_midi_zero(perf_midi_name)
            perf_midi = midi_utils.add_pedal_inf_to_notes(perf_midi)
            if pedal_elongate:
                perf_midi = midi_utils.elongate_offset_by_pedal(perf_midi)
            perf_midi_notes= perf_midi.instruments[0].notes
            corresp_name = path + file
            corresp = matching.read_corresp(corresp_name)

            xml_perform_match = matching.match_score_pair2perform(score_pairs, perf_midi_notes, corresp)
            perform_pairs = matching.make_xml_midi_pair(xml_notes, perf_midi_notes, xml_perform_match)
            print("performance name is " + perf_name)
            num_align_error = check_pairs(perform_pairs)

            if num_align_error > 1000:
                print('Too many align error in the performance')
                continue
            NUM_PERF_NOTES += len(score_midi_notes) - num_align_error
            NUM_NON_MATCHED_NOTES += num_align_error
            perform_features = extract_perform_features(XMLDocument, xml_notes, perform_pairs, perf_midi_notes, measure_positions)
            perform_feat_score = {'features': perform_features, 'score': perf_score, 'composer': composer_name_vec, 'graph': notes_graph}

            perform_features_piece.append(perform_feat_score)
    if perform_features_piece == []:
        return None
    return perform_features_piece



def find_notes_between_melody_notes(total_notes, melody_notes):
    num_melody_notes = len(melody_notes)
    num_total_notes = len(total_notes)
    between_notes = [[] for x in range(num_melody_notes)]
    melody_onset_positions = list(set(note.note_duration.xml_position for note in melody_notes))

    melody_onset_positions.sort()

    for i in range(num_total_notes):
        note = total_notes[i]
        if note in melody_notes:
            continue
        melody_index = binary_index(melody_onset_positions, note.note_duration.xml_position)
        between_notes[melody_index].append(note)

    return between_notes



def read_xml_to_array(path_name, means, stds, start_tempo, composer_name, vel_standard):
    xml_object, xml_notes = read_xml_to_notes(path_name)
    beats = xml_object.get_beat_positions()
    measure_positions = xml_object.get_measure_positions()
    features = extract_score_features(xml_notes, measure_positions, beats, qpm_primo=start_tempo, vel_standard=vel_standard)
    features = make_index_continuous(features, score=True)
    composer_vec = composer_name_to_vec(composer_name)
    edges = score_graph.make_edge(xml_notes)

    for i in range(len(stds[0])):
        if stds[0][i] < 1e-4 or isinstance(stds[0][i], complex):
            stds[0][i] = 1

    test_x = []
    note_locations = []
    for feat in features:
        temp_x = [(feat.midi_pitch - means[0][0]) / stds[0][0], (feat.duration - means[0][1]) / stds[0][1],
                    (feat.beat_importance-means[0][2])/stds[0][2], (feat.measure_length-means[0][3])/stds[0][3],
                   (feat.qpm_primo - means[0][4]) / stds[0][4],(feat.following_rest - means[0][5]) / stds[0][5],
                    (feat.distance_from_abs_dynamic - means[0][6]) / stds[0][6],
                  (feat.distance_from_recent_tempo - means[0][7]) / stds[0][7] ,
                  feat.beat_position, feat.xml_position, feat.grace_order,
                    feat.preceded_by_grace_note, feat.followed_by_fermata_rest] \
                   + feat.pitch + feat.tempo + feat.dynamic + feat.time_sig_vec + feat.slur_beam_vec + composer_vec + feat.notation + feat.tempo_primo
        # temp_x.append(feat.is_beat)
        test_x.append(temp_x)
        note_locations.append(feat.note_location)

    return test_x, xml_notes, xml_object, edges, note_locations
