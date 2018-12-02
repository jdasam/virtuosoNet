#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import csv
import math
import os
import pretty_midi
from musicxml_parser.mxp import MusicXMLDocument
# import sys
# # sys.setdefaultencoding() does not exist, here!
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')
import midi_utils.midi_utils as midi_utils
import copy
import evaluation

absolute_tempos_keywords = ['adagio', 'grave', 'lento', 'largo', 'larghetto', 'andante', 'andantino', 'moderato',
                            'allegretto', 'allegro', 'vivace', 'accarezzevole', 'languido', 'tempo giusto', 'mesto',
                            'presto', 'prestissimo', 'maestoso', 'lullaby', 'doppio movimento',
                            'Freely, with expression', "d'un rythme souple", 'agitato',
                            'leicht und zart', 'aufgeregt', 'bewegt', 'rasch', 'innig', 'lebhaft',
                            'lent', 'large', 'vif', 'animé']
relative_tempos_keywords = ['animato', 'pesante', 'veloce',
                            'acc', 'accel', 'rit', 'ritardando', 'accelerando', 'rall', 'rallentando', 'ritenuto', 'string',
                            'a tempo', 'stretto', 'slentando', 'meno mosso', 'meno vivo', 'più mosso', 'allargando', 'smorzando', 'appassionato', 'perdendo',
                            'langsamer', 'schneller', 'bewegter',
                            'plus lent', 'più lento', 'retenu', 'revenez', 'cédez', 'mesuré', 'élargissant', 'accélerez', 'rapide', 'Reprenez  le  mouvement']
relative_long_tempo_keywords = ['meno mosso', 'meno vivo', 'più mosso', 'animato', 'langsamer', 'schneller',
                                'stretto', 'bewegter', 'tranquillo',
                                'plus lent']
tempo_primo_words = ['tempo i', 'tempo primo', 'erstes tempo', '1er mouvement', '1er mouvt', 'au mouvtdu début', 'au mouvement', 'au mouvt', '1o tempo']
absolute_tempos_keywords += tempo_primo_words

tempos_keywords = absolute_tempos_keywords + relative_tempos_keywords
tempos_merged_key = ['adagio', 'lento', 'andante', 'andantino', 'moderato', 'allegretto', 'allegro', 'vivace',
                     'presto', 'prestissimo', 'animato', 'maestoso', 'pesante', 'veloce', 'tempo i', 'lullaby', 'agitato',
                     ['acc', 'accel', 'accelerando'],['rit', 'ritardando', 'rall', 'rallentando'], 'ritenuto',
                    'a tempo', 'stretto', 'slentando', 'meno mosso', 'più mosso', 'allargando' ]


absolute_dynamics_keywords = ['pppp','ppp', 'pp', 'p', 'piano', 'mp', 'mf', 'f', 'forte', 'ff', 'fff', 'fp', 'ffp']
relative_dynamics_keywords = ['crescendo', 'diminuendo', 'cresc', 'dim', 'dimin' 'sotto voce',
                              'mezza voce', 'sf', 'fz', 'sfz', 'rf,' 'sffz', 'rf', 'rinf',
                              'con forza', 'con fuoco', 'smorzando', 'appassionato', 'perdendo']

dynamics_keywords = absolute_dynamics_keywords + relative_dynamics_keywords
dynamics_merged_keys = ['ppp', 'pp', ['p', 'piano'], 'mp', 'mf', ['f', 'forte'], 'ff', 'fff', 'fp', ['crescendo', 'cresc'],  ['diminuendo', 'dim', 'dimin'],
                        'sotto voce', 'mezza voce', ['sf', 'fz', 'sfz', 'sffz'] ]


VALID_LIST =['Chopin/Chopin_Etude_op_10/1/',
             'Chopin/Chopin_Etude_op_10/10/',
             'Chopin/Chopin_Etude_op_10/12/',
             'Chopin/Chopin_Etude_op_25/4/',
             'Chopin/Chopin_Etude_op_25/10/',
             'Chopin/Chopin_Scherzos/31/',
             'Chopin/Chopin_Sonata_3/4th/',
             'Mozart/Piano_Sonatas/12-3/',
             'Haydn/Keyboard_Sonatas/46-1/',
             'Rachmaninoff/Preludes_op_23/4/',
             'Beethoven/Piano_Sonatas/3-1/',
             'Beethoven/Piano_Sonatas/12-1/',
             'Beethoven/Piano_Sonatas/15-4/',
             'Beethoven/Piano_Sonatas/21-2/',
             'Schumann/Kreisleriana/5/',
             'Liszt/Transcendental_Etudes/4',
             'Liszt/Transcendental_Etudes/9',
             'Liszt/Concert_Etude_S145/2'
             ]

def apply_tied_notes(xml_parsed_notes):
    tie_clean_list = []
    for i in range(len(xml_parsed_notes)):
        if xml_parsed_notes[i].note_notations.tied_stop == False:
            tie_clean_list.append(xml_parsed_notes[i])
        else:
            for j in reversed(range(len(tie_clean_list))):
                if tie_clean_list[j].note_notations.tied_start == True and tie_clean_list[j].pitch[1] == xml_parsed_notes[i].pitch[1]:
                    tie_clean_list[j].note_duration.seconds +=  xml_parsed_notes[i].note_duration.seconds
                    tie_clean_list[j].note_duration.duration +=  xml_parsed_notes[i].note_duration.duration
                    tie_clean_list[j].note_duration.midi_ticks +=  xml_parsed_notes[i].note_duration.midi_ticks
                    if xml_parsed_notes[i].note_notations.slurs:
                        for slur in  xml_parsed_notes[i].note_notations.slurs:
                            tie_clean_list[j].note_notations.slurs.append(slur)
                    break
    return tie_clean_list


def matchXMLtoMIDI(xml_notes, midi_notes):
    candidates_list = []
    match_list = []
    midi_positions = [note.start for note in midi_notes]
    def find_candidate_list(xml_note, midi_notes, midi_positions):
        num_midi = len(midi_notes)
        temp_list =[]
        match_threshold = 0.1
        if note.is_rest:
            return([])
        note_start = xml_note.note_duration.time_position
        if note.note_duration.after_grace_note:
            note_start += 0.5
            match_threshold = 0.6
        elif note.note_notations.is_arpeggiate:
            note_start += 0.3
            match_threshold = 0.4

        nearby_index = binaryIndex(midi_positions, note_start)

        for i in range(-10, 10):
            index = nearby_index+i
            if index < 0:
                index = 0
            elif index >= num_midi:
                break
            midi_note = midi_notes[index]
            if midi_note.pitch == note.pitch[1] or abs(midi_note.start - note_start) < match_threshold:
                temp_list.append({'index': index, 'midi_note':midi_note})

            if midi_note.start > note_start + match_threshold:
                break

        return temp_list

    # for each note in xml, make candidates of the matching midi note
    for note in xml_notes:
        match_threshold = 0.1
        if note.is_rest:
            candidates_list.append([])
            continue
        note_start = note.note_duration.time_position
        if note.note_duration.after_grace_note:
            note_start += 0.5
            match_threshold = 0.6
        # check grace note and adjust time_position
        note_pitch = note.pitch[1]
        temp_list = [{'index': index, 'midi_note': midi_note} for index, midi_note in enumerate(midi_notes) if abs(midi_note.start - note_start) < match_threshold and midi_note.pitch == note_pitch]
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
    match_list = matchXMLtoMIDI(xml_notes, midi_notes)
    score_pairs = make_xml_midi_pair(xml_notes, midi_notes, match_list)
    xml_perform_match = match_score_pair2perform(score_pairs, perform_notes, corresp)
    perform_pairs = make_xml_midi_pair(xml_notes, perform_notes, xml_perform_match)

    return score_pairs, perform_pairs


def extract_notes(xml_Doc, melody_only = False, grace_note = False):
    # parts = xml_Doc.parts[0]
    notes = []
    previous_grace_notes = []
    rests = []
    instrument_index = 0
    for part in xml_Doc.parts:
        measure_number = 1
        for measure in part.measures:
            for note in measure.notes:
                note.measure_number = measure_number
                note.voice += instrument_index * 10
                if melody_only:
                    if note.voice == 1:
                        notes, previous_grace_notes, rests= check_notes_and_append(note, notes, previous_grace_notes, rests, grace_note)
                else:
                    notes, previous_grace_notes, rests = check_notes_and_append(note, notes, previous_grace_notes, rests, grace_note)

            measure_number += 1
        notes = apply_after_grace_note_to_chord_notes(notes)
        if melody_only:
            notes = delete_chord_notes_for_melody(notes)
        notes = apply_tied_notes(notes)
        notes.sort(key=lambda x: (x.note_duration.xml_position, x.note_duration.grace_order, -x.pitch[1]))
        notes = check_overlapped_notes(notes)
        notes = apply_rest_to_note(notes, rests)
        notes = omit_trill_notes(notes)
        notes = extract_and_apply_slurs(notes)
        notes = rearrange_chord_index(notes)
    # for note in notes:
    #     print(note.staff, note.voice, note.note_duration.xml_position, note.note_duration.duration, note.pitch[1], note.chord_index)
        instrument_index += 1

    return notes


def check_notes_and_append(note, notes, previous_grace_notes, rests, include_grace_note):
    if note.note_duration.is_grace_note:
        previous_grace_notes.append(note)
        if include_grace_note:
            notes.append(note)
    elif not note.is_rest:
        if len(previous_grace_notes) > 0:
            rest_grc = []
            added_grc = []
            grace_order = -1
            num_grc = len(previous_grace_notes)
            for grc in reversed(previous_grace_notes):
                if grc.voice == note.voice:
                    note.note_duration.after_grace_note = True
                    grc.note_duration.grace_order = grace_order
                    grc.following_note = note
                    if grc.chord_index == 0:
                        grace_order -= 1
                    added_grc.append(grc)
                    # notes.append(grc)
                else:
                    rest_grc.append(grc)
            num_added = abs(grace_order) -1
            for grc in added_grc:
                # grc.note_duration.grace_order /= num_added
                grc.note_duration.num_grace = num_added

            previous_grace_notes = rest_grc
        notes.append(note)
    else:
        assert note.is_rest
        if note.is_print_object:
            rests.append(note)

    return notes, previous_grace_notes, rests


def apply_rest_to_note(xml_notes, rests):
    xml_positions = [note.note_duration.xml_position for note in xml_notes]
    # concat continuous rests
    previous_rest = rests[0]
    new_rests = []
    for rest in rests:
        previous_end = previous_rest.note_duration.xml_position + previous_rest.note_duration.duration
        if previous_rest.voice == rest.voice and\
                previous_end == rest.note_duration.xml_position:
            previous_rest.note_duration.duration += rest.note_duration.duration
        else:
            new_rests.append(rest)
            previous_rest = rest

    rests = new_rests

    for rest in rests:
        rest_position = rest.note_duration.xml_position
        closest_note_index = binaryIndex(xml_positions, rest_position)
        search_index = 1
        while closest_note_index - search_index >= 0:
            prev_note = xml_notes[closest_note_index - search_index]
            prev_note_end = prev_note.note_duration.xml_position + prev_note.note_duration.duration
            if prev_note_end == rest_position and prev_note.voice == rest.voice:
                prev_note.following_rest_duration = rest.note_duration.duration
                break
            search_index += 1

    return xml_notes


def apply_after_grace_note_to_chord_notes(notes):
    for note in notes:
        if note.note_duration.after_grace_note:
            onset= note.note_duration.xml_position
            voice = note.voice
            chords = find(lambda x: x.note_duration.xml_position == onset and x.voice == voice, notes)
            for chd in chords:
                chd.note_duration.after_grace_note = True
    return notes


def extract_measure_position(xml_Doc):
    parts = xml_Doc.parts[0]
    measure_positions = []

    for measure in parts.measures:
        measure_positions.append(measure.start_xml_position)

    return measure_positions


def delete_chord_notes_for_melody(melody_notes):
    note_onset_positions = list(set(note.note_duration.xml_position for note in melody_notes))
    note_onset_positions.sort()
    unique_melody = []
    for onset in note_onset_positions:
        notes = find(lambda x: x.note_duration.xml_position == onset, melody_notes)
        if len(notes) == 1:
            unique_melody.append(notes[0])
        else:
            notes.sort(key=lambda x: x.pitch[1])
            unique_melody.append(notes[-1])

    return unique_melody


def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    items_list = []
    for item in seq:
        if f(item):
            items_list.append(item)
    return items_list


class MusicFeature():
    def __init__(self):
        self.pitch = None
        self.pitch_interval = None
        self.duration = None
        self.duration_ratio = None
        self.beat_position = None
        self.measure_length = None
        self.voice = None
        self.xml_position = None
        self.grace_order = None
        self.melody = None
        self.time_sig_num = None
        self.time_sig_den = None
        self.is_beat = False
        self.following_rest = 0
        self.tempo_primo = None
        self.qpm_primo = None
        self.beat_index = 0
        self.measure_index = 0
        self.no_following_note=0
        self.note_location = self.NoteLocation(None,None,None,None)
        self.distance_from_abs_dynamic = None
        self.slur_index = None

        self.align_matched = 0
        self.dynamic  = None
        self.tempo = None
        self.notation = None
        self.qpm = None
        self.previous_tempo = None
        self.IOI_ratio = None
        self.articulation = None
        self.xml_deviation = None
        self.velocity = 0
        self.pedal_at_start = None
        self.pedal_at_end = None
        self.pedal_refresh = None
        self.pedal_refresh_time = None
        self.pedal_cut = None
        self.pedal_cut_time = None
        self.soft_pedal = None

        self.mean_piano_vel = None
        self.mean_piano_mark = None
        self.mean_forte_vel = None
        self.mean_forte_mark = None

        self.midi_start  = None
        self.passed_second = None
        self.duration_second = None


    class NoteLocation():
        def __init__(self, beat, measure, voice, onset):
            self.beat = beat
            self.measure = measure
            self.voice = voice
            self.onset = onset


def extract_score_features(xml_notes, measure_positions, beats=None, qpm_primo=0, vel_standard=False):
    features = []
    xml_length = len(xml_notes)
    melody_notes = extract_melody_only_from_notes(xml_notes)
    features = []
    dynamic_embed_table = define_dyanmic_embedding_table()
    tempo_embed_table = define_tempo_embedding_table()

    if qpm_primo == 0:
        qpm_primo = xml_notes[0].state_fixed.qpm
    tempo_primo_word = direction_words_flatten(xml_notes[0].tempo)
    tempo_primo = dynamic_embedding(tempo_primo_word, tempo_embed_table, 3)
    tempo_primo = tempo_primo[0:2]

    cresc_words = ['cresc', 'decresc', 'dim']

    onset_positions = list(set([note.note_duration.xml_position for note in xml_notes]))
    onset_positions.sort()

    for i in range(xml_length):
        note = xml_notes[i]
        feature = MusicFeature()
        note_position = note.note_duration.xml_position
        measure_index = binaryIndex(measure_positions, note_position)
        total_length = cal_total_xml_length(xml_notes)
        if measure_index+1 <len(measure_positions):
            measure_length = measure_positions[measure_index+1] - measure_positions[measure_index]
            # measure_sec_length = measure_seocnds[measure_index+1] - measure_seocnds[measure_index]
        else:
            measure_length = measure_positions[measure_index] - measure_positions[measure_index-1]
            # measure_sec_length = measure_seocnds[measure_index] - measure_seocnds[measure_index-1]

        feature.pitch = pitch_into_vector(note.pitch[1])
        # feature.pitch_interval = calculate_pitch_interval(xml_notes, i)
        # feature.duration = note.note_duration.duration / measure_length
        feature.duration = note.note_duration.duration / note.state_fixed.divisions
        # feature.duration_ratio = calculate_duration_ratio(xml_notes, i)
        pitch_interval, feature.duration_ratio = cal_pitch_interval_and_duration_ratio(xml_notes, i)
        # feature.pitch_interval = pitch_interval_into_vector(pitch_interval)
        if pitch_interval is None:
            feature.pitch_interval = 0
            feature.no_following_note = 1
        else:
            feature.pitch_interval = pitch_interval
            feature.no_following_note = 0

        beat_position = (note_position - measure_positions[measure_index]) / measure_length
        feature.beat_position = cal_beat_importance(beat_position, note.tempo.time_numerator)
        feature.measure_length = measure_length / note.state_fixed.divisions
        feature.note_location.voice = note.voice
        feature.note_location.onset = binaryIndex(onset_positions, note_position)
        feature.xml_position = note.note_duration.xml_position / total_length
        feature.grace_order = note.note_duration.grace_order
        feature.melody = int(note in melody_notes)
        feature.time_sig_num = 1/note.tempo.time_numerator
        feature.time_sig_den = 1/note.tempo.time_denominator
        feature.following_rest = note.following_rest_duration / note.state_fixed.divisions

        dynamic_words = direction_words_flatten(note.dynamic)
        tempo_words = direction_words_flatten(note.tempo)

        # feature.dynamic = keyword_into_onehot(dynamic_words, dynamics_merged_keys)
        feature.dynamic = dynamic_embedding(dynamic_words, dynamic_embed_table)
        if dynamic_words and feature.dynamic[0] == 0:
            print(dynamic_words.encode('utf-8'))
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
        feature.tempo = dynamic_embedding(tempo_words, tempo_embed_table, len_vec=3)
        # feature.tempo = keyword_into_onehot(note.tempo.absolute, tempos_merged_key)
        feature.notation = note_notation_to_vector(note)
        feature.qpm_primo = math.log(qpm_primo,10)
        feature.tempo_primo = tempo_primo
        feature.note_location.measure = note.measure_number-1
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
        beat = binaryIndex(beats, note.note_duration.xml_position)
        features[i].note_location.beat = beat
        if vel_standard:
            features[i].note_location.beat = beat
            features[i].mean_piano_mark = piano_mark
            features[i].mean_forte_mark = forte_mark
            features[i].mean_piano_vel = vel_standard[0]
            features[i].mean_forte_vel = vel_standard[1]

    return features


def extract_perform_features(xml_doc, xml_notes, pairs, perf_midi, measure_positions):
    # print(len(xml_notes), len(pairs))
    velocity_mean = calculate_mean_velocity(pairs)
    total_length_tuple = calculate_total_length(pairs)
    # measure_seocnds = make_midi_measure_seconds(pairs, measure_positions)
    melody_notes = extract_melody_only_from_notes(xml_notes)
    melody_onset_positions = list(set(note.note_duration.xml_position for note in melody_notes))
    melody_onset_positions.sort()
    if not len(melody_notes) == len(melody_onset_positions):
        print('length of melody notes and onset positions are different')
    beats = cal_beat_positions_of_piece(xml_doc)
    accidentals_in_words = extract_accidental(xml_doc)
    score_features = extract_score_features(xml_notes, measure_positions, beats=beats)
    feat_len = len(score_features)

    tempos = cal_tempo(xml_doc, xml_notes, pairs, score_features)
    # for tempo in tempos:
    #     print(tempo.qpm, tempo.time_position, tempo.end_time)
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

    prev_articulation = 1
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
        if tempo.qpm > 0:
            feature.qpm = math.log(tempo.qpm, 10)
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


        feature.previous_tempo = math.log(previous_qpm, 10)
        feature.qpm_primo = qpm_primo

    piano_vel, piano_vec, forte_vel, forte_vec = cal_mean_velocity_by_dynamics_marking(score_features)
    for feature in score_features:
        feature.mean_piano_vel = piano_vel
        feature.mean_piano_mark = piano_vec
        feature.mean_forte_vel = forte_vel
        feature.mean_forte_mark = forte_vec

    score_features = make_index_continuous(score_features, score=False)
    print('Number of Matched Notes: ' + str(num_matched_notes) + ', unmatched notes: ' + str(num_unmatched_notes))

    return score_features

def extract_melody_only_from_notes(xml_notes):
    melody_notes = []
    for note in xml_notes:
        if note.voice == 1 and not note.note_duration.is_grace_note:
            melody_notes.append(note)
    melody_notes = delete_chord_notes_for_melody(melody_notes)

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
        elif current_end < latest_start -  note.note_duration.duration * 4:
            break
    return latest_end


def calculate_mean_velocity(pairs):
    sum = 0
    length =0
    for pair in pairs:
        if not pair == []:
            sum += pair['midi'].velocity
            length += 1

    return sum/float(length)


def rearrange_chord_index(xml_notes):
    # assert all(xml_notes[i].pitch[1] >= xml_notes[i + 1].pitch[1] for i in range(len(xml_notes) - 1)
    #            if xml_notes[i].note_duration.xml_position ==xml_notes[i+1].note_duration.xml_position)

    previous_position = [-1]
    max_chord_index = [0]
    for note in xml_notes:
        voice = note.voice -1
        while voice >= len(previous_position):
            previous_position.append(-1)
            max_chord_index.append(0)
        if note.note_duration.is_grace_note:
            continue
        if note.staff ==1:
            if note.note_duration.xml_position > previous_position[voice]:
                previous_position[voice] = note.note_duration.xml_position
                max_chord_index[voice] = note.chord_index
                note.chord_index = 0
            else:
                note.chord_index = (max_chord_index[voice] - note.chord_index)
        else: #note staff ==2
            pass

    return xml_notes


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

def pitch_interval_into_vector(pitch_interval):
    vec_itv= [0, 0, 1] # [direction, octave, whether the next note exists]
    if pitch_interval == None:
        return [0] * 15
    elif pitch_interval > 0:
        vec_itv[0] = 1
    elif pitch_interval < 0:
        vec_itv[0] = -1
    else:
        vec_itv[0] = 0


    vec_itv[1] = abs(pitch_interval) // 12
    semiton_vec = [0] * 12
    semiton= abs(pitch_interval) % 12
    semiton_vec[semiton] = 1

    vec_itv += semiton_vec

    return vec_itv

def pitch_into_vector(pitch):
    pitch_vec = [0] * 13 #octave + pitch class
    octave = (pitch // 12) - 1
    octave = (octave - 4) / 4 # normalization
    pitch_class = pitch % 12

    pitch_vec[0] = octave
    pitch_vec[pitch_class+1] = 1

    return pitch_vec


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

def calculate_duration_ratio(xml_notes, index):
    search_index = 1
    num_notes = len(xml_notes)
    note = xml_notes[index]
    if note.note_duration.is_grace_note:
        return 0
    while index+search_index <num_notes:
        next_note = xml_notes[index+search_index]
        if next_note.voice == note.voice and next_note.chord_index == note.chord_index\
                and not next_note.note_duration.is_grace_note:
            # check whether the next note is directly following the previous note
            if note.note_duration.xml_position + note.note_duration.duration == next_note.note_duration.xml_position:
                return math.log(
                    xml_notes[index + search_index].note_duration.duration / xml_notes[index].note_duration.duration,
                    10)
            # the closes next note is too far from the note
            else:
                return 0

        search_index += 1
    # if index < len(xml_notes)-1:
    #     pitch_interval = xml_notes[index+1].pitch[1] - xml_notes[index].pitch[1]
    # else:
    #     pitch_interval = None
    # return pitch_interval
    return 0
    #
    # if index < len(xml_notes)-1:
    #     duration_ratio = math.log(xml_notes[index+1].note_duration.duration / xml_notes[index].note_duration.duration, 10)
    # else:
    #     duration_ratio = None
    # return duration_ratio

def cal_onset_deviation(xml_notes, melody_notes, melody_notes_onset_positions, pairs, index):
    # find previous closest melody index
    note = xml_notes[index]
    note_onset_position = note.note_duration.xml_position
    note_onset_time = pairs[index]['midi'].start
    corrsp_melody_index = binaryIndex(melody_notes_onset_positions, note_onset_position)
    corrsp_melody_note = melody_notes[corrsp_melody_index]
    xml_index = xml_notes.index(corrsp_melody_note)

    backward_search = 1
    # if the corresponding melody note has no MIDI note pair, search backward
    while pairs[xml_index]  == []:
        corrsp_melody_note = melody_notes[corrsp_melody_index-backward_search]
        xml_index = xml_notes.index(corrsp_melody_note)
        backward_search += 1

        if corrsp_melody_index - backward_search < 0:
            return 0

    previous_time_position = pairs[xml_index]['midi'].start
    previous_xml_position = pairs[xml_index]['xml'].note_duration.xml_position

    forward_search = 1
    if corrsp_melody_index + 1 == len(melody_notes):
        return 0
    next_melody_note = melody_notes[corrsp_melody_index+forward_search]
    xml_index_next = xml_notes.index(next_melody_note)
    while pairs[xml_index_next]  == []:
        next_melody_note = melody_notes[corrsp_melody_index+forward_search]
        xml_index_next = xml_notes.index(next_melody_note)
        forward_search += 1

        if corrsp_melody_index + forward_search == len(melody_notes):
            return 0
    next_time_position = pairs[xml_index_next]['midi'].start
    while next_time_position == previous_time_position:
        forward_search += 1
        if corrsp_melody_index + forward_search == len(melody_notes):
            return 0
        next_melody_note = melody_notes[corrsp_melody_index + forward_search]
        xml_index_next = xml_notes.index(next_melody_note)
        if not pairs[xml_index_next] == []:
            next_time_position = pairs[xml_index_next]['midi'].start
    next_xml_position =  pairs[xml_index_next]['xml'].note_duration.xml_position

    # calculate onset position (xml) of note with 'in tempo' circumstance
    onset_xml_position_in_tempo = previous_xml_position + (note_onset_time - previous_time_position) / (next_time_position - previous_time_position) * (next_xml_position - previous_xml_position)

    # onset_in_tempo = previous_time_position +  (note_onset_position - previous_xml_position) / (next_xml_position - previous_xml_position) * (next_time_position - previous_time_position)
    position_difference = onset_xml_position_in_tempo - note_onset_position
    if note.note_duration.is_grace_note:
        deviation = position_difference / note.following_note.note_duration.duration
    else:
        deviation = position_difference / note.note_duration.duration
    if math.isinf(deviation) or math.isnan(deviation):
        deviation = 0
    return deviation


def find_corresp_tempo(note, tempos):
    # note = pairs[index]['xml']
    tempo =  get_item_by_xml_position(tempos, note)
    # log_tempo = math.log(tempo, 10)
    # return log_tempo
    return tempo


def cal_articulation_with_tempo(pairs, i, tempo, trill_length):
    note = pairs[i]['xml']
    if note.note_duration.is_grace_note:
        return 0
    midi = pairs[i]['midi']
    xml_duration = note.note_duration.duration
    duration_as_quarter = xml_duration / note.state_fixed.divisions
    second_in_tempo = duration_as_quarter / tempo * 60
    if trill_length:
        actual_second = trill_length
    else:
        actual_second = midi.end - midi.start

    articulation = actual_second / second_in_tempo
    if articulation > 6:
        print('check: articulation is ' + str(articulation))
    articulation = math.log(articulation, 10)
    return articulation


def cal_onset_deviation_with_tempo(pairs, i, tempo_obj):
    note = pairs[i]['xml']
    midi = pairs[i]['midi']

    tempo_start = tempo_obj.time_position

    passed_duration = note.note_duration.xml_position - tempo_obj.xml_position
    actual_passed_second = midi.start - tempo_start
    actual_passed_duration = actual_passed_second / 60 * tempo_obj.qpm * note.state_fixed.divisions

    xml_pos_difference = actual_passed_duration - passed_duration
    pos_diff_in_quarter_note = xml_pos_difference / note.state_fixed.divisions
    deviation_time = xml_pos_difference / note.state_fixed.divisions / tempo_obj.qpm * 60

    if pos_diff_in_quarter_note >= 0:
        pos_diff_sqrt = math.sqrt(pos_diff_in_quarter_note)
    else:
        pos_diff_sqrt = -math.sqrt(-pos_diff_in_quarter_note)
    # pos_diff_cube_root = float(pos_diff_in_quarter_note) ** (1/3)
    return pos_diff_sqrt
    # return deviation_time, pos_diff_in_quarter_note


def cal_beat_importance(beat_position, numerator):
    # beat_position : [0-1), note's relative position in measure
    if beat_position == 0:
        beat_importance = 4
    elif beat_position == 0.5 and numerator in [2, 4, 6, 12]:
        beat_importance = 3
    elif abs(beat_position - (1/3)) < 0.001 and numerator in [3, 9]:
        beat_importance = 2
    elif (beat_position * 4) % 1 == 0  and numerator in [2, 4]:
        beat_importance = 1
    elif (beat_position * 5) % 1 == 0  and numerator in [5]:
        beat_importance = 2
    elif (beat_position * 6) % 1 == 0 and numerator in [3, 6, 12]:
        beat_importance = 1
    elif (beat_position * 8) % 1 == 0  and numerator in [2, 4]:
        beat_importance = 0.5
    elif (beat_position * 9) % 1 == 0 and numerator in [9]:
        beat_importance = 1
    elif (beat_position * 12) % 1 == 0 and numerator in [3, 6, 12]:
        beat_importance = 0.5
    else:
        beat_importance = 0
    return beat_importance


def get_item_by_xml_position(alist, item):
    if hasattr(item, 'xml_position'):
        item_pos = item.xml_position
    elif hasattr(item, 'note_duration'):
        item_pos = item.note_duration.xml_position
    elif hasattr(item, 'start_xml_position'):
        item_pos = item.start.xml_position
    else:
        item_pos = item

    repre = alist[0]

    if hasattr(repre, 'xml_position'):
        pos_list = [x.xml_position for x in alist]
    elif hasattr(repre, 'note_duration'):
        pos_list = [x.note_duration.xml_position for x in alist]
    elif hasattr(item, 'start_xml_position'):
        pos_list = [x.start_xml_position for x in alist]
    else:
        pos_list = alist

    index = binaryIndex(pos_list, item_pos)

    return alist[index]


def load_entire_subfolder(path):
    entire_pairs = []
    num_train_pairs = 0
    midi_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              f == 'midi_cleaned.mid']
    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        for valid_piece in VALID_LIST:
            if valid_piece in foldername:
                break
        else:
            xml_name = foldername + 'musicxml_cleaned.musicxml'

            if os.path.isfile(xml_name):
                print(foldername)
                piece_pairs = load_pairs_from_folder(foldername)
                if piece_pairs is not None:
                    entire_pairs.append(piece_pairs)
                    num_train_pairs += len(piece_pairs)


    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        for valid_piece in VALID_LIST:
            if valid_piece in foldername:
                xml_name = foldername + 'musicxml_cleaned.musicxml'

                if os.path.isfile(xml_name):
                    print(foldername)
                    piece_pairs = load_pairs_from_folder(foldername)
                    if piece_pairs is not None:
                        entire_pairs.append(piece_pairs)

    print('Number of train pairs: ', num_train_pairs)

    return entire_pairs, num_train_pairs


def load_pairs_from_folder(path):
    xml_name = path+'musicxml_cleaned.musicxml'
    score_midi_name = path+'midi_cleaned.mid'
    composer_name = copy.copy(path).split('/')[1]
    composer_name_vec = composer_name_to_vec(composer_name)

    XMLDocument = MusicXMLDocument(xml_name)
    xml_notes = extract_notes(XMLDocument, melody_only=False, grace_note=True)
    score_midi = midi_utils.to_midi_zero(score_midi_name)
    score_midi_notes = score_midi.instruments[0].notes
    score_midi_notes.sort(key=lambda x:x.start)
    match_list = matchXMLtoMIDI(xml_notes, score_midi_notes)
    score_pairs = make_xml_midi_pair(xml_notes, score_midi_notes, match_list)
    num_non_matched = check_pairs(score_pairs)
    if num_non_matched > 100:
        print("There are too many xml-midi matching errors")
        return None

    measure_positions = extract_measure_position(XMLDocument)
    filenames = os.listdir(path)
    perform_features_piece = []
    directions, time_signatures = extract_directions(XMLDocument)
    xml_notes = apply_directions_to_notes(xml_notes, directions, time_signatures)

    for file in filenames:
        if file[-18:] == '_infer_corresp.txt':
            perf_name = file.split('_infer')[0]
            # perf_score = evaluation.cal_score(perf_name)
            perf_score = 0

            perf_midi_name = path + perf_name + '.mid'
            perf_midi = midi_utils.to_midi_zero(perf_midi_name)
            perf_midi = midi_utils.add_pedal_inf_to_notes(perf_midi)
            perf_midi_notes= perf_midi.instruments[0].notes
            corresp_name = path + file
            corresp = read_corresp(corresp_name)

            xml_perform_match = match_score_pair2perform(score_pairs, perf_midi_notes, corresp)
            perform_pairs = make_xml_midi_pair(xml_notes, perf_midi_notes, xml_perform_match)
            print("performance name is " + perf_name)
            num_align_error = check_pairs(perform_pairs)
            if num_align_error > 1000:
                print('Too many align error in the performance')
                continue
            perform_features = extract_perform_features(XMLDocument, xml_notes, perform_pairs, perf_midi_notes, measure_positions)
            perform_feat_score = {'features': perform_features, 'score': perf_score, 'composer':composer_name_vec}

            perform_features_piece.append(perform_feat_score)

    if perform_features_piece == []:
        return None
    return perform_features_piece


def make_midi_measure_seconds(pairs, measure_positions):
    xml_positions = []
    pair_indexes = []
    for i in range(len(pairs)):
        if not pairs[i] == []:
            xml_positions.append(pairs[i]['xml'].note_duration.xml_position)
            pair_indexes.append(i)
    # xml_positions = [pair['xml'].note_duration.xml_position for pair in pairs]
    measure_seconds = []
    for measure_start in measure_positions:
        pair_index = pair_indexes[binaryIndex(xml_positions, measure_start)]
        if pairs[pair_index]['xml'].note_duration.xml_position == measure_start:
            measure_seconds.append(pairs[pair_index]['midi'].start)
        else:
            left_second = pairs[pair_index]['midi'].start
            left_position = pairs[pair_index]['xml'].note_duration.xml_position
            while pair_index+1<len(pairs) and pairs[pair_index+1] == []:
                pair_index += 1
            if pair_index+1 == len(pairs):
                measure_seconds.append(max(measure_seconds[-1], left_second ))
                continue
            right_second = pairs[pair_index+1]['midi'].start
            right_position = pairs[pair_index+1]['xml'].note_duration.xml_position

            interpolated_second = left_second + (measure_start-left_position) / (right_position-left_position) * (right_second-left_second)
            measure_seconds.append(interpolated_second)
    return measure_seconds

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



def applyIOI(xml_notes, midi_notes, features, feature_list):
    IOIratio = feature_list[0]
    articulation = feature_list[1]
    loudness = feature_list[2]
    # time_deviation = feature_list[3]

    #len(features) always equal to len(xml_notes) by its definition
    xml_ioi_ratio_pairs = []
    ioi_index =0
    if not len(xml_notes) == len(IOIratio):
        for i in range(len(features)):
            if not features[i]['IOI_ratio'] == None:
                # [xml_position, time_position, ioi_ratio]
                temp_pair = {'xml_pos': xml_notes[i].note_duration.xml_position, 'midi_pos' : xml_notes[i].note_duration.time_position, 'ioi': IOIratio[ioi_index]}
                xml_ioi_ratio_pairs.append(temp_pair)
                ioi_index += 1
        if not ioi_index  == len(IOIratio):
            print('check ioi_index', ioi_index)
    else:
        for i in range(len(xml_notes)):
            note = xml_notes[i]
            temp_pair = {'xml_pos': note.note_duration.xml_position, 'midi_pos' : note.note_duration.time_position, 'ioi': IOIratio[i]}
            xml_ioi_ratio_pairs.append(temp_pair)

    default_tempo = xml_notes[0].state_fixed.qpm / 60 * xml_notes[0].state_fixed.divisions
    default_velocity = 64

    # in case the xml_position of first note is not 0
    current_sec = (xml_ioi_ratio_pairs[0]['xml_pos'] - 0) / default_tempo
    tempo_mapping_list = [ [xml_ioi_ratio_pairs[0]['midi_pos'] ] , [current_sec]]
    for i in range(len(xml_ioi_ratio_pairs)-1):
        pair = xml_ioi_ratio_pairs[i]
        next_pair = xml_ioi_ratio_pairs[i+1]
        note_length = next_pair['xml_pos'] - pair['xml_pos']
        tempo_ratio =  1/ (10 ** pair['ioi'])
        tempo = default_tempo * tempo_ratio
        note_length_second = note_length / tempo
        next_sec = current_sec + note_length_second
        current_sec = next_sec
        tempo_mapping_list[0].append( next_pair['midi_pos'] )
        tempo_mapping_list[1].append( current_sec )

    for note in midi_notes:
        note = make_new_note(note, tempo_mapping_list[0], tempo_mapping_list[1], articulation, loudness, default_velocity)
    return midi_notes

def apply_perform_features(xml_notes, features):
    melody_notes = extract_melody_only_from_notes(xml_notes)
    default_tempo = xml_notes[0].state_fixed.qpm / 60 * xml_notes[0].state_fixed.divisions
    previous_ioi = 0
    current_sec = 0
    between_notes = find_notes_between_melody_notes(xml_notes, melody_notes)
    num_melody_notes = len(melody_notes)
    prev_vel = 64



    for i in range(num_melody_notes):
        note = melody_notes[i]
        xml_index = xml_notes.index(note)
        if features[xml_index]['IOI_ratio'] == None:
            ioi = previous_ioi
        else:
            ioi = features[xml_index]['IOI_ratio']
        tempo_ratio = 1 / (10 ** ioi)
        tempo = default_tempo * tempo_ratio

        note.note_duration.time_position = current_sec
        note.note_duration.seconds = note.note_duration.duration / tempo

        note, prev_vel = apply_feat_to_a_note(note, features[xml_index], prev_vel)

        num_between_notes = len(between_notes[i])
        # for j in range(-1, -num_between_notes+1, -1):
        for j in range(num_between_notes):
            betw_note = between_notes[i][j]
            betw_index = xml_notes.index(betw_note)

            if not features[betw_index]['xml_deviation'] == None:
                dur = betw_note.note_duration.duration
                if betw_note.note_duration.is_grace_note:
                    dur = betw_note.following_note.note_duration.duration
                betw_note.note_duration.xml_position += features[betw_index]['xml_deviation'] * dur

            passed_duration = betw_note.note_duration.xml_position - note.note_duration.xml_position
            passed_second = passed_duration / tempo
            betw_note.note_duration.time_position = current_sec + passed_second
            # betw_note.note_duration.seconds = betw_note.note_duration.duration / tempo

            betw_note, prev_vel = apply_feat_to_a_note(betw_note, features[betw_index], prev_vel)
        for j in range(num_between_notes):
            betw_note = between_notes[i][j]
            if betw_note.note_duration.is_grace_note:
                betw_note.note_duration.seconds = (betw_note.following_note.note_duration.time_position
                                                   - betw_note.note_duration.time_position)\
                                                  * betw_note.note_duration.grace_order
        if not i == num_melody_notes-1:
            duration_to_next = melody_notes[i + 1].note_duration.xml_position - melody_notes[i].note_duration.xml_position  # there can be a rest
            second_to_next_note = duration_to_next / tempo
            next_sec = current_sec + second_to_next_note
            current_sec = next_sec

    # after the new time_position of melody notes are fixed, calculate offset of btw notes
    xml_time_pairs = {'beat': [note.note_duration.xml_position for note in melody_notes],
                      'time': [note.note_duration.time_position for note in melody_notes]}
    for i in range(num_melody_notes):
        num_between_notes = len(between_notes[i])
        for j in range(num_between_notes):
            betw_note = between_notes[i][j]
            if betw_note.note_duration.is_grace_note:
                continue
            note_offset_beat = betw_note.note_duration.xml_position + betw_note.note_duration.duration
            offset_index = binaryIndex(xml_time_pairs['beat'], note_offset_beat)

            while offset_index + 1 >= num_melody_notes:
                offset_index += -1
            tempo = (xml_time_pairs['beat'][offset_index+1] - xml_time_pairs['beat'][offset_index]) \
                    / (xml_time_pairs['time'][offset_index+1] - xml_time_pairs['time'][offset_index])

            exceed_beat = note_offset_beat - xml_time_pairs['beat'][offset_index]
            if not math.isinf(tempo):
                offset_time = xml_time_pairs['time'][offset_index] + exceed_beat / tempo
            else:
                offset_time = xml_time_pairs['time'][offset_index]

            betw_note.note_duration.seconds = max(offset_time - betw_note.note_duration.time_position, 0.05)

    return xml_notes

def apply_tempo_perform_features(xml_doc, xml_notes, features, start_time=0, predicted=False):
    beats = cal_beat_positions_of_piece(xml_doc)
    num_beats = len(beats)
    num_notes = len(xml_notes)
    tempos=[]
    ornaments = []
    # xml_positions = [x.note_duration.xml_position for x in xml_notes]
    prev_vel = 64
    previous_position = None
    current_sec = start_time
    key_signatures = xml_doc.get_key_signatures()
    trill_accidentals = extract_accidental(xml_doc)

    valid_notes = make_available_note_feature_list(xml_notes, features, predicted=predicted)
    previous_tempo = 0

    for i in range(num_beats - 1):
        beat = beats[i]
        # note_index = binaryIndex(xml_positions, beat)
        # if previous_position not in beats:
        #     btw_index = binaryIndex(previous_position, beat) + 1
        #     btw_note = xml_notes[note_index]
        #     while xml_notes[btw_index+1].note_duration.xml_position == btw_note.note_duration.xml_position \
        #         and btw_index + 1 < num_notes:
        #         btw_index += 1
        #         btw_note = xml_notes[note_index]
        #
        #
        # while features[note_index]['qpm'] == None and note_index > 0:
        #     note_index += -1
        # while (xml_notes[note_index].note_duration.xml_position == previous_position
        #         or features[note_index]['qpm'] == None) and note_index +1 < num_notes :
        #     note_index += 1
        # note = xml_notes[note_index]
        # start_position = note.note_duration.xml_position
        # qpm = 10 ** features[note_index]['qpm']
        feat = get_item_by_xml_position(valid_notes, beat)
        start_position = feat.xml_position
        if start_position == previous_position:
            continue

        if predicted:
            qpm_saved = 10 ** feat.qpm
            num_added = 1
            next_beat = beats[i+1]
            start_index = feat.index

            for j in range(1,20):
                if start_index-j < 0:
                    break
                previous_note = xml_notes[start_index-j]
                previous_pos = previous_note.note_duration.xml_position
                if previous_pos == start_position:
                    qpm_saved += 10 ** features[start_index-j].qpm
                    num_added += 1
                else:
                    break

            for j in range(1,40):
                if start_index + j >= num_notes:
                    break
                next_note = xml_notes[start_index+j]
                next_position = next_note.note_duration.xml_position
                if next_position < next_beat:
                    qpm_saved += 10 ** features[start_index+j].qpm
                    num_added += 1
                else:
                    break

            qpm = qpm_saved / num_added
        else:
            qpm = 10 ** feat.qpm
        # qpm = 10 ** feat.qpm
        divisions = feat.divisions

        if previous_tempo != 0:
            passed_second = (start_position - previous_position) / divisions / previous_tempo * 60
        else:
            passed_second = 0
        current_sec += passed_second
        tempo = Tempo(start_position, qpm, time_position=current_sec, end_xml=0 ,end_time=0)
        if len(tempos) > 0:
            tempos[-1].end_time = current_sec
            tempos[-1].end_xml = start_position

        tempos.append(tempo)

        previous_position = start_position
        previous_tempo = qpm

    def cal_time_position_with_tempo(note, xml_dev, tempos):
        corresp_tempo = get_item_by_xml_position(tempos, note)
        previous_sec = corresp_tempo.time_position
        passed_duration = note.note_duration.xml_position + xml_dev - corresp_tempo.xml_position
        passed_second = passed_duration / note.state_fixed.divisions / corresp_tempo.qpm * 60

        return previous_sec + passed_second

    for i in range(num_notes):
        note = xml_notes[i]
        feat = features[i]
        if not feat.xml_deviation == None:
            xml_deviation = (feat.xml_deviation ** 2) * note.state_fixed.divisions
        else:
            xml_deviation = 0

        note.note_duration.time_position = cal_time_position_with_tempo(note, xml_deviation, tempos)

        # if not feat['xml_deviation'] == None:
        #     note.note_duration.time_position += feat['xml_deviation']

        end_note = copy.copy(note)
        end_note.note_duration = copy.copy(note.note_duration)
        end_note.note_duration.xml_position = note.note_duration.xml_position + note.note_duration.duration

        end_position = cal_time_position_with_tempo(end_note, 0, tempos)
        if note.note_notations.is_trill:
            note, _ = apply_feat_to_a_note(note, feat, prev_vel)
            trill_vec = feat.trill_param
            num_trills = trill_vec[0]
            if not isinstance(num_trills, int):
                num_trills = num_trills.astype(int)
            last_velocity = trill_vec[1] * note.velocity
            first_note_ratio = trill_vec[2]
            last_note_ratio = trill_vec[3]
            up_trill = trill_vec[4]
            total_second = end_position - note.note_duration.time_position
            first_velocity = note.velocity

            key = get_item_by_xml_position(key_signatures, note)
            key = key.key
            final_key = None
            for acc in trill_accidentals:
                if acc.xml_position == note.note_duration.xml_position:
                    if acc.type['content'] == '#':
                        final_key = 7
                    elif acc.type['content'] == '♭':
                        final_key = -7
                    elif acc.type['content'] == '♮':
                        final_key = 0
            measure_accidentals = get_measure_accidentals(xml_notes, i)
            trill_pitch = note.pitch
            up_pitch, up_pitch_string = cal_up_trill_pitch(note.pitch, key, final_key, measure_accidentals)

            if up_trill:
                up = True
            else:
                up = False

            if num_trills > 2:
                mean_second = total_second / num_trills
                normal_second = (total_second - mean_second * (first_note_ratio + last_note_ratio)) / (num_trills -2)
                prev_end = note.note_duration.time_position
                for j in range(num_trills):
                    if up:
                        pitch = (up_pitch_string, up_pitch)
                        up = False
                    else:
                        pitch = trill_pitch
                        up = True
                    if j == 0:
                        note.pitch = pitch
                        note.note_duration.seconds = mean_second * first_note_ratio
                        prev_end += mean_second * first_note_ratio
                    else:
                        new_note = copy.copy(note)
                        new_note.pitch = copy.copy(note.pitch)
                        new_note.pitch = pitch
                        new_note.note_duration = copy.copy(note.note_duration)
                        new_note.note_duration.time_position = prev_end
                        if j == num_trills -1:
                            new_note.note_duration.seconds = mean_second * last_note_ratio
                        else:
                            new_note.note_duration.seconds = normal_second
                        new_note.velocity = copy.copy(note.velocity)
                        new_note.velocity = first_velocity + (last_velocity - first_velocity) * (j / num_trills)
                        prev_end += new_note.note_duration.seconds
                        ornaments.append(new_note)
            elif num_trills == 2:
                mean_second = total_second / num_trills
                prev_end = note.note_duration.time_position
                for j in range(2):
                    if up:
                        pitch = (up_pitch_string, up_pitch)
                        up = False
                    else:
                        pitch = trill_pitch
                        up = True
                    if j == 0:
                        note.pitch = pitch
                        note.note_duration.seconds = mean_second * first_note_ratio
                        prev_end += mean_second * first_note_ratio
                    else:
                        new_note = copy.copy(note)
                        new_note.pitch = copy.copy(note.pitch)
                        new_note.pitch = pitch
                        new_note.note_duration = copy.copy(note.note_duration)
                        new_note.note_duration.time_position = prev_end
                        new_note.note_duration.seconds = mean_second * last_note_ratio
                        new_note.velocity = copy.copy(note.velocity)
                        new_note.velocity = last_velocity
                        prev_end += mean_second * last_note_ratio
                        ornaments.append(new_note)
            else:
                note.note_duration.seconds = total_second


        else:
            note.note_duration.seconds = end_position - note.note_duration.time_position

        note, prev_vel = apply_feat_to_a_note(note, feat, prev_vel)

    for i in range(num_notes):
        note = xml_notes[i]
        feat = features[i]

        if note.note_duration.is_grace_note and note.note_duration.duration == 0:
            following_note = note.following_note
            next_second = following_note.note_duration.time_position
            note.note_duration.seconds = (next_second - note.note_duration.time_position) / note.note_duration.num_grace

    xml_notes = xml_notes + ornaments
    xml_notes.sort(key=lambda x: (x.note_duration.xml_position, x.note_duration.time_position, -x.pitch[1]) )
    return xml_notes

def apply_time_position_features(xml_notes, features, start_time=0, include_unmatched=True):
    num_notes = len(xml_notes)
    tempos = []
    # xml_positions = [x.note_duration.xml_position for x in xml_notes]
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

        # if not feat['xml_deviation'] == None:
        #     note.note_duration.time_position += feat['xml_deviation']

        note.note_duration.seconds = feat.duration_second
        feat.articulation = 1
        note, prev_vel = apply_feat_to_a_note(note, feat, prev_vel)

        valid_notes.append(note)

    # for i in range(num_notes):
    #     note = xml_notes[i]
    #     feat = features[i]
    #
    #     if note.note_duration.is_grace_note:
    #         followed_note = note.followed_note
    #         next_second = followed_note.note_duration.time_position
    #         note.note_duration.seconds = (next_second - note.note_duration.time_position) / note.note_duration.num_grace

    return valid_notes


def make_available_note_feature_list(notes, features, predicted):
    class PosTempoPair:
        def __init__(self, xml_pos, pitch, qpm, index, divisions, time_pos):
            self.xml_position = xml_pos
            self.qpm = qpm
            self.index = index
            self.divisions = divisions
            self.pitch = pitch
            self.time_position = time_pos
            self.is_arpeggiate = False

    if not predicted:
        available_notes = []
        num_features = len(features)
        for i in range(num_features):
            feature = features[i]
            if not feature.qpm == None:
                xml_note = notes[i]
                xml_pos = xml_note.note_duration.xml_position
                time_pos = feature.midi_start
                divisions = xml_note.state_fixed.divisions
                qpm = feature.qpm
                pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair.is_arpeggiate = True
                available_notes.append(pos_pair)

    else:
        available_notes = []
        num_features = len(features)
        for i in range(num_features):
            feature = features[i]
            xml_note = notes[i]
            xml_pos = xml_note.note_duration.xml_position
            time_pos = xml_note.note_duration.time_position
            divisions = xml_note.state_fixed.divisions
            qpm = feature.qpm
            pos_pair = PosTempoPair(xml_pos, xml_note.pitch[1], qpm, i, divisions, time_pos)
            available_notes.append(pos_pair)

    # minimum_time_interval = 0.05
    # available_notes = save_lowest_note_on_same_position(available_notes, minimum_time_interval)
    available_notes, _ = make_average_onset_cleaned_pair(available_notes)
    return available_notes


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


def make_average_onset_cleaned_pair(position_pairs, maximum_qpm=500):
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
        current_position = pos_pair.xml_position
        current_time = pos_pair.time_position
        if current_position > previous_position and previous_position >= 0:
            minimum_time_interval = (current_position - previous_position) / pos_pair.divisions / maximum_qpm * 60 + 0.01
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair.time_position = get_average_onset_time(notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair.time_position > cleaned_list[-1].time_position + minimum_time_interval:
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            mismatched_indexes.append(note.index)
                else:
                    for note in notes_in_chord:
                        mismatched_indexes.append(note.index)
            notes_in_chord = list()
            notes_in_chord.append(pos_pair)
            previous_position = current_position
            previous_time = current_time
            previous_index = i
        elif current_position == previous_position:
            notes_in_chord.append(pos_pair)
        else:
            # print(previous_position, current_position, previous_time, current_time)
            mismatched_indexes.append(position_pairs[previous_index].index)
            mismatched_indexes.append(pos_pair.index)

    return cleaned_list, mismatched_indexes


# def check_midi_alignment_continuity(pairs):
#     previous_positino = -1
#     previous_time = -1
#
#     for pair in pairs:
#         if not pair == []:
#             xml_note = pair['xml']
#             midi_note = pair['midi']
#             current_position = xml_note.note_duration.xml_position
#             current_time = midi_note.start
#
#             if current_position > previous_position:
#
#     return


def get_average_onset_time(notes_in_chord_saved, threshold=0.2):
    # notes_in_chord: list of PosTempoPair, len > 0
    notes_in_chord = copy.copy(notes_in_chord_saved)
    average_onset_time = 0
    for pos_pair in notes_in_chord:
        average_onset_time += pos_pair.time_position
        if pos_pair.is_arpeggiate:
            threshold = 1
    average_onset_time /= len(notes_in_chord)

    # check whether there is mis-matched note
    deviations = list()
    for pos_pair in notes_in_chord:
        dev = abs(pos_pair.time_position - average_onset_time)
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
        melody_index = binaryIndex(melody_onset_positions, note.note_duration.xml_position)
        between_notes[melody_index].append(note)

    return between_notes

def apply_feat_to_a_note(note, feat, prev_vel):

    if not feat.articulation == None:
        note.note_duration.seconds *= 10 ** (feat.articulation)
    if not feat.velocity == None:
        note.velocity = feat.velocity
        prev_vel = note.velocity
    else:
        note.velocity = prev_vel
    if not feat.pedal_at_start == None:
        # note.pedal.at_start = feat['pedal_at_start']
        # note.pedal.at_end = feat['pedal_at_end']
        # note.pedal.refresh = feat['pedal_refresh']
        # note.pedal.refresh_time = feat['pedal_refresh_time']
        # note.pedal.cut = feat['pedal_cut']
        # note.pedal.cut_time = feat['pedal_cut_time']
        # note.pedal.soft = feat['soft_pedal']
        note.pedal.at_start = int(round(feat.pedal_at_start))
        note.pedal.at_end = int(round(feat.pedal_at_end))
        note.pedal.refresh = int(round(feat.pedal_refresh))
        note.pedal.refresh_time = feat.pedal_refresh_time
        note.pedal.cut = int(round(feat.pedal_cut))
        note.pedal.cut_time = feat.pedal_cut_time
        note.pedal.soft = int(round(feat.soft_pedal))
    return note, prev_vel

def make_new_note(note, time_a, time_b, articulation, loudness, default_velocity):
    index = binaryIndex(time_a, note.start)
    new_onset = cal_new_onset(note.start, time_a, time_b)
    temp_offset = cal_new_onset(note.end, time_a, time_b)
    new_duration = (temp_offset-new_onset) * articulation[index]
    new_offset = new_onset + new_duration
    # new_velocity = min([max([int(default_velocity * (10 ** loudness[index])) , 0]), 127])
    new_velocity = min([max( int(loudness[index]), 0), 127])

    note.start= new_onset
    note.end = new_offset
    note.velocity = new_velocity

    return note

def cal_new_onset(note_start, time_a, time_b):
    index = binaryIndex(time_a, note_start)
    time_org = time_a[index]
    new_time = interpolation(note_start, time_a, time_b, index)

    return new_time



def interpolation(a, list1, list2, index):
    if index == len(list1)-1:
        index += -1

    a1 = list1[index]
    b1 = list2[index]
    a2 = list1[index+1]
    b2 = list2[index+1]
    return b1+ (a-a1) / (a2-a1) * (b2 - b1)


def save_midi_notes_as_piano_midi(midi_notes, output_name, bool_pedal=False, disklavier=False):
    piano_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    for note in midi_notes:
        piano.notes.append(note)

    piano_midi.instruments.append(piano)

    piano_midi = midi_utils.save_note_pedal_to_CC(piano_midi)
    if bool_pedal:
        pedals = piano_midi.instruments[0].control_changes
        for pedal in pedals:
            if pedal.value < 30:
                pedal.value = 0

    if disklavier:
        pedals = piano_midi.instruments[0].control_changes
        pedals.sort(key=lambda x:x.time)
        previous_off_time = 0
        for pedal in pedals:
            if pedal.time <0.3:
                continue
            if pedal.value < 40:
                previous_off_time = pedal.time
            else:
                time_passed = pedal.time - previous_off_time
                if time_passed < 0.25:
                    pedals.remove(pedal)
        piano_midi.instruments[0].control_changes = pedals
    piano_midi.write(output_name)


def extract_directions(xml_doc):
    directions = []
    for part in xml_doc.parts:
        for measure in part.measures:
            for direction in measure.directions:
                directions.append(direction)

    directions.sort(key=lambda x: x.xml_position)
    cleaned_direction = []
    for i in range(len(directions)):
        dir = directions[i]
        if not dir.type == None:
            if dir.type['type'] == "none":
                for j in range(i):
                    prev_dir = directions[i-j-1]
                    if 'number' in prev_dir.type.keys():
                        prev_key = prev_dir.type['type']
                        prev_num = prev_dir.type['number']
                    else:
                        continue
                    if prev_num == dir.type['number']:
                        if prev_key == "crescendo":
                            dir.type['type'] = 'crescendo'
                            break
                        elif prev_key == "diminuendo":
                            dir.type['type'] = 'diminuendo'
                            break
            cleaned_direction.append(dir)
        else:
            print(vars(dir.xml_direction))

    time_signatures = xml_doc.get_time_signatures()
    return cleaned_direction, time_signatures


def merge_start_end_of_direction(directions):
    for i in range(len(directions)):
        dir = directions[i]
        type_name = dir.type['type']
        if type_name in ['crescendo', 'diminuendo', 'pedal'] and dir.type['content'] == "stop":
            for j in range(i):
                prev_dir = directions[i-j-1]
                prev_type_name = prev_dir.type['type']
                if type_name == prev_type_name and prev_dir.type['content'] == "start" and dir.staff == prev_dir.staff:
                    prev_dir.end_xml_position = dir.xml_position
                    break
    dir_dummy = []
    for dir in directions:
        type_name = dir.type['type']
        if type_name in ['crescendo', 'diminuendo', 'pedal'] and dir.type['content'] != "stop":
            # directions.remove(dir)
            dir_dummy.append(dir)
        elif type_name == 'words':
            dir_dummy.append(dir)
    directions = dir_dummy
    return directions


def apply_directions_to_notes(xml_notes, directions, time_signatures):
    absolute_dynamics, relative_dynamics, cresciutos = get_dynamics(directions)
    absolute_dynamics_position = [dyn.xml_position for dyn in absolute_dynamics]
    absolute_tempos, relative_tempos = get_tempos(directions)
    absolute_tempos_position = [tmp.xml_position for tmp in absolute_tempos]
    time_signatures_position = [time.xml_position for time in time_signatures]

    num_dynamics = len(absolute_dynamics)
    num_tempos = len(absolute_tempos)

    for note in xml_notes:
        note_position = note.note_duration.xml_position

        if num_dynamics > 0:
            index = binaryIndex(absolute_dynamics_position, note_position)
            note.dynamic.absolute = absolute_dynamics[index].type['content']
            note.dynamic.absolute_position = absolute_dynamics[index].xml_position

        if num_tempos > 0:
            tempo_index = binaryIndex(absolute_tempos_position, note_position)
        # note.tempo.absolute = absolute_tempos[tempo_index].type[absolute_tempos[tempo_index].type.keys()[0]]
            note.tempo.absolute = absolute_tempos[tempo_index].type['content']
            note.tempo.recently_changed_position = absolute_tempos[tempo_index].xml_position
        time_index = binaryIndex(time_signatures_position, note_position)
        note.tempo.time_numerator = time_signatures[time_index].numerator
        note.tempo.time_denominator = time_signatures[time_index].denominator

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
                continue
            if note_position < cresc.end_xml_position:
                note_cresciuto = note.dynamic.cresciuto
                if note_cresciuto is None:
                    note.dynamic.cresciuto = cresc
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

def extract_directions_by_keywords(directions, keywords):
    sub_directions =[]

    for dir in directions:
        included = check_direction_by_keywords(dir, keywords)
        if included:
            sub_directions.append(dir)
            # elif dir.type['words'].split('sempre ')[-1] in keywords:
            #     dir.type['dynamic'] = dir.type.pop('words')
            #     dir.type['dynamic'] = dir.type['dynamic'].split('sempre ')[-1]
            #     sub_directions.append(dir)
            # elif dir.type['words'].split('subito ')[-1] in keywords:
            #     dir.type['dynamic'] = dir.type.pop('words')
            #     dir.type['dynamic'] = dir.type['dynamic'].split('subito ')[-1]
            #     sub_directions.append(dir)

    return sub_directions


def check_direction_by_keywords(dir, keywords):
    if dir.type['type'] in keywords:
        return True
    elif dir.type['type'] == 'words':
        if dir.type['content'].replace(',', '').replace('.', '').lower() in keywords:
            return True
        else:
            word_split = dir.type['content'].replace(',', ' ').replace('.', ' ').split(' ')
            for w in word_split:
                if w.lower() in keywords:
                    # dir.type[keywords[0]] = dir.type.pop('words')
                    # dir.type[keywords[0]] = w
                    return True

        for key in keywords: # words like 'sempre più mosso'
            if len(key) > 2 and key in dir.type['content']:
                return True


def get_dynamics(directions):
    temp_abs_key = absolute_dynamics_keywords
    temp_abs_key.append('dynamic')

    absolute_dynamics = extract_directions_by_keywords(directions, temp_abs_key)
    relative_dynamics = extract_directions_by_keywords(directions, relative_dynamics_keywords)
    abs_dynamic_dummy = []
    for abs in absolute_dynamics:
        if abs.type['content'] == 'fp':
            abs.type['content'] = 'f sfz'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)
        elif abs.type['content'] == 'ffp':
            abs.type['content'] = 'ff sfz'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)
        elif abs.type['content'] == 'sfp':
            abs.type['content'] = 'sf'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)

        if abs.type['content'] in ['sf', 'fz', 'sfz', 'sffz', 'rf', 'rfz']:
            relative_dynamics.append(abs)
        else:
            abs_dynamic_dummy.append(abs)


    absolute_dynamics = abs_dynamic_dummy

    relative_dynamics.sort(key=lambda x:x.xml_position)
    relative_dynamics = merge_start_end_of_direction(relative_dynamics)
    absolute_dynamics_position = [dyn.xml_position for dyn in absolute_dynamics]
    relative_dynamics_position = [dyn.xml_position for dyn in relative_dynamics]
    cresc_name = ['crescendo', 'diminuendo']
    cresciuto_list = []
    class Cresciuto:
        def __init__(self, start, end, type):
            self.xml_position = start
            self.end_xml_position = end
            self.type = type #crescendo or diminuendo
            self.overlapped = 0

    for rel in relative_dynamics:
        index = binaryIndex(absolute_dynamics_position, rel.xml_position)
        rel.previous_dynamic = absolute_dynamics[index].type['content']
        if rel.type['type'] == 'dynamic' and not rel.type['content'] == 'rf': # sf, fz, sfz
            rel.end_xml_position = rel.xml_position + 0.1
        if index+1 < len(absolute_dynamics):
            rel.next_dynamic = absolute_dynamics[index+1] #.type['content']
            if not hasattr(rel, 'end_xml_position'):
                rel.end_xml_position = absolute_dynamics_position[index+1]
        else:
            rel.next_dynamic = absolute_dynamics[index]
        if not hasattr(rel, 'end_xml_position'):
            rel.end_xml_position = float("inf")

        if rel.type['type'] in cresc_name and rel.end_xml_position < rel.next_dynamic.xml_position:
            cresciuto = Cresciuto(rel.end_xml_position, rel.next_dynamic.xml_position, rel.type['type'])
            cresciuto_list.append(cresciuto)
    return absolute_dynamics, relative_dynamics, cresciuto_list


def get_tempos(directions):

    absolute_tempos = extract_directions_by_keywords(directions, absolute_tempos_keywords)
    relative_tempos = extract_directions_by_keywords(directions, relative_tempos_keywords)
    relative_long_tempos = extract_directions_by_keywords(directions, relative_long_tempo_keywords)

    absolute_tempos_position = [tmp.xml_position for tmp in absolute_tempos]
    num_abs_tempos = len(absolute_tempos)
    num_rel_tempos = len(relative_tempos)

    for abs in absolute_tempos:
        for wrd in tempo_primo_words:
            if wrd in abs.type['content'].lower():
                abs.type['content'] = absolute_tempos[0].type['content']

    for i in range(num_rel_tempos):
        rel = relative_tempos[i]
        if rel not in relative_long_tempos and i+1< num_rel_tempos:
            rel.end_xml_position = relative_tempos[i+1].xml_position
        index = binaryIndex(absolute_tempos_position, rel.xml_position)
        rel.previous_tempo = absolute_tempos[index].type['content']
        if index+1 < num_abs_tempos:
            rel.next_tempo = absolute_tempos[index+1].type['content']
            if not hasattr(rel, 'end_xml_position') or rel.end_xml_position > absolute_tempos_position[index+1]:
                rel.end_xml_position = absolute_tempos_position[index+1]
        if not hasattr(rel, 'end_xml_position'):
            rel.end_xml_position = float("inf")


    return absolute_tempos, relative_tempos


def get_all_words_from_folders(path):
    entire_words = []
    xml_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              f == 'musicxml_cleaned.musicxml']

    for xmlfile in xml_list:
        print(xmlfile)
        xml_doc = MusicXMLDocument(xmlfile)
        directions, _ = extract_directions(xml_doc)

        words = [dir for dir in directions if dir.type['type'] == 'words']

        for wrd in words:
            entire_words.append(wrd.type['content'])
            # print(wrd.type['content'], wrd.state.qpm)

    entire_words = list(set(entire_words))
    return entire_words

def keyword_into_onehot(attribute, keywords):
    one_hot = [0] * len(keywords)
    if attribute == None:
        return one_hot
    if attribute in keywords:
        index = find_index_list_of_list(attribute, keywords)
        one_hot[index] = 1


    # for i in range(len(keywords)):
    #     keys = keywords[i]
    #     if type(keys) is list:
    #         for key in keys:
    #             if len(key)>2 and (key.encode('utf-8') in
    word_split = attribute.replace(',', ' ').replace('.', ' ').split(' ')
    for w in word_split:
        index = find_index_list_of_list(w.lower(), keywords)
        if index:
            one_hot[index] = 1

    for key in keywords:

        if isinstance(key, str) and len(key) > 2 and key in attribute:
        # if type(key) is st and len(key) > 2 and key in attribute:
            index = keywords.index(key)
            one_hot[index] = 1

    return one_hot


def direction_words_flatten(note_attribute):
    flatten_words = note_attribute.absolute
    if not note_attribute.relative == []:
        for rel in note_attribute.relative:
            if rel.type['type'] == 'words':
                flatten_words = flatten_words + ' ' + rel.type['content']
            else:
                flatten_words = flatten_words + ' ' + rel.type['type']
    return flatten_words

def find_index_list_of_list(element, in_list):
    # isuni = isinstance(element, unicode) # for python 2.7
    if element in in_list:
        return in_list.index(element)
    else:
        for li in in_list:
            if isinstance(li, list):
                # if isuni:
                #     li = [x.decode('utf-8') for x in li]
                if element in li:
                    return in_list.index(li)

    return None


def note_notation_to_vector(note):
    # trill, tenuto, accent, staccato, fermata
    keywords = ['is_trill', 'is_tenuto', 'is_accent', 'is_staccato', 'is_fermata', 'is_arpeggiate', 'is_strong_accent', 'is_cue']
    # keywords = ['is_trill', 'is_tenuto', 'is_accent', 'is_staccato', 'is_fermata']

    notation_vec = [0] * len(keywords)

    for i in range(len(keywords)):
        key = keywords[i]
        if getattr(note.note_notations, key) == True:
            notation_vec[i] = 1

    return notation_vec

def apply_repetition(xml_notes, xml_doc):
    pass

# a. after applying_directions, apply repetition
# b. apply repetition at parsing stage.


def read_repetition(xml_doc):
    pass

def xml_notes_to_midi(xml_notes):
    midi_notes = []
    for note in xml_notes:
        if note.is_overlapped: # ignore overlapped notes.
            continue

        pitch = note.pitch[1]
        start = note.note_duration.time_position
        end = start + note.note_duration.seconds
        if note.note_duration.seconds <0.005:
            end = start + 0.005
        elif note.note_duration.seconds > 10:
            end = start + 10
        velocity = int(min(max(note.velocity,0),127))
        midi_note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)

        midi_note.pedal_at_start = note.pedal.at_start
        midi_note.pedal_at_end = note.pedal.at_end
        midi_note.pedal_refresh = note.pedal.refresh
        midi_note.pedal_refresh_time = note.pedal.refresh_time
        midi_note.pedal_cut = note.pedal.cut
        midi_note.pedal_cut_time = note.pedal.cut_time
        midi_note.soft_pedal = note.pedal.soft

        midi_notes.append(midi_note)

    return midi_notes


def check_pairs(pairs):
    non_matched = 0
    for pair in pairs:
        if pair == []:
            non_matched += 1

    print('Number of non matched pairs: ' + str(non_matched))
    return non_matched


def check_overlapped_notes(xml_notes):
    previous_onset = -1
    notes_on_onset = []
    pitches = []
    for note in xml_notes:
        if note.note_duration.is_grace_note:
            continue # does not count grace note, because it can have same pitch on same xml_position
        if note.note_duration.xml_position > previous_onset:
            previous_onset = note.note_duration.xml_position
            pitches = []
            pitches.append(note.pitch[1])
            notes_on_onset = []
            notes_on_onset.append(note)
        else: #note has same xml_position
            if note.pitch[1] in pitches: # same pitch with same
                index_of_same_pitch_note = pitches.index(note.pitch[1])
                previous_note = notes_on_onset[index_of_same_pitch_note]
                if previous_note.note_duration.duration > note.note_duration.duration:
                    note.is_overlapped = True
                else:
                    previous_note.is_overlapped = True
            else:
                pitches.append(note.pitch[1])
                notes_on_onset.append(note)

    return xml_notes


def read_xml_to_array(path_name, means, stds, start_tempo, composer_name, vel_standard):
    xml_name = path_name + 'musicxml_cleaned.musicxml'
    midi_name = path_name + 'midi_cleaned.mid'

    if not os.path.isfile(xml_name):
        xml_name = path_name + 'xml.xml'
        midi_name = path_name + 'midi.mid'


    xml_object = MusicXMLDocument(xml_name)
    beats = cal_beat_positions_of_piece(xml_object)
    xml_notes = extract_notes(xml_object, melody_only=False, grace_note=True)
    directions, time_signatures = extract_directions(xml_object)
    xml_notes = apply_directions_to_notes(xml_notes, directions, time_signatures)

    measure_positions = extract_measure_position(xml_object)
    features = extract_score_features(xml_notes, measure_positions, beats, qpm_primo=start_tempo, vel_standard=vel_standard)
    features = make_index_continuous(features, score=True)
    composer_vec = composer_name_to_vec(composer_name)

    for i in range(len(stds[0])):
        if stds[0][i] < 1e-4 or isinstance(stds[0][i], complex):
            stds[0][i] = 1

    test_x = []
    note_locations = []
    for feat in features:
        # if not feat['pitch_interval'] == None:
        # temp_x = [ (feat.pitch-means[0][0])/stds[0][0],  (feat.pitch_interval-means[0][1])/stds[0][1] ,
        #                 (feat.duration - means[0][2]) / stds[0][2],(feat.duration_ratio-means[0][3])/stds[0][3],
        #                 (feat.beat_position-means[0][4])/stds[0][4], (feat.measure_length-means[0][5])/stds[0][5],
        #            (feat.voice - means[0][6]) / stds[0][6], (feat.qpm_primo - means[0][7]) / stds[0][7],
        #                 feat.xml_position, feat.grace_order, feat.time_sig_num, feat.time_sig_den]\
        #          + feat.tempo + feat.dynamic + feat.notation + feat.tempo_primo
        # temp_x = [(feat.pitch_interval - means[0][0]) / stds[0][0],
        #           (feat.duration - means[0][1]) / stds[0][1],(feat.duration_ratio-means[0][2])/stds[0][2],
        #             (feat.beat_position-means[0][3])/stds[0][3], (feat.measure_length-means[0][4])/stds[0][4],
        #          (feat.qpm_primo - means[0][5]) / stds[0][5],(feat.following_rest - means[0][6]) / stds[0][6],
        #           (feat.mean_piano_vel - means[0][7]) / stds[0][7],  (feat.mean_forte_vel - means[0][8]) / stds[0][8],
        #           (feat.mean_piano_mark - means[0][9]) / stds[0][9],  (feat.mean_forte_mark - means[0][10]) / stds[0][10],
        #           (feat.distance_from_abs_dynamic - means[0][11]) / stds[0][11],
        #           feat.xml_position, feat.grace_order,
        #           feat.time_sig_num, feat.time_sig_den, feat.no_following_note] \
        #          + feat.pitch + feat.tempo + feat.dynamic + feat.notation + feat.tempo_primo
        # temp_x = [(feat.pitch_interval - means[0][0]) / stds[0][0],
        #           (feat.duration - means[0][1]) / stds[0][1],(feat.duration_ratio-means[0][2])/stds[0][2],
        #             (feat.beat_position-means[0][3])/stds[0][3], (feat.measure_length-means[0][4])/stds[0][4],
        #           (feat.following_rest - means[0][5]) / stds[0][5],
        #           (feat.distance_from_abs_dynamic - means[0][6]) / stds[0][6],
        #           feat.xml_position, feat.grace_order,
        #           feat.time_sig_num, feat.time_sig_den, feat.no_following_note] \
        #             + feat.pitch + feat.tempo + feat.dynamic + feat.notation
        temp_x = [(feat.pitch_interval - means[0][0]) / stds[0][0],
                    (feat.duration - means[0][1]) / stds[0][1],(feat.duration_ratio-means[0][2])/stds[0][2],
                      (feat.beat_position-means[0][3])/stds[0][3], (feat.measure_length-means[0][4])/stds[0][4],
                   (feat.qpm_primo - means[0][5]) / stds[0][5],(feat.following_rest - means[0][6]) / stds[0][6],
                    (feat.distance_from_abs_dynamic - means[0][7]) / stds[0][7],
                  (feat.distance_from_recent_tempo - means[0][8]) / stds[0][8] ,
                    feat.xml_position, feat.grace_order,
                    feat.time_sig_num, feat.time_sig_den, feat.no_following_note] \
                   + feat.pitch + feat.tempo + feat.dynamic + composer_vec + feat.notation + feat.tempo_primo
        # temp_x.append(feat.is_beat)
        test_x.append(temp_x)
        note_locations.append(feat.note_location)
        # else:
        #     test_x.append( [(feat['pitch']-means[0][0])/stds[0][0], 0,  (feat['duration'] - means[0][2]) / stds[0][2], 0,
        #                     (feat['beat_position']-means[0][4])/stds[0][4]]
        #                    + feat['tempo'] + feat['dynamic'] + feat['notation'] )

    return test_x, xml_notes, xml_object, note_locations


def cal_beat_positions_of_piece(xml_doc):
    piano = xml_doc.parts[0]
    num_measure = len(piano.measures)
    time_signatures = xml_doc.get_time_signatures()
    time_sig_position = [time.xml_position for time in time_signatures]
    beat_piece = []
    for i in range(num_measure):
        measure = piano.measures[i]
        measure_start = measure.start_xml_position
        corresp_time_sig_idx = binaryIndex(time_sig_position, measure_start)
        corresp_time_sig = time_signatures[corresp_time_sig_idx]
        # corresp_time_sig = measure.time_signature
        measure_length = corresp_time_sig.state.divisions * corresp_time_sig.numerator / corresp_time_sig.denominator * 4
        # if i +1 < num_measure:
        #     measure_length = piano.measures[i+1].start_xml_position - measure_start
        # else:
        #     measure_length = measure_start - piano.measures[i-1].start_xml_position

        num_beat_in_measure = corresp_time_sig.numerator
        if num_beat_in_measure == 6:
            num_beat_in_measure = 2
        elif num_beat_in_measure == 9:
            num_beat_in_measure = 3
        elif num_beat_in_measure == 12:
            num_beat_in_measure = 4

        inter_beat_interval = measure_length / num_beat_in_measure
        if measure.implicit:
            current_measure_length = piano.measures[i + 1].start_xml_position - measure_start
            length_ratio = current_measure_length / measure_length
            minimum_beat = 1 / corresp_time_sig.numerator
            num_beat_in_measure = int(math.ceil(length_ratio / minimum_beat))
            for j in range(-num_beat_in_measure, 0):
                beat = piano.measures[i + 1].start_xml_position + j * inter_beat_interval
                beat_piece.append(beat)
        else:
            for j in range(num_beat_in_measure):
                beat = measure_start + j * inter_beat_interval
                beat_piece.append(beat)
            #
        # for note in measure.notes:
        #     note.on_beat = check_note_on_beat(note, measure_start, measure_length)
    return beat_piece

class Tempo:
    def __init__(self, xml_position, qpm, time_position, end_xml, end_time):
        self.qpm = qpm
        self.xml_position = xml_position
        self.time_position = time_position
        self.end_time = end_time
        self.end_xml = end_xml

    def __str__(self):
        string = '{From ' + str(self.xml_position)
        string += ' to ' + str(self.end_xml)
        return string


def cal_tempo(xml_doc, xml_notes, pairs, features):
    beats = cal_beat_positions_of_piece(xml_doc)
    # xml_notes = extract_notes(xml_doc, melody_only=False, grace_note=True)
    # xml_positions = [note.note_duration.xml_position for note in xml_notes]
    tempos = []
    # num_notes = len(xml_notes)

    num_beats = len(beats)
    pairs, position_pairs = make_available_xml_midi_positions(pairs)
    # num_pos_pairs = len(position_pairs)
    previous_end = 0
    for i in range(num_beats-1):
        beat = beats[i]
        current_pos_pair = get_item_by_xml_position(position_pairs, beat)
        if current_pos_pair.xml_position < previous_end:
            # current_pos_pair = get_next_item(position_pairs, current_pos_pair)
            continue

        next_beat = beats[i+1]
        next_pos_pair = get_item_by_xml_position(position_pairs, next_beat)

        # if not next_pos_pair.xml_position == next_beat:
        #     next_pos_pair = get_next_item(position_pairs, next_pos_pair)

        if next_pos_pair.xml_position == previous_end:
            continue

        if current_pos_pair == next_pos_pair:
            continue

        cur_xml = current_pos_pair.xml_position
        cur_time = current_pos_pair.time_position
        # cur_time = get_average_of_onset_time(pairs, current_pos_pair.index)
        cur_divisions = current_pos_pair.divisions
        next_xml = next_pos_pair.xml_position
        next_time = next_pos_pair.time_position
        # next_time = get_average_of_onset_time(pairs, next_pos_pair.index)


        qpm = (next_xml - cur_xml) / (next_time - cur_time) / cur_divisions * 60

        if qpm > 1000:
            print('need check: qpm is ' + str(qpm) +', current xml_position is ' + str(cur_xml))
        tempo = Tempo(cur_xml, qpm, cur_time, next_xml, next_time)
        tempos.append(tempo)        #
        previous_end = next_pos_pair.xml_position


        current_index = current_pos_pair.index
        feat = features[current_index]
        feat.is_beat = True
        num = 1
        while current_index - num >= 0 and xml_notes[current_index - num].note_duration.xml_position == cur_xml:
            feat = features[current_index - num]
            feat.is_beat = True
            num += 1


        feat.is_beat = True
        # note_index = binaryIndex(xml_positions, beat)
        # while pairs[note_index] == [] and note_index >0:
        #     note_index += -1
        # while pairs[note_index] ==[]:
        #     note_index += 1
        # note = xml_notes[note_index]
        #
        # note_midi_start = pairs[note_index]['midi'].start

        # next_beat = beats[i + 1]
        # next_note_index = binaryIndex(xml_positions, next_beat)
        # next_note = xml_notes[next_note_index]
        #
        # # handle the case when there is no note on the next beat
        # if not next_note.note_duration.xml_position == next_beat and next_note_index+1 < num_notes:
        #     next_note = xml_notes[next_note_index+1] # this should be later than the beat
        #     if next_note.note_duration.xml_position < beat:
        #         # there is no notes later than the beat
        #         break
        #     # find the lowest note
        #     while next_note_index+1 < num_notes and \
        #             xml_notes[next_note_index+1].note_duration.xml_position == next_note.note_duration.xml_position:
        #         next_note_index +=1
        #         next_note = xml_notes[next_note_index]
        #
        # while pairs[next_note_index] == [] and next_note_index > note_index+1:
        #     next_note_index += -1
        #
        # next_note = xml_notes[next_note_index]
        # if not pairs[next_note_index] == []:
        #     next_midi_start = pairs[next_note_index]['midi'].start
        #     qpm = (next_note.note_duration.xml_position - note.note_duration.xml_position) / (next_midi_start - note_midi_start) / note.state_fixed.divisions * 60
        #
        #     if math.isnan(qpm) or math.isinf(qpm):
        #         qpm = tempos[-1].qpm
        # else:
        #     if len(tempos) > 1:
        #         qpm = tempos[-1].qpm
        #     else:
        #         print('Error: Could not calculate qpm in the first beat')
        #         continue # skip this beat
        # tempo = Tempo(note.note_duration.xml_position, qpm, note_midi_start, next_midi_start)
        # tempos.append(tempo)

    #
    # for tempo in tempos:
    #     print(tempo.xml_position,  tempo.end_xml, tempo.qpm, tempo.time_position, tempo.end_time)

    return tempos


def get_average_of_onset_time(pairs, index):
    current_note = pairs[index]['xml']
    current_position = current_note.note_duration.xml_position
    standard_time = pairs[index]['midi'].start
    cur_time =  pairs[index]['midi'].start
    added_num = 1

    for i in range(1,index):
        if not pairs[index-i] == []:
            prev_note = pairs[index-i]['xml']
            prev_midi = pairs[index-i]['midi']
            if prev_note.note_duration.xml_position == current_position \
                    and not prev_note.note_duration.is_grace_note and not prev_note.note_duration.after_grace_note:
                if abs(standard_time - prev_midi.start) < 0.5:
                    cur_time += prev_midi.start
                    added_num += 1
            else:
                break

    return cur_time / added_num


def get_next_item(alist, item):
    if item in alist:
        index = alist.index(item)
        if index+1 < len(alist):
            return alist[index+1]
        else:
            return item
    else:
        return None


def make_available_xml_midi_positions(pairs):
    class PositionPair:
        def __init__(self, xml_pos, time, pitch, index, divisions):
            self.xml_position = xml_pos
            self.time_position = time
            self.pitch = pitch
            self.index = index
            self.divisions = divisions
            self.is_arpeggiate = False

    available_pairs = []
    num_pairs = len(pairs)
    for i in range(num_pairs):
        pair = pairs[i]
        if not pair == []:
            xml_note = pair['xml']
            midi_note = pair['midi']
            xml_pos = xml_note.note_duration.xml_position
            time = midi_note.start
            divisions = xml_note.state_fixed.divisions
            if not xml_note.note_duration.is_grace_note:
                pos_pair = PositionPair(xml_pos, time, xml_note.pitch[1], i, divisions)
                if xml_note.note_notations.is_arpeggiate:
                    pos_pair.is_arpeggiate = True
                available_pairs.append(pos_pair)

    # available_pairs = save_lowest_note_on_same_position(available_pairs)
    available_pairs, mismatched_indexes = make_average_onset_cleaned_pair(available_pairs)
    print('Number of mismatched notes: ', len(mismatched_indexes))
    for index in mismatched_indexes:
        pairs[index] = []

    return pairs, available_pairs


def check_note_on_beat(note, measure_start, measure_length):
    note_position = note.note_duration.xml_position
    position_ratio = note_position / measure_length
    num_beat_in_measure = note.tempo.time_numerator

    on_beat = int(position_ratio * num_beat_in_measure) == (position_ratio * num_beat_in_measure)
    return on_beat


class EmbeddingTable():
    def __init__(self):
        self.keywords=[]
        self.embed_key = []

    def append(self, EmbeddingKey):
        self.keywords.append(EmbeddingKey.key)
        self.embed_key.append(EmbeddingKey)

class EmbeddingKey():
    def __init__(self, key_name, vec_idx, value):
        self.key = key_name
        self.vector_index = vec_idx
        self.value = value



def dynamic_embedding(dynamic_word, embed_table, len_vec=4):
    dynamic_vector = [0] * len_vec
    # dynamic_vector[0] = 0.5
    keywords = embed_table.keywords


    if dynamic_word == None:
        return dynamic_vector
    if dynamic_word in embed_table.keywords:
        index = find_index_list_of_list(dynamic_word, keywords)
        vec_idx = embed_table.embed_key[index].vector_index
        dynamic_vector[vec_idx] = embed_table.embed_key[index].value

    # for i in range(len(keywords)):
    #     keys = keywords[i]
    #     if type(keys) is list:
    #         for key in keys:
    #             if len(key)>2 and (key.encode('utf-8') in
    word_split = dynamic_word.replace(',', ' ').replace('.', ' ').split(' ')
    for w in word_split:
        index = find_index_list_of_list(w.lower(), keywords)
        if index:
            vec_idx = embed_table.embed_key[index].vector_index
            dynamic_vector[vec_idx] = embed_table.embed_key[index].value

    for key in keywords:
        if isinstance(key, str) and len(key) > 2 and key in dynamic_word:
            # if type(key) is st and len(key) > 2 and key in attribute:
            index = keywords.index(key)
            vec_idx = embed_table.embed_key[index].vector_index
            dynamic_vector[vec_idx] = embed_table.embed_key[index].value

    return dynamic_vector

def define_dyanmic_embedding_table():
    embed_table = EmbeddingTable()

    embed_table.append(EmbeddingKey('ppp', 0, -0.9))
    embed_table.append(EmbeddingKey('pp', 0, -0.7))
    embed_table.append(EmbeddingKey('piano', 0, -0.4))
    embed_table.append(EmbeddingKey('p', 0, -0.4))
    embed_table.append(EmbeddingKey('mp', 0, -0.2))
    embed_table.append(EmbeddingKey('mf', 0, 0.2))
    embed_table.append(EmbeddingKey('f', 0, 0.4))
    embed_table.append(EmbeddingKey('forte', 0, 0.4))
    embed_table.append(EmbeddingKey('ff', 0, 0.7))
    embed_table.append(EmbeddingKey('fff', 0, 0.9))

    embed_table.append(EmbeddingKey('più p', 0, -0.5))
    embed_table.append(EmbeddingKey('più piano', 0, -0.5))
    embed_table.append(EmbeddingKey('più pp', 0, -0.7))
    embed_table.append(EmbeddingKey('più f', 0, 0.5))
    embed_table.append(EmbeddingKey('più forte', 0, 0.5))
    embed_table.append(EmbeddingKey('più forte possibile', 0, 1))

    embed_table.append(EmbeddingKey('cresc', 1, 0.7))
    embed_table.append(EmbeddingKey('crescendo', 1, 0.7))
    embed_table.append(EmbeddingKey('allargando', 1, 0.4))
    embed_table.append(EmbeddingKey('dim', 1, -0.7))
    embed_table.append(EmbeddingKey('diminuendo', 1, -0.7))
    embed_table.append(EmbeddingKey('decresc', 1, -0.7))
    embed_table.append(EmbeddingKey('decrescendo', 1, -0.7))

    embed_table.append(EmbeddingKey('smorz', 1, -0.4))

    embed_table.append(EmbeddingKey('poco a poco meno f', 1, -0.2))
    embed_table.append(EmbeddingKey('poco cresc', 1, 0.5))
    embed_table.append(EmbeddingKey('molto cresc', 1, 1))
    embed_table.append(EmbeddingKey('cresc molto', 1, 1))


    embed_table.append(EmbeddingKey('fz', 2, 0.3))
    embed_table.append(EmbeddingKey('sf', 2, 0.5))
    embed_table.append(EmbeddingKey('sfz', 2, 0.7))
    embed_table.append(EmbeddingKey('ffz', 2, 0.8))
    embed_table.append(EmbeddingKey('sffz', 2, 0.9))

    embed_table.append(EmbeddingKey('rf', 3, 0.3))
    embed_table.append(EmbeddingKey('rinf', 3, 0.3))
    embed_table.append(EmbeddingKey('rinforzando', 3, 0.3))
    embed_table.append(EmbeddingKey('rinforzando molto', 3, 0.7))
    embed_table.append(EmbeddingKey('rinforzando assai', 3, 0.6))
    embed_table.append(EmbeddingKey('rinforz assai', 3, 0.6))
    embed_table.append(EmbeddingKey('molto rin', 3, 0.5))

    embed_table.append(EmbeddingKey('con brio', 3, 0.3))
    embed_table.append(EmbeddingKey('con forza', 3, 0.5))
    embed_table.append(EmbeddingKey('con fuoco', 3, 0.7))
    embed_table.append(EmbeddingKey('con più fuoco possibile', 3, 1))
    embed_table.append(EmbeddingKey('sotto voce', 3, -0.5))
    embed_table.append(EmbeddingKey('mezza voce', 3, -0.3))
    embed_table.append(EmbeddingKey('appassionato', 3, 0.5))
    embed_table.append(EmbeddingKey('più rinforz', 3, 0.5))

    return embed_table


def define_tempo_embedding_table():
    # [abs tempo, abs_tempo_added, tempo increase or decrease, sudden change]
    embed_table = EmbeddingTable()
    embed_table.append(EmbeddingKey('freely, with expression', 0, 0.2))
    embed_table.append(EmbeddingKey('lento', 0, -0.9))
    embed_table.append(EmbeddingKey('grave', 0, -0.9))
    embed_table.append(EmbeddingKey('largo', 0, -0.7))
    embed_table.append(EmbeddingKey('languido', 0, -0.6))
    embed_table.append(EmbeddingKey('molto languido', 0, -0.7))
    embed_table.append(EmbeddingKey('adagio', 0, -0.6))
    embed_table.append(EmbeddingKey('larghetto', 0, -0.6))
    embed_table.append(EmbeddingKey('adagietto', 0, -0.55))
    embed_table.append(EmbeddingKey('andante', 0, -0.5))
    embed_table.append(EmbeddingKey('andantino', 0, -0.3))
    embed_table.append(EmbeddingKey('mesto', 0, -0.3))
    embed_table.append(EmbeddingKey('andantino molto', 0, -0.4))
    embed_table.append(EmbeddingKey('maestoso', 0, -0.3))
    embed_table.append(EmbeddingKey('accarezzevole', 0, -0.4))
    embed_table.append(EmbeddingKey('moderato', 0, 0))
    embed_table.append(EmbeddingKey('tempo giusto', 0, 0.1))
    embed_table.append(EmbeddingKey('allegretto', 0, 0.3))
    embed_table.append(EmbeddingKey('allegro', 0, 0.5))
    embed_table.append(EmbeddingKey('allegro assai', 0, 0.6))
    embed_table.append(EmbeddingKey('vivace', 0, 0.6))
    embed_table.append(EmbeddingKey('vivacissimo', 0, 0.8))
    embed_table.append(EmbeddingKey('presto', 0, 0.8))
    embed_table.append(EmbeddingKey('prestissimo', 0, 0.9))

    embed_table.append(EmbeddingKey('doppio movimento', 0, 0.6))
    embed_table.append(EmbeddingKey('molto allegro', 0, 0.85))
    embed_table.append(EmbeddingKey('allegro molto', 0, 0.85))
    embed_table.append(EmbeddingKey('più presto possibile', 0, 1))


    embed_table.append(EmbeddingKey('a tempo', 1, 0))
    embed_table.append(EmbeddingKey('meno mosso', 1, -0.8))
    embed_table.append(EmbeddingKey('meno vivo', 1, -0.7))
    embed_table.append(EmbeddingKey('ritenuto', 1, -0.5))
    embed_table.append(EmbeddingKey('animato', 1, 0.5))
    embed_table.append(EmbeddingKey('più animato', 1, 0.6))
    embed_table.append(EmbeddingKey('agitato', 1, 0.4))
    embed_table.append(EmbeddingKey('più mosso', 1, 0.8))
    embed_table.append(EmbeddingKey('stretto', 1, 0.5))
    embed_table.append(EmbeddingKey('appassionato', 1, 0.2))
    embed_table.append(EmbeddingKey('più moderato', 1, -0.1))
    embed_table.append(EmbeddingKey('più allegro', 1, 0.8))

    embed_table.append(EmbeddingKey('poco riten', 1, -0.3))
    embed_table.append(EmbeddingKey('poco ritenuto', 1, -0.3))
    embed_table.append(EmbeddingKey('poco meno mosso', 1, -0.5))
    embed_table.append(EmbeddingKey('poco più mosso', 1, 0.5))
    embed_table.append(EmbeddingKey('più animato', 1, 0.3))
    embed_table.append(EmbeddingKey('molto agitato', 1, 0.8))

    embed_table.append(EmbeddingKey('tranquillo', 1, -0.2))


    # French
    embed_table.append(EmbeddingKey('très grave', 0, -1.0))
    embed_table.append(EmbeddingKey('très lent', 0, -0.9))
    # embed_table.append(EmbeddingKey('marche funèbre', 0, -0.8))
    embed_table.append(EmbeddingKey('lent', 0, -0.7))
    embed_table.append(EmbeddingKey('large', 0, -0.7))
    embed_table.append(EmbeddingKey("Assez doux, mais d'une sonoritè large", 0, -0.6))
    embed_table.append(EmbeddingKey('assez vif', 0, 0.6))
    embed_table.append(EmbeddingKey('assez animé', 0, 0.7))
    embed_table.append(EmbeddingKey("d'un rythme souple", 0, 0))

    embed_table.append(EmbeddingKey('un peau retenu', 1, -0.3))
    embed_table.append(EmbeddingKey('retenez', 1, -0.5))
    embed_table.append(EmbeddingKey('en élargissant beaucoup', 2, -0.4))
    embed_table.append(EmbeddingKey('plus lent', 1, -0.5))
    embed_table.append(EmbeddingKey('un peu plus lent', 1, -0.3))
    embed_table.append(EmbeddingKey('encore plus lent', 1, -0.5))
    embed_table.append(EmbeddingKey('cédez', 1, -0.4))
    embed_table.append(EmbeddingKey('un peu retenu', 1, -0.2))
    embed_table.append(EmbeddingKey('Un pe moins vif', 1, -0.2))
    embed_table.append(EmbeddingKey('cédez légèrement', 1, -0.2))
    embed_table.append(EmbeddingKey('mesuré', 1, 0))
    embed_table.append(EmbeddingKey('au mouvt', 1, 0))
    embed_table.append(EmbeddingKey('reprenez le mouvement', 1, 0))
    embed_table.append(EmbeddingKey('Un peu plus vif', 1, 0.2))
    embed_table.append(EmbeddingKey('en accélérant', 1, 0.3))
    embed_table.append(EmbeddingKey('en animant', 1, 0.6))
    embed_table.append(EmbeddingKey('rapide', 1, 0.6))
    embed_table.append(EmbeddingKey('rapido molto', 1, 0.8))

    embed_table.append(EmbeddingKey('accélerez', 2, 0.5))
    embed_table.append(EmbeddingKey('sans ralentir', 2, 0.01))

    # German
    embed_table.append(EmbeddingKey('sehr langsam', 0, -0.55))
    embed_table.append(EmbeddingKey('sehr innig', 0, -0.4))
    embed_table.append(EmbeddingKey('leicht uns zart', 0, 0.2))
    embed_table.append(EmbeddingKey('sehr lebhaft', 0, 0.6))
    embed_table.append(EmbeddingKey('sehr aufgeregt', 0, 0.6))
    embed_table.append(EmbeddingKey('sehr rasch', 0, 0.8))
    embed_table.append(EmbeddingKey('Sehr innig und nicht zu rasch', 0, -0.8))

    embed_table.append(EmbeddingKey('langsamer', 1, -0.5))
    embed_table.append(EmbeddingKey('etwas langsamer', 1, -0.3))
    embed_table.append(EmbeddingKey('bewegter', 1, 0.4))
    embed_table.append(EmbeddingKey('schneller', 1, 0.5))

    embed_table.append(EmbeddingKey('allargando', 2, -0.2))
    embed_table.append(EmbeddingKey('ritardando', 2, -0.5))
    embed_table.append(EmbeddingKey('rit', 2, -0.5))
    embed_table.append(EmbeddingKey('rallentando', 2, -0.5))
    embed_table.append(EmbeddingKey('rall', 2, -0.5))
    embed_table.append(EmbeddingKey('slentando', 2, -0.3))
    embed_table.append(EmbeddingKey('acc', 2, 0.5))
    embed_table.append(EmbeddingKey('accel', 2, 0.5))
    embed_table.append(EmbeddingKey('accelerando', 2, 0.5))
    embed_table.append(EmbeddingKey('smorz', 2, -0.5))
    embed_table.append(EmbeddingKey('string', 2, 0.4))
    embed_table.append(EmbeddingKey('stringendo', 2, 0.4))
    embed_table.append(EmbeddingKey('molto stringendo', 2, 0.7))
    embed_table.append(EmbeddingKey('stringendo molto', 2, 0.7))


    embed_table.append(EmbeddingKey('poco rall', 2, -0.3))
    embed_table.append(EmbeddingKey('poco rit', 2, -0.3))

    # non troppo presto
    # energico
    # tempo di marcia
    # marcia funbre
    # ben marcato
    # rinforzando assai
    # rinforzando molto
    # piu presto possibile
    # mesto
    # tempo giusto
    # sempre più largo
    # plus lent
    # 1er Mouvement

    return embed_table


def omit_trill_notes(xml_notes):
    num_notes = len(xml_notes)
    omit_index = []
    trill_sign = []
    wavy_lines = []
    for i in range(num_notes):
        note = xml_notes[i]
        if not note.is_print_object:
            omit_index.append(i)
            if note.note_notations.is_trill:
                trill = {'xml_pos': note.note_duration.xml_position, 'pitch': note.pitch[1]}
                trill_sign.append(trill)
        if note.note_notations.wavy_line:
            wavy_line = note.note_notations.wavy_line
            wavy_line.xml_position = note.note_duration.xml_position
            wavy_line.pitch = note.pitch
            wavy_lines.append(wavy_line)
    wavy_lines = combine_wavy_lines(wavy_lines)

    for index in reversed(omit_index):
        note = xml_notes[index]
        xml_notes.remove(note)

    if len(trill_sign) > 0:
        for trill in trill_sign:
            for note in xml_notes:
                if note.note_duration.xml_position == trill['xml_pos'] and abs(note.pitch[1] - trill['pitch']) <4 \
                        and not note.note_duration.is_grace_note:
                    note.note_notations.is_trill = True
                    break

    xml_notes = apply_wavy_lines(xml_notes, wavy_lines)

    return xml_notes


def find_corresp_trill_notes_from_midi(xml_doc, xml_notes, pairs, perf_midi, accidentals, index):
    #start of trill, end of trill
    key_signatures = xml_doc.get_key_signatures()
    note = xml_notes[index]
    num_note = len(xml_notes)

    key = get_item_by_xml_position(key_signatures, note)
    key = key.key
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

    measure_accidentals = get_measure_accidentals(xml_notes, index)
    trill_pitch = note.pitch[1]

    up_pitch, _ = cal_up_trill_pitch(note.pitch, key, final_key, measure_accidentals)

    prev_search = 1
    prev_idx = index
    next_idx = index
    while index - prev_search >= 0:
        prev_idx = index - prev_search
        prev_note = xml_notes[prev_idx]
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
        next_note = xml_notes[next_idx]
        if next_note.note_duration.xml_position > trill_end_position:
            break
        else:
            next_search += 1

    skipped_pitches_start = []
    skipped_pitches_end = []
    while pairs[prev_idx] == [] and prev_idx >0:
        skipped_note = xml_notes[prev_idx]
        skipped_pitches_start.append(skipped_note.pitch[1])
        prev_idx -= 1
    while pairs[next_idx] == [] and next_idx < num_note -1:
        skipped_note = xml_notes[next_idx]
        skipped_pitches_end.append(skipped_note.pitch[1])
        next_idx += 1

    if pairs[next_idx] == [] or pairs[prev_idx] == []:
        print("Error: Cannot find trill start or end note")
        return [0] * 5, 0
    prev_midi_note = pairs[prev_idx]['midi']
    next_midi_note = pairs[next_idx]['midi']
    search_range_start = perf_midi.index(prev_midi_note)
    search_range_end = perf_midi.index(next_midi_note)
    trills = []
    prev_pitch = None
    if len(skipped_pitches_end) > 0:
        end_cue = skipped_pitches_end[0]
    else:
        end_cue = None
    for i in range(search_range_start+1, search_range_end):
        midi_note = perf_midi[i]
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
            print('check trill pitch - detected pitch: ', midi_note.pitch, ' trill note pitch: ', trill_pitch, 'expected up trill pitch: ', up_pitch)

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

    trills_vec[0] = num_trills
    trills_vec[1] = trills[-1].velocity / trills[0].velocity
    trills_vec[2] = ioi_seconds[0]
    trills_vec[3] = ioi_seconds[-1]
    trills_vec[4] = int(up_trill)


    if pairs[index] == []:
        for pair in pairs:
            if not pair ==[] and pair['midi'] == trills[0]:
                pair = []
        pairs[index] = {'xml': note, 'midi': trills[0]}

    return trills_vec, trill_length

def cal_up_trill_pitch(pitch_tuple, key, final_key, measure_accidentals):
    pitches = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    corresp_midi_pitch = [0, 2, 4, 5, 7, 9, 11]
    pitch_name = pitch_tuple[0][0:1].lower()
    octave = int(pitch_tuple[0][-1])
    next_pitch_name = pitches[(pitches.index(pitch_name)+1)%7]
    if next_pitch_name == 'c':
        octave += 1

    accidentals = ['f', 'c', 'g', 'd', 'a', 'e', 'b']
    if key > 0 and next_pitch_name in accidentals[:key]:
        acc = +1
    elif key < 0 and next_pitch_name in accidentals[key:]:
        acc = -1
    else:
        acc= 0

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

def extract_accidental(xml_doc):
    directions = []
    accs = ['#', '♭','♮']
    # accs = ' # ♭ ♮ '
    for part in xml_doc.parts:
        for measure in part.measures:
            for direction in measure.directions:
                if direction.type['type'] == 'words' and direction.type['content'] in accs:
                    directions.append(direction)
    return directions


def combine_wavy_lines(wavy_lines):
    num_wavy = len(wavy_lines)
    for i in reversed(range(num_wavy)):
        wavy = wavy_lines[i]
        if wavy.type == 'stop':
            deleted = False
            for j in range(1, i+1):
                prev_wavy = wavy_lines[i - j]
                if prev_wavy.type == 'start' and prev_wavy.number == wavy.number:
                    prev_wavy.end_xml_position = wavy.xml_position
                    wavy_lines.remove(wavy)
                    deleted = True
                    break
            if not deleted:
                wavy_lines.remove(wavy)
    num_wavy = len(wavy_lines)
    for i in reversed(range(num_wavy)):
        wavy = wavy_lines[i]
        if wavy.type == 'start' and wavy.end_xml_position == 0:
            wavy_lines.remove(wavy)
    return wavy_lines


def apply_wavy_lines(xml_notes, wavy_lines):
    xml_positions = [x.note_duration.xml_position for x in xml_notes]
    num_notes = len(xml_notes)
    omit_indices = []
    for wavy in wavy_lines:
        index = binaryIndex(xml_positions, wavy.xml_position)
        while abs(xml_notes[index].pitch[1] - wavy.pitch[1]) > 3 and index > 0 \
                and xml_notes[index-1].note_duration.xml_position == xml_notes[index].note_duration.xml_position:
            index -= 1
        note = xml_notes[index]
        wavy_duration = wavy.end_xml_position - wavy.xml_position
        note.note_duration.duration = wavy_duration
        trill_pitch = note.pitch[1]
        next_idx = index+1
        while next_idx < num_notes and xml_notes[next_idx].note_duration.xml_position < wavy.end_xml_position:
            if xml_notes[next_idx].pitch[1] == trill_pitch:
                omit_indices.append(next_idx)
            next_idx += 1

    omit_indices.sort()
    if len(omit_indices)> 0:
        for idx in reversed(omit_indices):
            del xml_notes[idx]

    return xml_notes


def get_measure_accidentals(xml_notes, index):
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
                        continue
                    accident = accs.index(acc) - 2
                    temp_pair = {'pitch': pitch, 'accident': accident}
                    measure_accidentals.append(temp_pair)
                    break
    return measure_accidentals


def make_index_continuous(features, score=False):
    prev_beat = 0
    prev_measure = 0

    beat_compensate = 0
    measure_compensate = 0

    for feat in features:
        if feat.qpm is not None or score:
            if feat.note_location.beat - prev_beat > 1:
                beat_compensate -= (feat.note_location.beat - prev_beat) -1
            if feat.note_location.measure - prev_measure > 1:
                measure_compensate -= (feat.note_location.measure - prev_measure) -1

            prev_beat = feat.note_location.beat
            prev_measure = feat.note_location.measure

            feat.note_location.beat += beat_compensate
            feat.note_location.measure += measure_compensate
        else:
            continue
    return features


def check_index_continuity(features):
    prev_beat = 0
    prev_measure = 0

    for feat in features:
        if feat.beat_index - prev_beat > 1:
            print(feat.beat_index, prev_beat)
        if feat.measure_index - prev_measure > 1:
            print(feat.measure_index, prev_measure)

        prev_beat = feat.beat_index
        prev_measure = feat.measure_index


def cal_mean_velocity_by_dynamics_marking(features):
    piano_velocities = []
    forte_velocities = []
    forte_markings = []
    piano_markings = []
    entire_velocities = []
    entire_markings = []

    for feat in features:
        absolute_dynamic = feat.dynamic[0]
        rel_dynamic = feat.dynamic[1]
        velocity = feat.velocity
        if absolute_dynamic > 0 and rel_dynamic == 0:
            forte_velocities.append(velocity)
            forte_markings.append(absolute_dynamic)
        elif absolute_dynamic < 0 and rel_dynamic == 0 :
            piano_velocities.append(velocity)
            piano_markings.append(absolute_dynamic)

        entire_velocities.append(velocity)
        entire_markings.append(absolute_dynamic)
    mean_vel = sum(entire_velocities) / len(entire_velocities)
    mean_dynamic = sum(entire_markings) / len(entire_markings)

    if len(piano_velocities) >0:
        mean_piano_vel = sum(piano_velocities) / len(piano_velocities)
        mean_piano_marking = sum(piano_markings) / len(piano_markings)
    else:
        mean_piano_vel = mean_vel
        mean_piano_marking = -0.2
    if len(forte_velocities) > 0:
        mean_forte_vel = sum(forte_velocities) / len(forte_velocities)
        mean_forte_marking = sum(forte_markings) / len(forte_markings)
    else:
        mean_forte_vel = mean_vel
        mean_forte_marking = 0.2

    return mean_piano_vel, mean_piano_marking, mean_forte_vel, mean_forte_marking


def extract_and_apply_slurs(xml_notes):
    resolved_slurs = []
    unresolved_slurs = []
    slur_index = 0
    for note in xml_notes:
        slurs = note.note_notations.slurs
        if slurs:
            for slur in slurs:
                slur.xml_position = note.note_duration.xml_position
                slur.voice = note.voice
                type= slur.type
                if type == 'start':
                    slur.index = slur_index
                    unresolved_slurs.append(slur)
                    slur_index += 1
                elif type == 'stop':
                    for prev_slur in unresolved_slurs:
                        if prev_slur.number == slur.number and prev_slur.voice == slur.voice:
                            prev_slur.end_xml_position = slur.xml_position
                            resolved_slurs.append(prev_slur)
                            unresolved_slurs.remove(prev_slur)
                            note.note_notations.slurs.remove(slur)
                            note.note_notations.slurs.append(prev_slur)
                            break

    for note in xml_notes:
        slurs = note.note_notations.slurs
        if not slurs:
            for prev_slur in resolved_slurs:
                if prev_slur.voice == note.voice and prev_slur.xml_position < note.note_duration.xml_position <= prev_slur.end_xml_position:
                    note.note_notations.slurs.append(prev_slur)

    return xml_notes


def pedal_sigmoid(pedal_value, k=8):
    sigmoid_pedal = 127 / (1 + math.exp(-(pedal_value-64)/k))
    return int(sigmoid_pedal)


def composer_name_to_vec(composer_name):
    composer_name_list = ['Balakirev', 'Beethoven', 'Brahms', 'Chopin', 'Debussy', 'Haydn',
                          'Liszt', 'Mozart', 'Rachmaninoff', 'Ravel', 'Schumann', 'Scriabin']

    index = composer_name_list.index(composer_name)
    one_hot_vec = [0] * len(composer_name_list)
    one_hot_vec[index] = 1

    return one_hot_vec