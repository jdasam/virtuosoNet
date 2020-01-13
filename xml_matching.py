#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import os
import pretty_midi
import copy
import scipy.stats
import numpy as np
import random
import shutil
import subprocess
import ntpath
import pickle
import pandas


from .musicxml_parser import MusicXMLDocument
from .midi_utils import midi_utils
from . import performanceWorm as perf_worm, xml_direction_encoding as dir_enc, \
    score_as_graph as score_graph, xml_midi_matching as matching
from . import pedal_cleaning
from binary_index import binary_index

NUM_SCORE_NOTES = 0
NUM_PERF_NOTES = 0
NUM_NON_MATCHED_NOTES = 0
NUM_EXCLUDED_NOTES = 0

DYN_EMB_TAB = dir_enc.define_dyanmic_embedding_table()
TEM_EMB_TAB = dir_enc.define_tempo_embedding_table()

ALIGN_DIR = '/home/jdasam/AlignmentTool_v190813'

class MusicFeature:
    def __init__(self):
        self.midi_pitch = None
        self.pitch = None
        self.pitch_interval = None
        self.duration = None
        self.duration_ratio = None
        self.beat_position = None
        self.beat_importance = 0
        self.measure_length = None
        self.voice = None
        self.xml_position = None
        self.grace_order = None
        self.melody = None
        self.time_sig_num = None
        self.time_sig_den = None
        self.time_sig_vec = None
        self.is_beat = False
        self.following_rest = 0
        self.tempo_primo = None
        self.qpm_primo = None
        self.beat_index = 0
        self.measure_index = 0
        self.no_following_note=0
        self.note_location = self.NoteLocation(None, None, None, None, None)
        self.distance_from_abs_dynamic = None
        self.slur_index = None
        self.slur_beam_vec = None
        self.is_grace_note = False
        self.preceded_by_grace_note = False

        self.align_matched = 0
        self.dynamic = None
        self.tempo = None
        self.notation = None
        self.qpm = None
        self.beat_dynamic = None
        self.measure_tempo = None
        self.measure_dynamic = None
        self.section_tempo = None
        self.section_dynamic = None
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
        self.articulation_loss_weight = 1
        self.attack_deviation = 0


        self.mean_piano_vel = None
        self.mean_piano_mark = None
        self.mean_forte_vel = None
        self.mean_forte_mark = None

        self.midi_start = None
        self.passed_second = None
        self.duration_second = None


def cal_tempo_by_positions(beats, position_pairs):
    tempos = []
    num_beats = len(beats)
    previous_end = 0

    for i in range(num_beats - 1):
        beat = beats[i]
        current_pos_pair = get_item_by_xml_position(position_pairs, beat)
        if current_pos_pair.xml_position < previous_end:
            # current_pos_pair = get_next_item(position_pairs, current_pos_pair)
            continue

        next_beat = beats[i + 1]
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
            print('need check: qpm is ' + str(qpm) + ', current xml_position is ' + str(cur_xml))
        tempo = Tempo(cur_xml, qpm, cur_time, next_xml, next_time)
        tempos.append(tempo)  #
        previous_end = next_pos_pair.xml_position

    return tempos

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
    accidentals_in_words = extract_accidental(xml_doc)
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

'''
move cal_total_xml_length() to xml_utils.py
'''



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


def find_corresp_tempo(note, tempos):
    # note = pairs[index]['xml']
    tempo =  get_item_by_xml_position(tempos, note)
    # log_tempo = math.log(tempo, 10)
    # return log_tempo
    return tempo


def cal_articulation_with_tempo(pairs, i, tempo, trill_length):
    note = pairs[i]['xml']
    midi = pairs[i]['midi']
    xml_duration = note.note_duration.duration
    if xml_duration == 0:
        return 0
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
    # return pos_diff_sqrt
    return pos_diff_in_quarter_note


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
    elif numerator == 7:
        if abs((beat_position * 7) - 2) < 0.001:
            beat_importance = 2
        elif abs((beat_position * 5) - 2) < 0.001:
            beat_importance = 2
        else:
            beat_importance = 0
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

    index = binary_index(pos_list, item_pos)

    return alist[index]


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

#
# def load_pairs_by_path(score_path, perform_list, pedal_elongate=False):
#     score_midi_path = score_path + '.mid'
#     path_split = score_path.split('/')
#     composer_name = path_split[-1].split('.')[1]
#
#     if composer_name == 'Mendelssohn':
#         composer_name = 'Schubert'
#     composer_name_vec = composer_name_to_vec(composer_name)
#
#     XMLDocument, xml_notes = read_xml_to_notes(score_path)
#     score_midi = midi_utils.to_midi_zero(score_midi_path)
#     score_midi_notes = score_midi.instruments[0].notes
#     score_midi_notes.sort(key=lambda x:x.start)
#     match_list = matching.match_xml_to_midi(xml_notes, score_midi_notes)
#     score_pairs = matching.make_xml_midi_pair(xml_notes, score_midi_notes, match_list)
#     num_non_matched = check_pairs(score_pairs)
#     if num_non_matched > 100:
#         print("There are too many xml-midi matching errors")
#         return None
#
#     measure_positions = XMLDocument.get_measure_positions()
#     perform_features_piece = []
#     notes_graph = score_graph.make_edge(xml_notes)
#
#     for file in perform_list:
#         perf_name
#         # perf_score = evaluation.cal_score(perf_name)
#
#         perf_midi_name = path + perf_name + '.mid'
#         perf_midi = midi_utils.to_midi_zero(perf_midi_name)
#         perf_midi = midi_utils.add_pedal_inf_to_notes(perf_midi)
#         if pedal_elongate:
#             perf_midi = midi_utils.elongate_offset_by_pedal(perf_midi)
#         perf_midi_notes= perf_midi.instruments[0].notes
#         corresp_name = path + file
#         corresp = matching.read_corresp(corresp_name)
#
#         xml_perform_match = matching.match_score_pair2perform(score_pairs, perf_midi_notes, corresp)
#         perform_pairs = matching.make_xml_midi_pair(xml_notes, perf_midi_notes, xml_perform_match)
#         print("performance name is " + perf_name)
#         num_align_error = check_pairs(perform_pairs)
#
#         if num_align_error > 1000:
#             print('Too many align error in the performance')
#             continue
#         NUM_PERF_NOTES += len(score_midi_notes) - num_align_error
#         NUM_NON_MATCHED_NOTES += num_align_error
#         perform_features = extract_perform_features(XMLDocument, xml_notes, perform_pairs, perf_midi_notes, measure_positions)
#         perform_feat_score = {'features': perform_features, 'score': perf_score, 'composer': composer_name_vec, 'graph': notes_graph}
#
#         perform_features_piece.append(perform_feat_score)
#     if perform_features_piece == []:
#         return None
#     return perform_features_piece

def convert_features_to_vector(features, composer_vec):
    score_features = []
    perform_features = []
    for feature in features:
        score_features.append(
            [feature.midi_pitch, feature.duration, feature.beat_importance, feature.measure_length,
             feature.qpm_primo, feature.following_rest, feature.distance_from_abs_dynamic,
             feature.distance_from_recent_tempo, feature.beat_position, feature.xml_position,
             feature.grace_order, feature.preceded_by_grace_note, feature.followed_by_fermata_rest]
            + feature.pitch + feature.tempo + feature.dynamic + feature.time_sig_vec +
            feature.slur_beam_vec + composer_vec + feature.notation + feature.tempo_primo)

        perform_features.append(
            [feature.qpm, feature.velocity, feature.xml_deviation,
             feature.articulation, feature.pedal_refresh_time, feature.pedal_cut_time,
             feature.pedal_at_start, feature.pedal_at_end, feature.soft_pedal,
             feature.pedal_refresh,
             feature.pedal_cut, feature.qpm, feature.beat_dynamic, feature.measure_tempo, feature.measure_dynamic] \
            + feature.trill_param)

    return score_features, perform_features


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
        pair_index = pair_indexes[binary_index(xml_positions, measure_start)]
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


def binary_index_for_edge(alist, item):
    first = 0
    last = len(alist) - 1
    midpoint = 0

    if (item < alist[first][0]):
        return 0

    while first < last:
        midpoint = (first + last) // 2
        currentElement = alist[midpoint][0]

        if currentElement < item:
            if alist[midpoint + 1][0] > item:
                return midpoint
            else:
                first = midpoint + 1
            if first == last and alist[last][0] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint - 1
        else:
            if midpoint + 1 == len(alist):
                return midpoint
            while midpoint >= 1 and alist[midpoint - 1][0] == item:
                midpoint -= 1
                if midpoint == 0:
                    return midpoint
            return midpoint
    return last


def model_prediction_to_feature(prediction):
    output_features = []
    num_notes = len(prediction)
    for i in range(num_notes):
        pred = prediction[i]
        # feat = {'IOI_ratio': pred[0], 'articulation': pred[1], 'loudness': pred[2], 'xml_deviation': 0,
        feat = MusicFeature()
        feat.qpm = pred[0]
        feat.velocity = pred[1]
        feat.xml_deviation = pred[2]
        feat.articulation = pred[3]
        feat.pedal_refresh_time = pred[4]
        feat.pedal_cut_time = pred[5]
        feat.pedal_at_start = pred[6]
        feat.pedal_at_end = pred[7]
        feat.soft_pedal = pred[8]
        feat.pedal_refresh = pred[9]
        feat.pedal_cut = pred[10]

        feat.trill_param = pred[11:16]
        feat.trill_param[0] = feat.trill_param[0]
        feat.trill_param[1] = (feat.trill_param[1])
        feat.trill_param[2] = (feat.trill_param[2])
        feat.trill_param[3] = (feat.trill_param[3])
        feat.trill_param[4] = round(feat.trill_param[4])

        # if test_x[i][is_trill_index_score] == 1:
        #     print(feat.trill_param)
        output_features.append(feat)

    return output_features


def add_note_location_to_features(features, note_locations):
    for feat, loc in zip(features, note_locations):
        feat.note_location.beat = loc.beat
        feat.note_location.measure = loc.measure
    return features


def apply_tempo_perform_features(xml_doc, xml_notes, features, start_time=0, predicted=False):
    beats = xml_doc.get_beat_positions()
    num_beats = len(beats)
    num_notes = len(xml_notes)
    tempos = []
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
        tempo = Tempo(start_position, qpm, time_position=current_sec, end_xml=0, end_time=0)
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
        # passed_duration = note.note_duration.xml_position - corresp_tempo.xml_position
        passed_second = passed_duration / note.state_fixed.divisions / corresp_tempo.qpm * 60

        return previous_sec + passed_second

    for i in range(num_notes):
        note = xml_notes[i]
        feat = features[i]
        if not feat.xml_deviation == None:
            xml_deviation = feat.xml_deviation * note.state_fixed.divisions
            # if feat.xml_deviation >= 0:
            #     xml_deviation = (feat.xml_deviation ** 2) * note.state_fixed.divisions
            # else:
            #     xml_deviation = -(feat.xml_deviation ** 2) * note.state_fixed.divisions
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
            trill_density = trill_vec[0]
            last_velocity = trill_vec[1] * note.velocity
            first_note_ratio = trill_vec[2]
            last_note_ratio = trill_vec[3]
            up_trill = trill_vec[4]
            total_second = end_position - note.note_duration.time_position
            num_trills = int(trill_density * total_second)
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
                        new_note.pedals = None
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
                        new_note.pedals = None
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
            for j in range(i+1, num_notes):
                next_note = xml_notes[j]
                if not next_note.note_duration.duration == 0 \
                    and next_note.note_duration.xml_position == note.note_duration.xml_position \
                    and next_note.voice == note.voice:
                    next_second = next_note.note_duration.time_position
                    note.note_duration.seconds = (next_second - note.note_duration.time_position) / note.note_duration.num_grace
                    break

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
        current_position = pos_pair.xml_position
        current_time = pos_pair.time_position
        if current_position > previous_position >= 0:
            minimum_time_interval = (current_position - previous_position) / pos_pair.divisions / maximum_qpm * 60 + 0.01
        else:
            minimum_time_interval = 0
        if current_position > previous_position and current_time > previous_time + minimum_time_interval:
            if len(notes_in_chord) > 0:
                average_pos_pair = copy.copy(notes_in_chord[0])
                notes_in_chord_cleaned, average_pos_pair.time_position = get_average_onset_time(notes_in_chord)
                if len(cleaned_list) == 0 or average_pos_pair.time_position > cleaned_list[-1].time_position + (
                        (average_pos_pair.xml_position - cleaned_list[-1].xml_position) /
                        average_pos_pair.divisions / maximum_qpm * 60 + 0.01):
                    cleaned_list.append(average_pos_pair)
                    for note in notes_in_chord:
                        if note not in notes_in_chord_cleaned:
                            # print('the note is far from other notes in the chord')
                            mismatched_indexes.append(note.index)
                else:
                    # print('the onset is too close to the previous onset', average_pos_pair.xml_position, cleaned_list[-1].xml_position, average_pos_pair.time_position, cleaned_list[-1].time_position)
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
            # print('the note is too close to the previous note', current_position - previous_position, current_time - previous_time)
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
        melody_index = binary_index(melody_onset_positions, note.note_duration.xml_position)
        between_notes[melody_index].append(note)

    return between_notes

def apply_feat_to_a_note(note, feat, prev_vel):

    if not feat.articulation == None:
        note.note_duration.seconds *= 10 ** (feat.articulation)
        # note.note_duration.seconds *= feat.articulation
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
    index = binary_index(time_a, note.start)
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
    index = binary_index(time_a, note_start)
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

'''
moved save_midi_notes_as_piano_midi to xml_utils.py
-> have issue
'''


'''
moved apply_directions_to_notes to xml_utils.py
'''


'''
moved divide_cresc_staff to xml_utils.py
'''

'''
note_notation_to_vector -> feature_extraction.py
'''

'''
time_signature_to_vector -> feature_extraction.py
'''

'''
moved xml_notes_to_midi to xml_utils.py
'''


def check_pairs(pairs):
    non_matched = 0
    for pair in pairs:
        if pair == []:
            non_matched += 1

    print('Number of non matched pairs: ' + str(non_matched))
    return non_matched


def read_xml_to_notes(path):
    xml_name = path + 'musicxml_cleaned.musicxml'
    if not os.path.isfile(xml_name):
        xml_name = path + 'xml.xml'
    xml_object = MusicXMLDocument(xml_name)
    xml_notes = get_direction_encoded_notes(xml_object)

    return xml_object, xml_notes


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

'''
moved find_tempo_change to xml_utils.py
'''



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


def cal_tempo(xml_doc, xml_notes, pairs):
    beats = xml_doc.get_beat_positions()
    measure_positions = xml_doc.get_beat_positions(in_measure_level=True)
    tempo_change_positions = find_tempo_change(xml_notes)
    note_positions = [note.note_duration.xml_position for note in xml_notes]

    pairs, position_pairs = make_available_xml_midi_positions(pairs)

    def cal_tempo_by_positions(beats, position_pairs):
        tempos = []
        num_beats = len(beats)
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


            # current_index = current_pos_pair.index
            # feat = features[current_index]
            # feat.is_beat = True
            # num = 1
            # while current_index - num >= 0 and xml_notes[current_index - num].note_duration.xml_position == cur_xml:
            #     feat = features[current_index - num]
            #     feat.is_beat = True
            #     num += 1
            #
            # feat.is_beat = True

        return tempos

    beats_tempo = cal_tempo_by_positions(beats, position_pairs)
    measure_tempo = cal_tempo_by_positions(measure_positions, position_pairs)
    section_tempo = cal_tempo_by_positions(tempo_change_positions, position_pairs)

    return beats_tempo, measure_tempo, section_tempo


def cal_and_add_dynamic_by_position(features):
    num_notes = len(features)

    prev_beat = 0
    prev_measure = 0
    prev_section = 0
    prev_beat_index = 0
    prev_measure_index = 0
    prev_section_index = 0

    num_notes_in_beat = 0
    num_notes_in_measure = 0
    num_notes_in_section = 0

    temp_beat_dynamic = 0
    temp_measure_dynamic = 0
    temp_section_dynamic = 0

    for i in range(num_notes):
        feat = features[i]
        if not feat.align_matched:
            continue
        current_beat = feat.note_location.beat
        current_measure = feat.note_location.measure
        current_section = feat.note_location.section

        if current_beat > prev_beat and num_notes_in_beat != 0:
            prev_beat_dynamic = temp_beat_dynamic / num_notes_in_beat
            for j in range(prev_beat_index, i):
                features[j].beat_dynamic = prev_beat_dynamic
            temp_beat_dynamic = 0
            num_notes_in_beat = 0
            prev_beat = current_beat
            prev_beat_index = i

        if current_measure > prev_measure and num_notes_in_measure != 0:
            prev_measure_dynamic = temp_measure_dynamic / num_notes_in_measure
            for j in range(prev_measure_index, i):
                features[j].measure_dynamic = prev_measure_dynamic
            temp_measure_dynamic = 0
            num_notes_in_measure = 0
            prev_measure = current_measure
            prev_measure_index = i

        if current_section > prev_section and num_notes_in_section != 0:
            prev_section_dynamic = temp_section_dynamic / num_notes_in_section
            for j in range(prev_section_index, i):
                features[j].section_dynamic = prev_section_dynamic
            temp_section_dynamic = 0
            num_notes_in_section = 0
            prev_section = current_section
            prev_section_index = i

        temp_beat_dynamic += feat.velocity
        temp_measure_dynamic += feat.velocity
        temp_section_dynamic += feat.velocity
        num_notes_in_beat += 1
        num_notes_in_measure += 1
        num_notes_in_section += 1

    if num_notes_in_beat != 0:
        prev_beat_dynamic = temp_beat_dynamic / num_notes_in_beat
        prev_measure_dynamic = temp_measure_dynamic / num_notes_in_measure
        prev_section_dynamic = temp_section_dynamic / num_notes_in_section

        for j in range(prev_beat_index, num_notes):
            features[j].beat_dynamic = prev_beat_dynamic
        for j in range(prev_measure_index, num_notes):
            features[j].measure_dynamic = prev_measure_dynamic
        for j in range(prev_section_index, num_notes):
            features[j].section_dynamic = prev_section_dynamic

    return features


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
                    and not prev_note.note_duration.is_grace_note and not prev_note.note_duration.preceded_by_grace_note:
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
    global NUM_EXCLUDED_NOTES
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
    NUM_EXCLUDED_NOTES += len(mismatched_indexes)
    for index in mismatched_indexes:
        pairs[index] = []

    return pairs, available_pairs


def check_note_on_beat(note, measure_start, measure_length):
    note_position = note.note_duration.xml_position
    position_ratio = note_position / measure_length
    num_beat_in_measure = note.tempo.time_numerator

    on_beat = int(position_ratio * num_beat_in_measure) == (position_ratio * num_beat_in_measure)
    return on_beat


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


    if pairs[index] == []:
        for pair in pairs:
            if not pair ==[] and pair['midi'] == trills[0]:
                pair = []
        pairs[index] = {'xml': note, 'midi': trills[0]}

    for num in trills_vec:
        if math.isnan(num):
            print('trill vec is nan')
            num = 0

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
                        accident = accs.index(acc) - 2
                        temp_pair = {'pitch': pitch, 'accident': accident}
                        measure_accidentals.append(temp_pair)
                        break
    return measure_accidentals

'''
make_index_continous -> feature_extraciton.py
'''

def check_index_continuity(features):
    prev_beat = 0
    prev_measure = 0

    for feat in features:
        if feat.beat_index - prev_beat > 1:
            print('index_continuity', feat.beat_index, prev_beat)
        if feat.measure_index - prev_measure > 1:
            print('index_continuity', feat.measure_index, prev_measure)

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


def pedal_sigmoid(pedal_value, k=8):
    sigmoid_pedal = 127 / (1 + math.exp(-(pedal_value-64)/k))
    return int(sigmoid_pedal)


def composer_name_to_vec(composer_name):
    composer_name_list = ['Bach','Balakirev', 'Beethoven', 'Brahms', 'Chopin', 'Debussy', 'Glinka', 'Haydn',
                          'Liszt', 'Mozart', 'Prokofiev', 'Rachmaninoff', 'Ravel', 'Schubert', 'Schumann', 'Scriabin']

    index = composer_name_list.index(composer_name)
    one_hot_vec = [0] * len(composer_name_list)
    one_hot_vec[index] = 1

    return one_hot_vec


def read_score_perform_pair(path, perf_name, composer_name, means, stds, search_by_file_name=False):
    if search_by_file_name:
        folder_path = '/'.join(path.split('/')[0:-1]) + '/'
        score_midi_name = path + '.mid'
        perf_midi_name = folder_path + perf_name + '.mid'
        corresp_name = folder_path + perf_name + '_infer_corresp.txt'
        xml_object = MusicXMLDocument(path+'.musicxml')
        xml_notes = get_direction_encoded_notes(xml_object)

    else:
        score_midi_name = path + 'midi_cleaned.mid'
        if not os.path.isfile(score_midi_name):
            score_midi_name = path + 'midi.mid'
        perf_midi_name = path + perf_name + '.mid'
        corresp_name = path + perf_name + '_infer_corresp.txt'
        xml_object, xml_notes = read_xml_to_notes(path)
    score_midi = midi_utils.to_midi_zero(score_midi_name)
    score_midi_notes = score_midi.instruments[0].notes
    score_midi_notes.sort(key=lambda x:x.start)
    match_list = matching.match_xml_to_midi(xml_notes, score_midi_notes)
    score_pairs = matching.make_xml_midi_pair(xml_notes, score_midi_notes, match_list)
    measure_positions = xml_object.get_measure_positions()

    perf_midi = midi_utils.to_midi_zero(perf_midi_name)
    perf_midi = midi_utils.add_pedal_inf_to_notes(perf_midi)
    perf_midi_notes = perf_midi.instruments[0].notes
    corresp = matching.read_corresp(corresp_name)

    xml_perform_match = matching.match_score_pair2perform(score_pairs, perf_midi_notes, corresp)
    perform_pairs = matching.make_xml_midi_pair(xml_notes, perf_midi_notes, xml_perform_match)
    print("performance name is " + perf_name)
    check_pairs(perform_pairs)
    perform_features = extract_perform_features(xml_object, xml_notes, perform_pairs, perf_midi_notes, measure_positions)
    features = make_index_continuous(perform_features, score=True)
    composer_vec = composer_name_to_vec(composer_name)
    edges = score_graph.make_edge(xml_notes)

    for i in range(len(stds[0])):
        if stds[0][i] < 1e-4 or isinstance(stds[0][i], complex):
            stds[0][i] = 1

    test_x = []
    test_y = []
    note_locations = []
    for feat in features:
        temp_x = [(feat.midi_pitch - means[0][0]) / stds[0][0], (feat.duration - means[0][1]) / stds[0][1],
                  (feat.beat_importance - means[0][2]) / stds[0][2], (feat.measure_length - means[0][3]) / stds[0][3],
                  (feat.qpm_primo - means[0][4]) / stds[0][4], (feat.following_rest - means[0][5]) / stds[0][5],
                  (feat.distance_from_abs_dynamic - means[0][6]) / stds[0][6],
                  (feat.distance_from_recent_tempo - means[0][7]) / stds[0][7],
                  feat.beat_position, feat.xml_position, feat.grace_order,
                  feat.preceded_by_grace_note, feat.followed_by_fermata_rest] \
                 + feat.pitch + feat.tempo + feat.dynamic + feat.time_sig_vec + feat.slur_beam_vec + composer_vec + feat.notation + feat.tempo_primo
        temp_y = [feat.qpm, feat.velocity, feat.xml_deviation,
                  feat.articulation, feat.pedal_refresh_time, feat.pedal_cut_time,
                  feat.pedal_at_start, feat.pedal_at_end, feat.soft_pedal,
                  feat.pedal_refresh, feat.pedal_cut, feat.qpm, feat.beat_dynamic, feat.measure_tempo, feat.measure_dynamic] + feat.trill_param
        for i in range(19):
            temp_y[i] = (temp_y[i] - means[1][i]) / stds[1][i]
        test_x.append(temp_x)
        test_y.append(temp_y)
        note_locations.append(feat.note_location)

    return test_x, test_y, edges, note_locations


def get_all_words_from_folders(path):
    entire_words = []
    xml_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              f == 'musicxml_cleaned.musicxml']
    for xmlfile in xml_list:
        print(xmlfile)
        xml_doc = MusicXMLDocument(xmlfile)
        directions = xml_doc.get_directions()

        words = [dir for dir in directions if dir.type['type'] == 'words']
        # for wrd in words:
        #     entire_words.append(wrd.type['content'])
            # print(wrd.type['content'], wrd.state.qpm)
        for word in words:
            dynamic_vec = dir_enc.dynamic_embedding(word.type['content'], TEM_EMB_TAB, len_vec=3)
            print(word.type['content'], dynamic_vec, word.state.qpm)
        # print(words[0].type['content'], words[0].state.qpm, time_signatures[0], dynamic_vec)

    entire_words = list(set(entire_words))
    return entire_words


def cal_correlation(feat_a, feat_b):
    if len(feat_a) != len(feat_b):
        print('Error: length of two tempos are different, length a: {}, length b: {}'.format(len(feat_a), len(feat_b)))
        return None

    return scipy.stats.pearsonr(feat_a, feat_b)


def check_feature_pair_is_from_same_piece(prev_feat, new_feat, num_check=10):
    for i in range(num_check):
        if not prev_feat[i][0:4] == new_feat[i][0:4]:
            return False

    return True


class CorrelationResult:
    def __init__(self, path=None, slc_idx=None):
        self.tempo_r = []
        self.dynamic_r = []
        self.path_name = path
        self.slice_index = slc_idx
        self.tempo_features = []
        self.dynamic_features = []
        self.num_performance = 0

    def _append_result(self, tempo_r, velocity_r):
        self.tempo_r.append(tempo_r)
        self.dynamic_r.append(velocity_r)

    def cal_median(self):
        return np.median(self.tempo_r), np.median(self.dynamic_r)

    def cal_minimum(self):
        return min(self.tempo_r), min(self.dynamic_r)

    def cal_maximum(self):
        return max(self.tempo_r), max(self.dynamic_r)

    def _cal_correlation_of_features(self):
        for i in range(self.num_performance-1):
            for j in range(i+1, self.num_performance):
                tempo_r, _ = cal_correlation(self.tempo_features[i], self.tempo_features[j])
                dynamic_r, _ = cal_correlation(self.dynamic_features[i], self.dynamic_features[j])
                self._append_result(tempo_r, dynamic_r)


    def __str__(self):
        if len(self.tempo_r) > 0:
            tempo_r_median, dynamic_r_median = self.cal_median()
            tempo_r_min, dynamic_r_min = self.cal_minimum()
            tempo_r_max, dynamic_r_max = self.cal_maximum()
        else:
            return 'No correlation'

        return 'Piece: {}, Note index : {}, Tempo Med: {:.4f}, Min: {:.4f}, Max: {:.4f} - Dynamic Med: {:.4f}, Min: {:.4f}, Max: {:.4f}'\
            .format(self.path_name, self.slice_index, tempo_r_median, tempo_r_min, tempo_r_max, dynamic_r_median, dynamic_r_min, dynamic_r_max)


def cal_correlation_of_pairs_in_folder(path):
    features_in_folder = load_pairs_from_folder(path)
    if features_in_folder is None:
        return None
    num_performance = len(features_in_folder)
    if num_performance < 3:
        print('Error: There are only {} performances in the folder'.format(num_performance))
        return None

    num_notes = len(features_in_folder[0]['features'])
    beat_numbers = [x.note_location.beat for x in features_in_folder[0]['features']]
    slice_indexes = make_slicing_indexes_by_beat(beat_numbers, 30, overlap=True)
    correlation_result_total = []

    for slc_idx in slice_indexes:
        correlation_result = CorrelationResult(path, slc_idx)
        correlation_result.num_performance = num_performance

        for features in features_in_folder:
            sliced_features = features['features'][slc_idx[0]:slc_idx[1]]
            tempos, dynamics = perf_worm.cal_tempo_and_velocity_by_beat(sliced_features)
            correlation_result.tempo_features.append(tempos)
            correlation_result.dynamic_features.append(dynamics)
        correlation_result._cal_correlation_of_features()

        min_tempo_r, min_vel_r = correlation_result.cal_minimum()
        if min_tempo_r > 0.7:
            save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_', 10) + '_note{}-{}.png'.format(slc_idx[0], slc_idx[1])
            perf_worm.plot_normalized_feature(correlation_result.tempo_features, save_name)
            correlation_result_total.append(correlation_result)

    return correlation_result_total


def cal_actual_note_articulation(path):
    features_in_folder = load_pairs_from_folder(path, pedal_elongate=True)
    for feat in features_in_folder:
        num_notes = len(feat['features'])
        articulations = []
        for i in range(num_notes):
            pass


def check_data_split(path):
    entire_pairs = []
    num_train_pairs = 0
    num_valid_pairs = 0
    num_test_pairs = 0

    num_piece_train = 0
    num_piece_valid = 0
    num_piece_test = 0

    num_notes_in_train = 0
    num_notes_in_valid = 0
    num_notes_in_test = 0

    def load_pairs_and_add_num_notes(path):
        xml_object, xml_notes = read_xml_to_notes(path)
        filenames = os.listdir(path)
        perform_features_piece = []

        for file in filenames:
            if file[-18:] == '_infer_corresp.txt':
                perform_features_piece.append(len(xml_notes))
        if perform_features_piece == []:
            return None
        return perform_features_piece


    midi_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                 f == 'midi_cleaned.mid']
    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        skip = False
        for valid_piece in VALID_LIST:
            if valid_piece in foldername:
                skip = True
                break
        for test_piece in TEST_LIST:
            if test_piece in foldername:
                skip = True
                break
        if not skip:
            xml_name = foldername + 'musicxml_cleaned.musicxml'

            if os.path.isfile(xml_name):
                print(foldername)
                piece_pairs = load_pairs_and_add_num_notes(foldername)
                if piece_pairs is not None:
                    entire_pairs.append(piece_pairs)
                    num_train_pairs += len(piece_pairs)
                    num_piece_train += 1
                    for pair in piece_pairs:
                        num_notes_in_train += pair

    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        for valid_piece in VALID_LIST:
            if valid_piece in foldername:
                xml_name = foldername + 'musicxml_cleaned.musicxml'

                if os.path.isfile(xml_name):
                    print(foldername)
                    piece_pairs = load_pairs_and_add_num_notes(foldername)
                    if piece_pairs is not None:
                        entire_pairs.append(piece_pairs)
                        num_valid_pairs += len(piece_pairs)
                        num_piece_valid += 1
                        for pair in piece_pairs:
                            num_notes_in_valid += pair

    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        for test_piece in TEST_LIST:
            if test_piece in foldername:
                xml_name = foldername + 'musicxml_cleaned.musicxml'

                if os.path.isfile(xml_name):
                    print(foldername)
                    piece_pairs = load_pairs_and_add_num_notes(foldername)
                    if piece_pairs is not None:
                        entire_pairs.append(piece_pairs)
                        num_test_pairs += len(piece_pairs)
                        num_piece_test += 1
                        for pair in piece_pairs:
                            num_notes_in_test += pair

    print('Number of train pieces: ', num_piece_train, 'valid pieces: ', num_piece_valid, 'test pieces: ', num_piece_test)
    print('Number of train pairs: ', num_train_pairs, 'valid pairs: ', num_valid_pairs, 'test pairs: ', num_test_pairs)
    print('Number of train notes: ', num_notes_in_train, 'valid notes: ', num_notes_in_valid, 'test notes: ',
          num_notes_in_test)
    return entire_pairs, num_train_pairs, num_valid_pairs, num_test_pairs



