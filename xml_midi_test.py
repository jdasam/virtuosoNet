#!/usr/bin/env python
# -*- coding: utf-8 -*-

from musicxml_parser.mxp import MusicXMLDocument
import midi_utils.midi_utils as midi_utils
import xml_matching
import pickle
import score_as_graph as score_graph

xml_matching.read_all_tempo_vector('chopin_cleaned/Chopin/')

# folderDir = 'mxp/testdata/chopin10-3/'
# folderDir = 'chopin/Chopin_Polonaises/61/'
folderDir = 'test_pieces/schumann/'
# folderDir = 'mxp/testdata/dummy/chopin_ballade3/'
artistName = 'SunMeiting08'
# artistName = 'CHEN03'
xmlname = 'musicxml_cleaned.musicxml'
# xmlname = 'xml.xml'
midiname= 'midi_cleaned.mid'


XMLDocument = MusicXMLDocument(folderDir + xmlname)
melody_notes = xml_matching.extract_notes(XMLDocument, melody_only=False, grace_note=True)
melody_notes.sort(key=lambda x: x.note_duration.time_position)

# for note in melody_notes:
#     print(note.note_notations.is_beam_start, note.note_duration.xml_position, note.pitch)
score_midi = midi_utils.to_midi_zero(folderDir + midiname)
perform_midi = midi_utils.to_midi_zero(folderDir + artistName + '.mid')
# perform_midi = midi_utils.elongate_offset_by_pedal(perform_midi)
perform_midi = midi_utils.add_pedal_inf_to_notes(perform_midi)
score_midi_notes = score_midi.instruments[0].notes
score_midi_notes.sort(key=lambda note: note.start)

notes_graph = score_graph.make_edge(melody_notes)

# for part in XMLDocument.parts:
#     for measure in part.measures:
#         print(measure.first_ending_start, measure.first_ending_stop, measure.fine, measure.dacapo)

perform_midi_notes = perform_midi.instruments[0].notes
corresp = xml_matching.read_corresp(folderDir + artistName + "_infer_corresp.txt")
score_pairs, perform_pairs = xml_matching.match_xml_midi_perform(melody_notes,score_midi_notes, perform_midi_notes, corresp)
xml_matching.check_pairs(score_pairs)

# for pair in perform_pairs:
#     if not pair ==[]:
#         print(pair['midi'])
#         print(pair['xml'])

# for note in perform_midi_notes:
#     print(note.pedal_at_start, note.pedal_at_end, note.pedal_refresh, note.pedal_cut)

# for note in melody_notes:
#     print(note.dynamic.cresciuto)
    # if note.note_notations.slurs:
    #     print(note)
    #     print(note.measure_number, note.note_notations.slurs[0].index, note.note_notations.slurs[0].xml_position ,note.note_notations.slurs[0].end_xml_position)
    # print(note.note_duration.is_grace_note, note.note_duration.grace_order)

# Check xml notes
#  for i in range(len(melody_notes)-1):
#     # diff = (melody_notes[i+1].note_duration.time_position - melody_notes[i].note_duration.time_position) * 10000
#     # print(diff, melody_notes[i].note_duration.xml_position)
#     print(melody_notes[i].pitch, melody_notes[i].note_duration.xml_position,  melody_notes[i].note_duration.time_position)
#
# for i in range(100):
#     print(melody_notes[i].note_duration.time_position, score_midi_notes[i].start, melody_notes[i].note_duration.time_position - score_midi_notes[i].start)
# Check xml_midi_pairs
# for note in perform_midi_notes:
#     print [note.pedal_at_start, note.pedal_at_end, note.pedal_refresh, note.pedal_refresh_before,
#            note.soft_pedal,
#            note.sostenuto_at_start, note.sostenuto_at_end, note.sostenuto_refresh, note.sostenuto_refresh_before]


# non_matched_count = 0
# for i in range(len(perform_pairs)):
#     pair = perform_pairs[i]
#     if pair ==[]:
#         non_matched_count += 1
#         print(melody_notes[i])
#     #     print (pair)
#     # else:
#     #     print('XML Note pitch:', pair['xml'].pitch , ' and time: ', pair['xml'].note_duration.time_position , '-- MIDI: ', pair['midi'])
# print('Number of non matched XML notes: ', non_matched_count)

directions, time_signatures = xml_matching.extract_directions(XMLDocument)
# for dir in directions:
#     print(dir)


melody_notes = xml_matching.apply_directions_to_notes(melody_notes, directions, time_signatures)
for note in melody_notes:
    if note.note_notations.is_trill:
        print(note)
#
# for note in melody_notes:
#     # print(note.pitch, note.note_duration.xml_position, note.dynamic.absolute, note.tempo)
#     print(note.pitch, note.note_duration.xml_position, note.dynamic.absolute, note.tempo.absolute) #, note.note_notations)
#     # print(vars(note.note_notations))
#     if not note.tempo.relative == []:
#         for rel in note.tempo.relative:
#             print(rel)
#             print(rel.end_xml_position)

# for dir in directions:
#     # print(dir)
#     if not dir.type == None and dir.type.keys()[0] == 'words':
#         # print(dir)
#         pass

# words = xml_matching.get_all_words_from_folders('chopin_cleaned/Beethoven/')
# for wrd in words:
#     print (wrd)


#
# melody = xml_matching.extract_melody_only_from_notes(melody_notes)
# for note in melody:
#     note_index = melody_notes.index(note)
#     if not features[note_index] == []:
#         print features[note_index]['IOI_ratio'], note.note_duration.after_grace_note

# print(len(features[0]['dynamic']), len(features[0]['tempo']))

#
# ioi_list = []
# articul_list =[]
# loudness_list = []
# for feat in features:
#     print(feat.dynamic, feat.tempo)
    # if not feat['IOI_ratio'] == None:
        # ioi_list.append(feat['IOI_ratio'])
        # articul_list.append(feat['articulation'])
        # loudness_list.append(feat['loudness'])


# feature_list = [ioi_list, articul_list, loudness_list]
# #
# ioi_list = [feat['IOI_ratio'] for feat in features ]


measure_positions = xml_matching.extract_measure_position(XMLDocument)
# previous_pos=0
# for i in range(len(measure_positions)-1):
#     print('measure ' + str(i+1) + ' position is ' + str(measure_positions[i]) + ' and length is' + str(measure_positions[i+1]-measure_positions[i]))
features = xml_matching.extract_perform_features(XMLDocument, melody_notes, perform_pairs, perform_midi_notes, measure_positions)


# new_midi = xml_matching.applyIOI(melody_notes, score_midi_notes, features, feature_list)

# new_xml = xml_matching.apply_tempo_perform_features(XMLDocument, melody_notes, features, start_time = perform_midi_notes[0].start)
# new_xml = xml_matching.apply_tempo_perform_features(XMLDocument, melody_notes, features, start_time = 0.518162)
# new_xml = xml_matching.apply_time_position_features(melody_notes, features, start_time = perform_midi_notes[0].start)

new_midi = xml_matching.xml_notes_to_midi(melody_notes)
#
# for note in new_midi:
#     print(note)
#
xml_matching.save_midi_notes_as_piano_midi(new_midi, 'my_first_midi.mid', bool_pedal=True)


# load and save data
# chopin_pairs = xml_matching.load_entire_subfolder('chopin_cleaned/')
# print(chopin_pairs)

#
# with open("pairs_entire4.dat", "wb") as f:
#     pickle.dump(chopin_pairs, f, protocol=2)
