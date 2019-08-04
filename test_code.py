#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .musicxml_parser import MusicXMLDocument
from .midi_utils import midi_utils as midi_utils
from . import xml_matching
from . import xml_midi_matching as matching
from . import score_as_graph as score_graph
import pickle


# folderDir = 'mxp/testdata/chopin10-3/'
# folderDir = 'chopin/Chopin_Polonaises/61/'
folderDir = 'pyScoreParser/chopin_cleaned/Schubert/Piano_Sonatas/664-1/'
# folderDir = 'mxp/testdata/dummy/chopin_ballade3/'
artistName = 'BuiJL06'
# artistName = 'CHEN03'
xmlname = 'musicxml_cleaned.musicxml'
# xmlname = 'xml.xml'
midiname= 'midi_cleaned.mid'


XMLDocument = MusicXMLDocument(folderDir + xmlname)
melody_notes = xml_matching.get_direction_encoded_notes(XMLDocument)

# for note in melody_notes:
#     print(note.note_notations.is_beam_start, note.note_duration.xml_position, note.pitch)
score_midi = midi_utils.to_midi_zero(folderDir + midiname)
perform_midi = midi_utils.to_midi_zero(folderDir + artistName + '.mid')
perform_midi = midi_utils.elongate_offset_by_pedal(perform_midi)
perform_midi = midi_utils.add_pedal_inf_to_notes(perform_midi)
score_midi_notes = score_midi.instruments[0].notes
score_midi_notes.sort(key=lambda note: note.start)


notes_graph = score_graph.make_edge(melody_notes)



# for part in XMLDocument.parts:
#     for measure in part.measures:
#         print(measure.first_ending_start, measure.first_ending_stop, measure.fine, measure.dacapo)

perform_midi_notes = perform_midi.instruments[0].notes
corresp = matching.read_corresp(folderDir + artistName + "_infer_corresp.txt")
score_pairs, perform_pairs = matching.match_xml_midi_perform(melody_notes,score_midi_notes, perform_midi_notes, corresp)
xml_matching.check_pairs(score_pairs)


for note in melody_notes:
    if note.note_notations.is_trill:
        print('trill note', note, note.measure_number)
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


measure_positions = XMLDocument.get_measure_positions()
# previous_pos=0
# for i in range(len(measure_positions)-1):
#     print('measure ' + str(i+1) + ' position is ' + str(measure_positions[i]) + ' and length is' + str(measure_positions[i+1]-measure_positions[i]))
features = xml_matching.extract_perform_features(XMLDocument, melody_notes, perform_pairs, perform_midi_notes, measure_positions)
for feat in features:
    if feat.grace_order != 0:
        print(feat.xml_deviation)

print(features[1000].note_location.measure, features[1176].note_location.measure)
# new_midi = xml_matching.applyIOI(melody_notes, score_midi_notes, features, feature_list)

# new_xml = xml_matching.apply_tempo_perform_features(XMLDocument, melody_notes, features, start_time = perform_midi_notes[0].start)
# new_xml = xml_matching.apply_tempo_perform_features(XMLDocument, melody_notes, features, start_time = 0.518162)
# new_xml = xml_matching.apply_time_position_features(melody_notes, features, start_time = perform_midi_notes[0].start)

new_midi = xml_matching.xml_notes_to_midi(melody_notes)
#
# for note in new_midi:
#     print(note)
#
# xml_matching.save_midi_notes_as_piano_midi(new_midi, 'my_first_midi.mid', bool_pedal=True)


# load and save data
# chopin_pairs = xml_matching.load_entire_subfolder('chopin_cleaned/')
# print(chopin_pairs)

#
# with open("pairs_entire4.dat", "wb") as f:
#     pickle.dump(chopin_pairs, f, protocol=2)
