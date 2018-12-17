#!/usr/bin/env python
# -*- coding: utf-8 -*-

from musicxml_parser.mxp import MusicXMLDocument
import midi_utils.midi_utils as midi_utils
import xml_matching
import pickle
import score_as_graph as score_graph

# folderDir = 'mxp/testdata/chopin10-3/'
# folderDir = 'chopin/Chopin_Polonaises/61/'
folderDir = 'chopin_cleaned/Chopin/Chopin_Etude_op_10/3/'
# folderDir = 'mxp/testdata/dummy/chopin_ballade3/'
artistName = 'SunMeiting08'
# artistName = 'CHEN03'
xmlname = 'musicxml_cleaned.musicxml'
# xmlname = 'xml.xml'

XMLDocument = MusicXMLDocument(folderDir + xmlname)
melody_notes = xml_matching.extract_notes(XMLDocument, melody_only=False, grace_note=True)
melody_notes.sort(key=lambda x: x.note_duration.time_position)

new_midi = xml_matching.xml_notes_to_midi(melody_notes)
xml_matching.save_midi_notes_as_piano_midi(new_midi, 'my_first_midi.mid', bool_pedal=True)

