#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .musicxml_parser.mxp import MusicXMLDocument
from .midi_utils import midi_utils as midi_utils
from . import xml_matching as xml_matching, score_as_graph as score_graph

import pickle

# folderDir = 'mxp/testdata/chopin10-3/'
# folderDir = 'chopin/Chopin_Polonaises/61/'
folderDir = 'chopin_cleaned/Beethoven/Piano_Sonatas/32-1/'
# folderDir = 'mxp/testdata/dummy/chopin_ballade3/'
# artistName = 'SunMeiting08'
# artistName = 'CHEN03'
xmlname = 'musicxml_cleaned.musicxml'
# xmlname = 'xml.xml'

XMLDocument = MusicXMLDocument(folderDir + xmlname)
melody_notes = xml_matching.extract_notes(XMLDocument, melody_only=False, grace_note=True)
melody_notes.sort(key=lambda x: x.note_duration.time_position)

new_midi = xml_matching.xml_notes_to_midi(melody_notes)
xml_matching.save_midi_notes_as_piano_midi(new_midi, folderDir+'midi_cleaned.mid', bool_pedal=True)

