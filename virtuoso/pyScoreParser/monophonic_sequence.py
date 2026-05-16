from pathlib import Path
from .data_class import ScoreData
from .xml_utils import xml_notes_to_midi
from .midi_utils.midi_utils import save_midi_notes_as_piano_midi_without_pedal
import argparse
import pickle
import pandas as pd

def notes_sequence_to_list_of_dict(notes):
  measure_number = 1 # measure_number starts from 1
  entire_measures = []
  measure = []
  for note in notes:
      if note.measure_number != measure_number:
          entire_measures.append(measure)
          measure = []
          measure_number = note.measure_number
      if note.pitch:
          pitch = note.pitch[1]
      else:
          pitch = 0 # the note is a rest 
      measure.append({'pitch':pitch, 'duration':note.note_duration.duration / note.state_fixed.divisions})
  return entire_measures

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--xml_dataset_dir', type=str)
  parser.add_argument('--output_dir', type=str)

  args = parser.parse_args()
  output_dir = Path(args.output_dir)

  xml_dataset_dir = Path(args.xml_dataset_dir)
  xml_list = list(xml_dataset_dir.rglob('*.xml')) + list(xml_dataset_dir.rglob('*.musicxml'))
  voice_list = [1,5]
  for xml_path in xml_list:
    try:
      score = ScoreData(xml_path, None, '', read_xml_only=True)
    except:
      continue
    for voice_idx in voice_list:
      notes = score.xml_obj.get_monophonic_notes_by_voice(voice_idx=voice_idx, part_idx=0) # voice_idx in music_xml starts from 1
      notes_list_of_dict = notes_sequence_to_list_of_dict(notes)
    
      piece_name = '.'.join(str(xml_path.parent).split('/')[-3:])
      if 'chopin_cleaned' in piece_name:
        piece_name = piece_name.replace('chopin_cleaned.', '')
    
      output_path = output_dir/ f"{piece_name}_voice{voice_idx}.pkl"
      midi_out_path =  output_dir/ f"{piece_name}_voice{voice_idx}.mid"
      csv_out_path =  output_dir/ f"{piece_name}_voice{voice_idx}.csv"

      with open(output_path, 'wb') as f:
        pickle.dump(notes_list_of_dict, f)
      pd.DataFrame(notes_list_of_dict).to_csv(str(csv_out_path))
      notes_without_rest = [note for note in notes if note.pitch]
      midi_notes, _ = xml_notes_to_midi(notes_without_rest)
      save_midi_notes_as_piano_midi_without_pedal(midi_notes, midi_out_path)