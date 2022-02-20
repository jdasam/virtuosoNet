import pretty_midi
import argparse
from pathlib import Path

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_path", type=Path)
  parser.add_argument("-o", "--output_path", type=Path)
  parser.add_argument("-r", "--range", type=float)
  parser.add_argument("-m", "--mean", type=float)


  args = parser.parse_args()
  input_path = args.input_path

  midi_file_list = input_path.glob("*.mid")
  output_dir = args.output_path 
  output_dir.mkdir(exist_ok=True)
  min_vel= args.mean
  dynamic_range = args.range

  for midi_fname in midi_file_list:
    midi = pretty_midi.PrettyMIDI(str(midi_fname))

    for instruments in midi.instruments:
        for midi_note in instruments.notes:
            midi_note.velocity = max(min(round((midi_note.velocity - 64) * dynamic_range + min_vel), 127), 1)
    
    save_name = str(output_dir/midi_fname.name)
    midi.write(save_name)

  # for midi_fname in midi_file_list:
  #   midi_fname = str(midi_fname)
  #   for min_vel in range(64, 72, 4):
  #     for dynamic_range in range(70, 90, 4):
  #       dynamic_range /= 100
  #       midi = pretty_midi.PrettyMIDI(midi_fname)

  #       for instruments in midi.instruments:
  #           for midi_note in instruments.notes:
  #               midi_note.velocity = max(min(round((midi_note.velocity - 64) * dynamic_range + min_vel), 127), 1)
        
  #       save_name = f"{output_dir}/{Path(midi_fname).stem}_min{min_vel}_dyn{dynamic_range}.mid"

  #       # save_name = f"{output_dir}/{midi_fname.split('/')[1].split('_')[0]}_min{min_vel}_dyn{dynamic_range}.mid"
  #       midi.write(save_name)
