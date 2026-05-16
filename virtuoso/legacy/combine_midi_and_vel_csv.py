import csv
import os
import pyScoreParser.midi_utils.midi_utils as midi_utils

def combine_midi_and_vel_csv(path, output_path):
    # the code is to combine a specific data type, which was used for the PerformScore
    midi_path = path + '.mid'
    csv_path = path + '_vel.csv'
    midi_file = midi_utils.to_midi_zero(midi_path)
    midi_notes = midi_file.instruments[0].notes

    midi_notes.sort(key=lambda x: (x.start, x.pitch))

    with open(csv_path, newline='') as csvfile:
        velocities = list(csv.reader(csvfile))

    for note, vel in zip(midi_notes, velocities[0]):
        note.velocity = int(vel)

    midi_file.write(output_path)


def find_midi_in_path(path):
    perf_list = [file for file in os.listdir(path) if file.endswith('.mid') and file not in ('(midi).mid, (midi)old.mid') and '-mix.mid' not in file]
    return perf_list

def save_for_all_subfolder(path):
    midi_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                 f == '(midi).mid']
    print(midi_list)
    for piece in midi_list:
        piece_path = os.path.split(piece)[0] + '/'
        perf_list = find_midi_in_path(piece_path)
        output_path = piece_path + 'vel_mid/'
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        for perf_path in perf_list:
            perf_name = os.path.splitext(perf_path)[0]
            full_path = piece_path + perf_name
            output_name = output_path + perf_name + '_vel.mid'
            print(full_path)
            combine_midi_and_vel_csv(full_path, output_name)


save_for_all_subfolder('/Users/jeongdasaem/Documents/GitHub/virtuosoNet/test_pieces/performScore_test')