from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess
from. import utils as utils
import argparse
import pretty_midi

# find midi files in INPUT_DIR, and align it using Nakamura's alignment tool.
# (https://midialignment.github.io/demo.html)
# midi.mid in same subdirectory will be regarded as score file.
# make alignment result files in same directory. read Nakamura's manual for detail.

INPUT_DIR = '/Users/jeongdasaem/Documents/GitHub/virtuosoNet/pyScoreParser/EmotionData/'

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default=INPUT_DIR,
                    help="Abs path to midi folder")
parser.add_argument("--align_dir", default='/Users/jeongdasaem/Documents/AlignmentTool_v190813',
                    help="Abs path to Nakamura's Alignment tool")
args = parser.parse_args()
INPUT_DIR = args.input_dir

os.chdir(args.align_dir)

'''
# read from text list
f = open('temp_fix.txt', 'rb')
lines = f.readlines()
f.close()
midi_files = [el.strip() for el in lines]
'''

# read from folder
midi_files = utils.find_files_in_subdir(INPUT_DIR, '*.mid')
xml_files = utils.find_files_in_subdir(INPUT_DIR, '*.musicxml')
score_file_name = {'piece': [], 'path': []}

for xml_file in xml_files:
    split_name = xml_file.split('/')[-1].split('.')
    score_file_name['piece'].append('.'.join(split_name[1:4]))
    score_file_name['path'].append(xml_file)

print(score_file_name)

n_match = 0
n_unmatch = 0
for midi_file in midi_files:
    file_name = midi_file.split('/')[-1]
    if file_name.split('.')[0][0].isdigit():
        continue

    if os.path.isfile(midi_file.replace('.mid', '_infer_corresp.txt')):
        n_match += 1
        continue

    file_folder, file_name = utils.split_head_and_tail(midi_file)
    perform_midi = midi_file
    piece_name = '.'.join(file_name.split('.')[0:3])

    if piece_name in score_file_name['piece']:
        piece_index = score_file_name['piece'].index(piece_name)
    else:
        continue
    score_midi = ".".join(score_file_name['path'][piece_index].split('.')[0:-1]) + ".mid"

    print(perform_midi)
    print(score_midi)

    mid = pretty_midi.PrettyMIDI(score_midi)

    n_notes = len(mid.instruments[0].notes)

    '''
    if n_notes >= 8000:
        n_unmatch +=1
        continue
    '''

    shutil.copy(perform_midi, os.path.join(args.align_dir, 'infer.mid'))
    shutil.copy(score_midi, os.path.join(args.align_dir, 'score.mid'))

    try:
        subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
    except:
        print('Error to process {}'.format(midi_file))
        pass
    else:
        shutil.move('infer_corresp.txt', midi_file.replace('.mid', '_infer_corresp.txt'))
        shutil.move('infer_match.txt', midi_file.replace('.mid', '_infer_match.txt'))
        shutil.move('infer_spr.txt', midi_file.replace('.mid', '_infer_spr.txt'))
        shutil.move('score_spr.txt', os.path.join(args.align_dir, '_score_spr.txt'))
print('match:{:d}, unmatch:{:d}'.format(n_match, n_unmatch))


