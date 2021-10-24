'''
This code is to forcely fit one MIDI file to certain time mark of the other MIDI
'''

from .pyScoreParser.midi_utils import midi_utils
from .pyScoreParser import xml_midi_matching as matching
import argparse
from pathlib import Path
import shutil
import os
import subprocess
import numpy as np
import pretty_midi
from copy import deepcopy

class PerformMIDI:
  def __init__(self, midi_path):
    self.midi_path = midi_path
    self.midi = midi_utils.to_midi_zero(self.midi_path)
    self.midi_events = [note for instrument in self.midi.instruments for note in instrument.notes] + [event for instrument in self.midi.instruments for event in instrument.control_changes]
    self.midi_events.sort(key=lambda x: x.start if hasattr(x, 'start') else x.time)

  def force_fit(self, other_midi, align_dir, fit_timemark):
    '''
    other_midi: PerformMIDI
    ''' 
    align_name = self.midi_path.parent / (f"align_coresp_{self.midi_path.stem}_{other_midi.midi_path.stem}.txt") 
    if not align_name.exists():
      self.align(other_midi, align_dir)
    
    corresp= matching.read_corresp(align_name)
    clean_corresp = [x for x in corresp if (x['alignOntime']!="-1" and x['refOntime']!="-1" ) ]
    # time_match = {'ref_time':[float(x['refOntime']) for x in clean_corresp], 'perf_time': [float(x['alignOntime']) for x in clean_corresp]} 
    time_match = np.asarray([(float(x['refOntime']), float(x['alignOntime'])) for x in clean_corresp ])

    corresp_time = [0] 
    for time in fit_timemark:
      idx = np.argmin(np.abs(time_match[:,0]-time))
      corresp_time.append(time_match[idx,1])

    fit_timemark = [0] + fit_timemark

    new_events = deepcopy(self.midi_events)

    for i in range(1, len(corresp_time)):
      self.adjust_midi(corresp_time[i-1], corresp_time[i], fit_timemark[i-1], fit_timemark[i])

  def adjust_midi(self, start_time, end_time, ref_start, ref_end):
    for event in self.midi_events:
      if hasattr(event, 'start'):
        if start_time <= event.start < end_time:
          event.start = interpolation(event.start, start_time, end_time, ref_start, ref_end)
        if start_time <= event.end < end_time:
          event.end = interpolation(event.end, start_time, end_time, ref_start, ref_end)
      else:
        if start_time <= event.time < end_time:
          event.time = interpolation(event.time, start_time, end_time, ref_start, ref_end)
  

  def make_time_matching_line(self):
    return

  def align(self, other_midi, align_dir):
    '''
    other_midi: PerformMIDI
    '''
    shutil.copy(self.midi_path, align_dir / 'perform_a.mid')
    shutil.copy(other_midi.midi_path, align_dir / 'ref_perform.mid')
    current_dir = os.getcwd()
    try:
        os.chdir(align_dir)
        subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "ref_perform", "perform_a"])
    except:
        print('Error to process {}'.format(self.midi_path))
    else:
        align_success = True
    if align_success:
        shutil.move('perform_a_corresp.txt', self.midi_path.parent / (f"align_coresp_{self.midi_path.stem}_{other_midi.midi_path.stem}.txt") )
        # shutil.move('infer_match.txt', self.midi_path.replace('.mid', '_infer_match.txt')   )
        # shutil.move('infer_spr.txt', self.midi_path.replace('.mid', '_infer_spr.txt'))
        # shutil.move('score_spr.txt', os.path.join(align_dir, '_score_spr.txt'))
        os.chdir(current_dir)

def interpolation(a0, a1,a2, b1,b2):
  return b1+ (a0-a1)/(a2-a1)*(b2-b1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--align_program_dir", type=Path, default="/Users/jeongdasaem/Documents/AlignmentTool_v190813/")
  parser.add_argument("--ref_midi", type=Path)
  parser.add_argument("--perf_midi", type=Path)
  parser.add_argument("--output_name", type=str)


  args = parser.parse_args()
  perf_midi = PerformMIDI(args.perf_midi)
  ref_midi = PerformMIDI(args.ref_midi)

  time_marks = [330, 37363, 73891, 84528, 106693, 126726, 149099, 181000, 182500]
  time_marks = [x/1000 for x in time_marks]

  perf_midi.force_fit(ref_midi, args.align_program_dir, time_marks)

  piano_midi = pretty_midi.PrettyMIDI()
  piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
  piano = pretty_midi.Instrument(program=piano_program)
  piano.notes = [x for x in perf_midi.midi_events if hasattr(x, 'start')]
  piano.control_changes =  [x for x in perf_midi.midi_events if hasattr(x, 'time')]
  piano_midi.instruments.append(piano)
  piano_midi.write(args.output_name)