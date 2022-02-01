import torch


import torch
import torch.nn as nn
from .utils import note_feature_to_beat_mean, note_tempo_infos_to_beat


class TempoVecSelector(nn.Module):
  def __init__(self, stats):
    super(TempoVecSelector, self).__init__()
    self.key_to_dim = stats['key_to_dim']['input']
    self.qpm_primo_idx = [self.key_to_dim['qpm_primo'][0]]
    self.tempo_primo_idx = list(range(self.key_to_dim['tempo_primo'][0], self.key_to_dim['tempo_primo'][1]))
    self.tempo_vec_idx = list(range(self.key_to_dim['tempo'][0], self.key_to_dim['tempo'][1]))

  def forward(self, x, note_locations):
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
      x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, True)
    measure_numbers = note_locations['measure']
    max_num_measures = torch.max(measure_numbers - measure_numbers[:,0:1]) + 1
    qpm_primo = x[:, :, self.qpm_primo_idx]
    tempo_primo = x[:, :, self.tempo_primo_idx ]

    beat_qpm_primo = qpm_primo[:,0:1].repeat(1, max_num_measures, 1)
    beat_tempo_primo = tempo_primo[:,0:1].repeat(1,max_num_measures, 1)
    beat_tempo_vector = note_feature_to_beat_mean(x[:,:,self.tempo_vec_idx], measure_numbers, use_mean=False)
    # return torch.cat([beat_qpm_primo, beat_tempo_primo, beat_tempo_vector], dim=-1)


    beat_numbers = note_locations['beat']
    max_num_beats = torch.max(beat_numbers - beat_numbers[:,0:1]) + 1

    qpm_primo = x[:, :, self.qpm_primo_idx]
    tempo_primo = x[:, :, self.tempo_primo_idx]
    # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
    beat_qpm_primo = qpm_primo[:, 0:1].repeat((1, max_num_beats, 1))
    beat_tempo_primo = tempo_primo[:, 0:1].repeat((1, max_num_beats, 1))
    beat_tempo_vector = note_feature_to_beat_mean(x[:,:,self.tempo_vec_idx], beat_numbers, use_mean=False)

    return torch.cat((beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), dim=-1)

class TempoVecMeasSelector(nn.Module):
  def __init__(self, stats):
    super(TempoVecMeasSelector, self).__init__()
    self.key_to_dim = stats['key_to_dim']['input']
    self.qpm_primo_idx = [self.key_to_dim['qpm_primo'][0]]
    self.tempo_primo_idx = list(range(self.key_to_dim['tempo_primo'][0], self.key_to_dim['tempo_primo'][1]))
    self.tempo_vec_idx = list(range(self.key_to_dim['tempo'][0], self.key_to_dim['tempo'][1]))


  def forward(self, x, note_locations):
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
      x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, True)
    measure_numbers = note_locations['measure']
    max_num_measures = torch.max(measure_numbers - measure_numbers[:,0:1]) + 1
    qpm_primo = x[:, :, self.qpm_primo_idx]
    tempo_primo = x[:, :, self.tempo_primo_idx]

    beat_qpm_primo = qpm_primo[:,0:1].repeat(1, max_num_measures, 1)
    beat_tempo_primo = tempo_primo[:,0:1].repeat(1,max_num_measures, 1)
    beat_tempo_vector = note_feature_to_beat_mean(x[:,:,self.tempo_vec_idx], measure_numbers, use_mean=False)

    return torch.cat([beat_qpm_primo, beat_tempo_primo, beat_tempo_vector], dim=-1)

    # num_measures = measure_numbers[-1] - measure_numbers[0] + 1

    # qpm_primo = x[:, :, QPM_PRIMO_IDX].view(1, -1, 1)
    # tempo_primo = x[:, :, TEMPO_PRIMO_IDX:].view(1, -1, 2)
    # # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
    # beat_qpm_primo = qpm_primo[0, 0, 0].repeat((1, num_measures, 1))
    # beat_tempo_primo = tempo_primo[0, 0, :].repeat((1, num_measures, 1))
    # beat_tempo_vector = note_feature_to_beat_mean(x[:,:,TEMPO_IDX:TEMPO_IDX+5], measure_numbers, use_mean=False)

    # return torch.cat((beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), dim=-1)