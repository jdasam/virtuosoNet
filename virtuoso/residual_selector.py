import torch


import torch
import torch.nn as nn
from .model_constants import TEMPO_PRIMO_IDX, QPM_PRIMO_IDX, TEMPO_IDX
from .utils import note_feature_to_beat_mean, note_tempo_infos_to_beat


class TempoVecSelector(nn.Module):
    def __init__(self):
        super(TempoVecSelector, self).__init__()

    def forward(self, x, note_locations):
        beat_numbers = note_locations['beat']
        num_beats = beat_numbers[-1] - beat_numbers[0] + 1

        qpm_primo = x[:, :, QPM_PRIMO_IDX].view(1, -1, 1)
        tempo_primo = x[:, :, TEMPO_PRIMO_IDX:].view(1, -1, 2)
        # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
        beat_qpm_primo = qpm_primo[0, 0, 0].repeat((1, num_beats, 1))
        beat_tempo_primo = tempo_primo[0, 0, :].repeat((1, num_beats, 1))
        beat_tempo_vector = note_feature_to_beat_mean(x[:,:,TEMPO_IDX:TEMPO_IDX+5], beat_numbers, use_mean=False)

        return torch.cat((beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), dim=-1)

class TempoVecMeasSelector(nn.Module):
    def __init__(self):
        super(TempoVecMeasSelector, self).__init__()

    def forward(self, x, note_locations):
        measure_numbers = note_locations['measure']
        max_num_measures = torch.max(measure_numbers - measure_numbers[:,0:1]) + 1
        qpm_primo = x[:, :, QPM_PRIMO_IDX:QPM_PRIMO_IDX+1]
        tempo_primo = x[:, :, TEMPO_PRIMO_IDX:]

        beat_qpm_primo = qpm_primo[:,0:1].repeat(1, max_num_measures, 1)
        beat_tempo_primo = tempo_primo[:,0:1].repeat(1,max_num_measures, 1)
        beat_tempo_vector = note_feature_to_beat_mean(x[:,:,TEMPO_IDX:TEMPO_IDX+5], measure_numbers, use_mean=False)
        return torch.cat([beat_qpm_primo, beat_tempo_primo, beat_tempo_vector], dim=-1)

        # num_measures = measure_numbers[-1] - measure_numbers[0] + 1

        # qpm_primo = x[:, :, QPM_PRIMO_IDX].view(1, -1, 1)
        # tempo_primo = x[:, :, TEMPO_PRIMO_IDX:].view(1, -1, 2)
        # # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
        # beat_qpm_primo = qpm_primo[0, 0, 0].repeat((1, num_measures, 1))
        # beat_tempo_primo = tempo_primo[0, 0, :].repeat((1, num_measures, 1))
        # beat_tempo_vector = note_feature_to_beat_mean(x[:,:,TEMPO_IDX:TEMPO_IDX+5], measure_numbers, use_mean=False)

        # return torch.cat((beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), dim=-1)