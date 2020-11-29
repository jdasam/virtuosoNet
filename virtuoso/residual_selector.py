import torch


import torch
import torch.nn as nn
from .model_constants import TEMPO_PRIMO_IDX, QPM_PRIMO_IDX, TEMPO_IDX
from .utils import note_tempo_infos_to_beat


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
        beat_tempo_vector = note_tempo_infos_to_beat(x, beat_numbers, TEMPO_IDX)

        return torch.cat((beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), dim=-1)