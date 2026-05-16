import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class NoteEmbedder(nn.Module):
  def __init__(self, net_param, stats):
    super().__init__()
    self.output_size = net_param.note.size
    self.key_to_dim = stats['key_to_dim']['input']

    self.ignored_indices = []
    if hasattr(net_param, 'use_continuos_feature_only') and net_param.use_continuos_feature_only:
      for key in self.key_to_dim:
        if 'unnorm' in key:
          idx_range = self.key_to_dim[key]
          self.ignored_indices += list(range(idx_range[0], idx_range[1]))
    self.input_size = net_param.input_size - len(self.ignored_indices)

  def _range_to_ids(self, range_in_tuple):
    return list(range(range_in_tuple[0], range_in_tuple[1]))

  def _update_idx_for_key(self, key_lists):
    indices =  [self._range_to_ids(self.key_to_dim[key]) for key in key_lists]
    return [y for x in indices for y in x]

  def _embed_note(self, x):
    raise NotImplementedError

  def forward(self, x):
    if isinstance(x, PackedSequence):
      x_data = x.data
    else:
      x_data = x

    emb = self._embed_note(x_data)

    if isinstance(x, PackedSequence):
      x = PackedSequence(emb, x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    else:
      x = emb
    return x

class MixEmbedder(NoteEmbedder):
  def __init__(self, net_param, stats):
    super().__init__(net_param, stats)
    self.fc = nn.Linear(self.input_size, self.output_size)
    self.valid_indices = [i for i in range(net_param.input_size) if i not in self.ignored_indices]

  def _embed_note(self, x):
    return self.fc(x[..., self.valid_indices])

class CatEmbedder(NoteEmbedder):
  def __init__(self, net_param, stats):
    super().__init__(net_param, stats)
    self.pitch_keys = ['midi_pitch', 'pitch']
    self.duration_keys = ['duration', 'beat_importance', 'measure_length', 'qpm_primo', 'section_tempo', 'following_rest', 
                          'distance_from_recent_tempo', 'beat_position', 'grace_order', 'preceded_by_grace_note', 'followed_by_fermata_rest',
                          'tempo', 'time_sig_vec', 'slur_beam_vec',  'notation', 'tempo_primo']
    self.etc_keys = ['distance_from_abs_dynamic', 'dynamic', 'xml_position', 'composer_vec']

    self.pitch_ids = self._update_idx_for_key(self.pitch_keys)
    self.duration_ids = self._update_idx_for_key(self.duration_keys)
    self.etc_ids = self._update_idx_for_key(self.etc_keys)

    self.pitch_fc = nn.Linear(len(self.pitch_ids), net_param.note.size//8 * 3)
    self.duration_fc = nn.Linear(len(self.duration_ids), net_param.note.size//8 * 3)
    self.etc_fc = nn.Linear(len(self.etc_ids), net_param.note.size//8 * 2)

  def _embed_note(self, x):
    pitch_emb = self.pitch_fc(x[..., self.pitch_ids])
    dur_emb = self.duration_fc(x[..., self.duration_ids])
    etc_emb = self.etc_fc(x[..., self.etc_ids])
    cat = torch.cat([pitch_emb, dur_emb, etc_emb], dim=-1)
    return cat


class CategoryEmbedder(NoteEmbedder):
  def __init__(self, net_param, stats):
    super().__init__(net_param, stats)
    self.pitch_keys = ['midi_pitch_unnorm']
    self.duration_keys = ['duration', 'beat_importance', 'measure_length', 'qpm_primo', 'section_tempo', 'following_rest', 
                            'distance_from_recent_tempo', 'beat_position', 'grace_order', 'preceded_by_grace_note', 'followed_by_fermata_rest',
                            'tempo', 'time_sig_vec', 'slur_beam_vec',  'notation', 'tempo_primo']
    self.etc_keys = ['distance_from_abs_dynamic', 'dynamic', 'xml_position', 'composer_vec']

    self.pitch_ids = self._update_idx_for_key(self.pitch_keys)
    self.duration_ids = self._update_idx_for_key(self.duration_keys)
    self.etc_ids = self._update_idx_for_key(self.etc_keys)

    self.pitch_embedder = nn.Embedding(88, net_param.note.size//8 * 3)
    self.duration_fc = nn.Linear(len(self.duration_ids), net_param.note.size//8 * 3)
    self.etc_fc = nn.Linear(len(self.etc_ids), net_param.note.size//8 * 2)

  def _embed_note(self, x):
    pitch = x[..., self.pitch_ids[0]].long() - 21
    pitch_emb = self.pitch_embedder(pitch)
    dur_emb = self.duration_fc(x[..., self.duration_ids])
    etc_emb = self.etc_fc(x[..., self.etc_ids])
    cat = torch.cat([pitch_emb, dur_emb, etc_emb], dim=-1)
    return cat