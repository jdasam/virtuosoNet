import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

class NoteEmbedder(nn.Module):
  def __init__(self, net_param, stats):
    super().__init__()
    self.output_size = net_param.note.size
    self.key_to_dim = stats['key_to_dim']['input']

    self.ignored_indices = []
    if net_param.use_continuos_feature_only:
      for key in self.key_to_dim:
        if 'unnorm' in key:
          idx_range = self.key_to_dim[key]
          self.ignored_indices += list(range(idx_range[0], idx_range[1]))
    # print(f'These indices will be ignored: {self.ignored_indices}')
    self.input_size = net_param.input_size - len(self.ignored_indices)
    self.fc = nn.Linear(self.input_size, self.output_size)

    self.valid_indices = [i for i in range(net_param.input_size) if i not in self.ignored_indices]

  def forward(self, x):
    '''
    x (torch.Tensor): N x T x C
    '''
    if isinstance(x, PackedSequence):
      x_data = x.data[..., self.valid_indices]
      x_data = self.fc(x_data)
      x = PackedSequence(x_data, x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    else:
      x = x[..., self.valid_indices]
      x = self.fc(x)
    return x