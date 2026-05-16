import torch
import random
import math

# from .pyScoreParser import xml_matching
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from collections import Counter, OrderedDict
from .utils import load_dat
from .data_process import make_slicing_indexes_by_measure, make_slice_with_same_measure_number, key_augmentation
from . import graph

class ScorePerformDataset:
  def __init__(self, path, type, len_slice, len_graph_slice, graph_keys, hier_type=[]):
    # type = one of ['train', 'valid', 'test', 'entire']
    path = Path(path)
    self.type = type
    if type == 'entire':
      self.path = path
    else:
      self.path = path / type
    self.stats = load_dat(path/"stat.pkl")
    self.key_augmentor = KeyAugmentor(self.stats)

    self.data_paths = self.get_data_path()
    self.data = self.load_data()
    self.len_slice = len_slice
    self.len_graph_slice = len_graph_slice
    self.graph_margin = 100
    if graph_keys and len(graph_keys)>0:
      self.is_graph = True
      self.graph_keys = graph_keys
      self.stats['graph_keys'] = graph_keys
    else:
      self.is_graph = False
      self.stats['graph_keys'] = []
    hier_keys = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas', 'meas_note']
    for key in hier_keys:
      if key in hier_type:
        setattr(self, key, True)
      else:
        setattr(self, key, False)

    self.midi_pitch_idx = self.stats['key_to_dim']['input']['midi_pitch_unnorm'][0]
    self.update_slice_info()

  def update_slice_info(self):
    self.slice_info = []
    for i, data in enumerate(self.data):
      data_size = len(data['input'])
      measure_numbers = data['note_location']['measure']
      if self.is_hier and self.hier_meas:
        slice_indices = make_slice_with_same_measure_number(data_size, measure_numbers, measure_steps=self.len_slice)
      else:
        slice_indices = make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.len_slice)
      for idx in slice_indices:
        self.slice_info.append((i, idx))
  
  def __getitem__(self, index):
    idx, sl_idx = self.slice_info[index]
    data = self.data[idx]
    return self.data_to_formatted_tensor(data, sl_idx)

  def data_to_formatted_tensor(self, data, sl_idx):
    batch_start, batch_end = sl_idx
    batch_x = torch.Tensor(data['input'][batch_start:batch_end])
    if self.type == 'train':
      max_up = min(108 - torch.max(batch_x[:, self.midi_pitch_idx]), 7)
      max_down = min(torch.min(batch_x[:, self.midi_pitch_idx]) - 21, 5)
      aug_key = random.randrange(-max_down, max_up)
      batch_x = self.key_augmentor(batch_x, aug_key)
      # batch_x = torch.Tensor(key_augmentation(data['input'][batch_start:batch_end], aug_key, self.stats['stats']["midi_pitch"]["stds"]))
    if self.in_hier:
      if self.hier_meas:
        batch_x = torch.cat((batch_x, torch.Tensor(data['meas'][batch_start:batch_end])), dim=-1)
    if self.is_hier:
      if self.hier_meas:
        batch_y = torch.Tensor(data['meas'][batch_start:batch_end])
    else:
      batch_y = torch.Tensor(data['output'][batch_start:batch_end])
    note_locations = {
        'beat': torch.Tensor(data['note_location']['beat'][batch_start:batch_end]).type(torch.int32),
        'measure': torch.Tensor(data['note_location']['measure'][batch_start:batch_end]).type(torch.int32),
        'section': torch.Tensor(data['note_location']['section'][batch_start:batch_end]).type(torch.int32),
        'voice': torch.Tensor(data['note_location']['voice'][batch_start:batch_end]).type(torch.int32),
    }

    align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end])
    articulation_loss_weight = torch.Tensor(data['articulation_loss_weight'][batch_start:batch_end])
    if self.is_graph:
      graphs = graph.edges_to_matrix_short(data['graph'], sl_idx, self.graph_keys)
      if self.len_graph_slice != self.len_slice:
        graphs = split_graph_to_batch(graphs, self.len_graph_slice, self.graph_margin)
    else:
      graphs = None

    meas_y = torch.Tensor(data['meas'][batch_start:batch_end])
    beat_y = torch.Tensor(data['beat'][batch_start:batch_end])
    return [batch_x, batch_y, beat_y, meas_y, note_locations, align_matched, articulation_loss_weight, graphs]
    # else:
    #     return [batch_x, batch_y, note_locations, align_matched, articulation_loss_weight, graphs]

  def get_data_path(self):
    return [x for x in self.path.rglob("*.pkl") if x.name != 'stat.pkl']
  
  def load_data(self):
    return [load_dat(x) for x in self.data_paths]

  def __len__(self):
    return len(self.slice_info)


class KeyAugmentor:
  def __init__(self, stat_dict):
    self.stat = stat_dict
  
  def _augment_continuous_MIDI_pitch(self, input_tensor, key_shift):
    '''
    input_tensor (torch.Tensor)
    '''
    output = torch.clone(input_tensor)
    pitch_std = self.stat['stats']["midi_pitch"]["stds"]
    midi_pitch_idx = self.stat['key_to_dim']['input']["midi_pitch"][0]
    output[:, midi_pitch_idx] = input_tensor[:, midi_pitch_idx] + key_shift/pitch_std
    return output

  def _augment_pitch_vec(self, input_tensor, key_shift):
    '''
    input_tensor (torch.Tensor)
    pitch_vec: 12-dim one-hot vector of pitch class + octave
    '''

    output = torch.clone(input_tensor)

    vec_start_id, vec_end_id  = self.stat['key_to_dim']['input']['pitch']
    original_pitch = input_tensor[:, vec_start_id:vec_end_id]
    shifted_pitch = torch.zeros_like(original_pitch)
    if key_shift > 0:
      shifted_pitch[:, 1+key_shift:] = original_pitch[:, 1:-key_shift]
      shifted_pitch[:, 1:1+key_shift] = original_pitch[:, -key_shift:]
      shifted_pitch[torch.sum(original_pitch[:,-key_shift:], dim=1)==1, 0] += 0.25
    else:
      shifted_pitch[:, 1:key_shift] = original_pitch[:,1-key_shift:]
      shifted_pitch[:, key_shift:] = original_pitch[:,1:1-key_shift]
      shifted_pitch[torch.sum(original_pitch[:,1:1-key_shift], dim=1)==1, 0] += 0.25
    
    output[:, vec_start_id:vec_end_id] = shifted_pitch
    return output

  def _augment_pitch_categorical_value(self, input_tensor, key_shift):
    output = torch.clone(input_tensor)
    output[:, self.stat['key_to_dim']['input']['midi_pitch_unnorm'][0]] += key_shift
    return output

  def __call__(self, input_tensor, key_shift):
    if key_shift == 0:
      return input_tensor
    output = self._augment_continuous_MIDI_pitch(input_tensor, key_shift)
    output = self._augment_pitch_categorical_value(output, key_shift)
    output = self._augment_pitch_vec(output, key_shift)
    return output




class EmotionDataset(ScorePerformDataset):
  def __init__(self, path, type, len_slice, len_graph_slice, graph_keys, hier_type=[]):
    super(EmotionDataset, self).__init__(path, type, len_slice, len_graph_slice, graph_keys, hier_type)
    
    self.cross_valid_split = self.make_cross_validation_split()

  def get_data_path(self):
    entire_list = list(self.path.rglob("*.pkl"))
    entire_list.sort()
    entire_list = [x for x in entire_list if 'mm_1-' in x.stem]
    return entire_list

  def update_slice_info(self):
    self.slice_info = []
    for i, data in enumerate(self.data):
        data_size = len(data['input'])
        self.slice_info.append((i, (0,data_size)))
  
  def make_cross_validation_split(self):
    samples = [x.stem.split('.')[:-1] for x in self.data_paths]
    samples = [{'composer': x[0], 'piece': x[1], 'slice':x[2], 'player': x[3], 'emotion':x[4]} for x in samples]
    piece_names = [x['composer'] + '_' + x['piece'] for x in samples]
    unique_piece_names = list(OrderedDict.fromkeys(piece_names))
    random.seed(0)
    random.shuffle(unique_piece_names)
    valid_slices = []
    slice_indices = list(range(0, len(unique_piece_names), len(unique_piece_names)//5+1)) + [len(unique_piece_names)]
    for i in range(1,len(slice_indices)):
        selected_pieces = unique_piece_names[slice_indices[i-1]:slice_indices[i]]
        # selected_ids = [j for j, x in enumerate(piece_names)  if x in selected_pieces]
        selected_ids = []
        for j,x in enumerate(piece_names):
            if x in selected_pieces:
                selected_ids.append(j)
        valid_slices.append(selected_ids)
    return valid_slices

class MultiplePerformSet(ScorePerformDataset):
    def __init__(self, path, type, len_slice, len_graph_slice, graph_keys, hier_type=[], min_perf=5):
        self.min_perf = min_perf
        super(MultiplePerformSet, self).__init__(path, type, len_slice, len_graph_slice, graph_keys, hier_type)

    def get_data_path(self):
        data_lists = list(self.path.glob("*.pkl"))
        return filter_performs_by_num_perf_by_piece(data_lists, min_perf=self.min_perf)

    def load_data(self):
        return [[load_dat(x) for x in piece] for piece in self.data_paths] 

    def update_slice_info(self):
        self.slice_info = []
        for i, piece in enumerate(self.data):
            data = piece[0]
            data_size = len(data['input'])
            measure_numbers = data['note_location']['measure']
            if self.is_hier and self.hier_meas:
                slice_indices = make_slice_with_same_measure_number(data_size, measure_numbers, measure_steps=self.len_slice)
            else:
                slice_indices = make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.len_slice)
            for idx in slice_indices:
                self.slice_info.append((i, idx))

    def __getitem__(self, index):
        idx, sl_idx = self.slice_info[index]
        piece = self.data[idx]
        selected_piece = random.sample(piece, self.min_perf)
        batch_start, batch_end = sl_idx
        aug_key = random.randrange(-5, 7)
        total_batch_x = []
        total_batch_y = []
        for data in selected_piece:
            batch_x = torch.Tensor(key_augmentation(data['input'][batch_start:batch_end], aug_key, self.stats['stats']["midi_pitch"]["stds"]))
            if self.in_hier:
                if self.hier_meas:
                    batch_x = torch.cat((batch_x, torch.Tensor(data['meas'][batch_start:batch_end])), dim=-1)
            if self.is_hier:
                if self.hier_meas:
                    batch_y = torch.Tensor(data['meas'][batch_start:batch_end])
            else:
                batch_y = torch.Tensor(data['output'][batch_start:batch_end])
            total_batch_x.append(batch_x)
            total_batch_y.append(batch_y)
        data = selected_piece[0]
        note_locations = {
            'beat': torch.Tensor(data['note_location']['beat'][batch_start:batch_end]).type(torch.int32),
            'measure': torch.Tensor(data['note_location']['measure'][batch_start:batch_end]).type(torch.int32),
            'section': torch.Tensor(data['note_location']['section'][batch_start:batch_end]).type(torch.int32),
            'voice': torch.Tensor(data['note_location']['voice'][batch_start:batch_end]).type(torch.int32),
        }

        if self.is_graph:
            graphs = graph.edges_to_matrix_short(data['graph'], sl_idx, self.graph_keys)
            if self.len_graph_slice != self.len_slice:
                graphs = split_graph_to_batch(graphs, self.len_graph_slice, self.graph_margin)
                
            return [torch.mean(torch.stack(total_batch_x),dim=0, keepdim=True), torch.stack(total_batch_y), note_locations, graphs]
        else:
            graphs = None
            return [torch.stack(total_batch_x), torch.stack(total_batch_y), note_locations, graphs]

    
def multi_collate(batch):
    return batch[0]


def filter_performs_by_num_perf_by_piece(perform_dat_paths, min_perf=5):
    '''
    Input: List of PosixPath for performance data
    output: List of PosixPath that has multiple performances per piece  (more than min_perf)
    '''
    piece_name = ['_'.join(x.stem.split('_')[:-1]) for x in perform_dat_paths]
    perf_counter = Counter(piece_name)
    filtered_piece = [piece for piece in set(piece_name) if perf_counter[piece] >= min_perf]
    return [[perf for perf in perform_dat_paths if '_'.join(perf.stem.split('_')[:-1])==piece] for piece in filtered_piece]


def split_graph_to_batch(graphs, len_slice, len_margin):
  '''
  graphs (torch.Tensor): Adjacency matrix (Edge x T x T)
  len_slice (int): Number of Notes per sliced adjacency matrix 

  graph_split (torch.Tensor): Adjacency matrix in (N x E x T x T)
  '''
  if graphs.shape[1] < len_slice:
    return graphs.unsqueeze(0)
  num_types = graphs.shape[0]
  num_slice = 1 + math.ceil( (graphs.shape[1] - len_slice) / (len_slice - len_margin*2) )
  hop_size = len_slice - len_margin * 2

  graph_split = torch.zeros((num_slice, num_types, len_slice, len_slice)).to(graphs.device)
  for i in range(num_slice-1):
    graph_split[i] = graphs[:, hop_size*i:hop_size*i+len_slice, hop_size*i:hop_size*i+len_slice]
  graph_split[-1] = graphs[:,-len_slice:, -len_slice:]
  return graph_split

class FeatureCollate:
  def __call__(self, batch):
    # batch_x = pad_sequence([sample[0] for sample in batch], batch_first=True)
    # batch_y = pad_sequence([sample[1] for sample in batch], batch_first=True)
    # if len(batch[0]) == 6:
    #   note_locations = {'beat': pad_sequence([sample[2]['beat'] for sample in batch], True).long(),
    #                     'measure': pad_sequence([sample[2]['measure'] for sample in batch], True).long(),
    #                     'section': pad_sequence([sample[2]['section'] for sample in batch], True).long(),
    #                     'voice': pad_sequence([sample[2]['voice'] for sample in batch], True).long()
    #                     }
    #   align_matched = pad_sequence([sample[3] for sample in batch], batch_first=True)
    #   pedal_status = pad_sequence([sample[4] for sample in batch], batch_first=True)
    #   if batch[0][5] is not None:
    #     edges = pad_sequence([sample[5] for sample in batch], batch_first=True) # TODO:
    #   else:
    #     edges = None
    #   return (batch_x,
    #           batch_y,
    #           note_locations, 
    #           align_matched.unsqueeze(-1), 
    #           pedal_status.unsqueeze(-1), 
    #           edges
    #         ) 
    # else:
      batch_x = pack_sequence([sample[0] for sample in batch], enforce_sorted=False)
      batch_y = pad_sequence([sample[1] for sample in batch], batch_first=True)
      beat_y = pad_sequence([sample[2] for sample in batch], batch_first=True)
      meas_y = pad_sequence([sample[3] for sample in batch], batch_first=True)
      note_locations = {'beat': pad_sequence([sample[4]['beat'] for sample in batch], True).long(),
                        'measure': pad_sequence([sample[4]['measure'] for sample in batch], True).long(),
                        'section': pad_sequence([sample[4]['section'] for sample in batch], True).long(),
                        'voice': pad_sequence([sample[4]['voice'] for sample in batch], True).long()
                        }
      align_matched = pad_sequence([sample[5] for sample in batch], batch_first=True)
      pedal_status = pad_sequence([sample[6] for sample in batch], batch_first=True)
      if batch[0][7] is not None:
        edges = pad_sequence([sample[7] for sample in batch], batch_first=True) # TODO:
      else:
        edges = None
      return (batch_x,
              batch_y,
              beat_y, 
              meas_y,
              note_locations, 
              align_matched.unsqueeze(-1), 
              pedal_status.unsqueeze(-1), 
              edges
            ) 
