import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .model_utils import make_higher_node, reparameterize, masking_half, encode_with_net, run_hierarchy_lstm_with_pack
from .module import GatedGraph, SimpleAttention, ContextAttention, GatedGraphX, GatedGraphXBias, GraphConvStack

class PerformanceEncoder(nn.Module):
  def __init__(self, net_params):
    super().__init__()
    self.performance_embedding_size = net_params.performance.size
    self.encoder_size = net_params.encoder.size
    self.encoded_vector_size = net_params.encoded_vector_size
    self.encoder_input_size = net_params.encoder.input
    self.encoder_layer_num = net_params.encoder.layer
    self.num_attention_head = net_params.num_attention_head

    self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)

    self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
    self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
    self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
    self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)


  def _get_note_hidden_states(self, perform_style_contracted, edges):
    raise NotImplementedError

  def _get_perform_style_from_input(self, perform_concat, edges, measure_numbers):
    perform_style_contracted = self.performance_contractor(perform_concat)
    perform_style_contracted[(perform_concat==0).all(dim=-1)] = 0
    perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
    performance_measure_nodes = make_higher_node(perform_style_note_hidden, self.performance_measure_attention, measure_numbers,
                                            measure_numbers, lower_is_note=True)
    perform_style_encoded = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder)
    # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
    perform_style_vector = self.performance_final_attention(perform_style_encoded)
    perform_z, perform_mu, perform_var = \
        encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
    return perform_z, perform_mu, perform_var

  def _expand_perf_feature(self, y):
    '''
    Simply expand performance features to larger dimension

    y (torch.Tensor): performance features (N x T x C)
    '''
    is_padded = (y==0).all(dim=-1)

    expanded_y = self.performance_embedding_layer(y)

    mask = torch.ones_like(expanded_y)
    mask[is_padded] = 0
    expanded_y *= mask
    # expanded_y[is_padded] = 0
    return expanded_y

  def _masking_notes(self, perform_concat):
    '''
    perform_concat (torch.Tensor): N x T x C
    out (torch.Tensor): N x T//2 x C
    '''
    return masking_half(perform_concat)

  def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
    measure_numbers = note_locations['measure']
    total_note_cat = score_embedding['total_note_cat']

    expanded_y = self._expand_perf_feature(y)
    perform_concat = torch.cat((total_note_cat, expanded_y), 2)
    perform_concat = self._masking_notes(perform_concat)
    
    perform_z, perform_mu, perform_var = self._get_perform_style_from_input(perform_concat, edges, measure_numbers)
    if return_z:
        return sample_multiple_z(perform_mu, perform_var, num_samples)
    return perform_z, perform_mu, perform_var


class HanPerfEncoder(PerformanceEncoder):
    def __init__(self, net_params) -> None:
      super(HanPerfEncoder, self).__init__(net_params)
      self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True, batch_first=True)
      self.performance_embedding_layer = nn.Sequential(
          nn.Linear(net_params.output_size, self.performance_embedding_size),
          nn.Dropout(net_params.drop_out),
          nn.ReLU(),
      )
      self.performance_contractor = nn.Sequential(
          nn.Linear(self.encoder_input_size, self.encoder_size),
          nn.Dropout(net_params.drop_out),
          # nn.BatchNorm1d(self.encoder_size),
          nn.ReLU()
      )

    def _get_note_hidden_states(self, perform_style_contracted, edges):
      perform_note_encoded = run_hierarchy_lstm_with_pack(perform_style_contracted, self.performance_note_encoder)
      # perform_note_encoded, _ = self.performance_note_encoder(perform_style_contracted)
      return perform_note_encoded

class NonMaskingHanPerfEncoder(HanPerfEncoder):
  def __init__(self, net_params) -> None:
    super(NonMaskingHanPerfEncoder, self).__init__(net_params)

  def _masking_notes(self, perform_concat):
    '''
    This Encoder does not mask notes
    '''
    return perform_concat 


class IsgnPerfEncoder(PerformanceEncoder):
  def __init__(self, net_params):
    super(IsgnPerfEncoder, self).__init__(net_params)
    self.performance_contractor = nn.Sequential(
        nn.Linear(net_params.encoder.input, net_params.encoder.size * 2),
        nn.Dropout(net_params.drop_out),
        nn.ReLU(),
    )
    self.performance_embedding_layer = nn.Sequential(
        nn.Linear(net_params.output_size, net_params.performance.size),
        # nn.Dropout(net_params.drop_out),
        # nn.ReLU(),
    )
    self.performance_graph_encoder = GatedGraph(net_params.encoder.size * 2, net_params.num_edge_types)

  def _get_note_hidden_states(self, perform_style_contracted, edges):
    return self.performance_graph_encoder(perform_style_contracted, edges)

  def _masking_notes(self, perform_concat):
    return perform_concat # TODO: Implement it with sliced graph

class IsgnPerfEncoderX(IsgnPerfEncoder):
  def __init__(self, net_params):
      super(IsgnPerfEncoderX, self).__init__(net_params)
      self.performance_contractor = nn.Sequential(
        nn.Linear(net_params.encoder.input, net_params.encoder.size),
        nn.Dropout(net_params.drop_out),
        nn.ReLU(),
      )
      self.performance_graph_encoder = GatedGraphX(net_params.encoder.size, net_params.encoder.size * 2, net_params.num_edge_types)

  def _get_note_hidden_states(self, perf_sty, edges):
    zero_hidden = torch.zeros(perf_sty.shape[0], perf_sty.shape[1], perf_sty.shape[2]*2).to(perf_sty).device
    return self.performance_graph_encoder(perf_sty, zero_hidden, edges)


class IsgnPerfEncoderXBias(IsgnPerfEncoderX):
  def __init__(self, net_params):
    super(IsgnPerfEncoderXBias, self).__init__(net_params)
    self.performance_graph_encoder = GatedGraphXBias(net_params.encoder.size, net_params.encoder.size, net_params.num_edge_types)



class IsgnPerfEncoderMasking(IsgnPerfEncoder):
    def __init__(self, net_params):
        super(IsgnPerfEncoderMasking, self).__init__(net_params)

    def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
        measure_numbers = note_locations['measure']
        # note_out, _, = score_embedding
        note_out = score_embedding['total_note_cat']

        expanded_y = self.performance_embedding_layer(y)
        perform_concat = torch.cat((note_out.repeat(y.shape[0], 1, 1), expanded_y), 2)

        if self.training():
          perform_concat = masking_half(perform_concat)

        perform_z, perform_mu, perform_var = self.get_perform_style_from_input(perform_concat, edges, measure_numbers)
        if return_z:
            return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_z, perform_mu, perform_var


class GcnPerfEncoderMasking(IsgnPerfEncoder):
    def __init__(self, net_params):
        super().__init__(net_params)
        self.performance_graph_encoder = GraphConvStack(net_params.encoder.size, 
                                                        net_params.encoder.size, 
                                                        net_params.num_edge_types, 
                                                        num_layers=net_params.encoder.layer,
                                                        drop_out=net_params.drop_out)

    
def sample_multiple_z(perform_mu, perform_var, num=10):
    assert perform_mu.dim() == 2
    total_perform_z = []
    for i in range(num):
      temp_z = reparameterize(perform_mu, perform_var)
      total_perform_z.append(temp_z)
    return torch.stack(total_perform_z, dim=1)