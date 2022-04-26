import torch
import torch.nn as nn
from torch.autograd import Variable

from virtuoso import note_embedder

from . import model_constants as cons
from .model_utils import make_higher_node, reparameterize, span_beat_to_note_num
from . import model_utils as utils
from .module import GatedGraph
from .model_constants import QPM_INDEX, QPM_PRIMO_IDX, TEMPO_IDX, PITCH_IDX
from .pyScoreParser.xml_utils import xml_notes_to_midi
from .pyScoreParser.feature_to_performance import apply_tempo_perform_features

from . import encoder_score as encs
from . import encoder_perf as encp
from . import decoder as dec
from . import residual_selector as res
from . import note_embedder as nemb

# VOICE_IDX = 11
# PITCH_IDX = 13
# TEMPO_IDX = PITCH_IDX + 13
DYNAMICS_IDX = TEMPO_IDX + 5
LEN_DYNAMICS_VEC = 4
TEMPO_PRIMO_IDX = -2
NUM_VOICE_FEED_PARAM = 2

class VirtuosoNet(nn.Module):
    def __init__(self, net_param, data_stats):
        super(VirtuosoNet, self).__init__()
        self.note_embedder = getattr(nemb, net_param.note_embedder_name)(net_param, data_stats)
        self.score_encoder = getattr(encs, net_param.score_encoder_name)(net_param)
        self.performance_encoder = getattr(encp, net_param.performance_encoder_name)(net_param)
        self.residual_info_selector = getattr(res, net_param.residual_info_selector_name)(data_stats)
        self.performance_decoder = getattr(dec, net_param.performance_decoder_name)(net_param)
        self.network_params = net_param
        self.stats = data_stats
        self.stats['graph_keys'] = net_param.graph_keys

        # self.midi_converter = ModelToMIDIConverter(self.stats)

    def encode_style(self, x, y, edges, note_locations, num_samples=10):
        x_embedded = self.note_embedder(x)
        score_embedding = self.score_encoder(x_embedded, edges, note_locations)
        performance_embedding = self.performance_encoder(score_embedding, y, edges, note_locations, return_z=True, num_samples=num_samples)

        return performance_embedding

    def encode_style_distribution(self, x, y, edges, note_locations):
        x_embedded = self.note_embedder(x)
        score_embedding = self.score_encoder(x_embedded, edges, note_locations)
        _, perform_mu, perform_var = self.performance_encoder(score_embedding, y, edges, note_locations)

        return perform_mu, perform_var

    def sample_style_vector_from_normal_distribution(self, batch_size):
      zero_mean = torch.zeros(batch_size, self.performance_encoder.performance_encoder_mean.out_features)
      one_std = torch.zeros_like(zero_mean) # log std 0
      performance_embedding = reparameterize(zero_mean, one_std).to(next(self.parameters()).device)
      return performance_embedding

    def get_score_embedding(self, x, edges, note_locations):
      x_embedded = self.note_embedder(x)
      score_embedding = self.score_encoder(x_embedded, edges, note_locations)
      return score_embedding

    def forward(self, x, y, edges, note_locations, initial_z=None):
        x_embedded = self.note_embedder(x)
        score_embedding = self.score_encoder(x_embedded, edges, note_locations)
        if initial_z is None:
            performance_embedding, perform_mu, perform_var = self.performance_encoder(score_embedding, y, edges, note_locations, return_z=False)
        else: 
            if type(initial_z) is str and initial_z == 'zero':
                zero_mean = torch.zeros(x.shape[0], self.performance_encoder.performance_encoder_mean.out_features)
                one_std = torch.zeros_like(zero_mean) # log std 0
                performance_embedding = reparameterize(zero_mean, one_std).to(x.device)
            elif isinstance(initial_z, torch.Tensor) and not initial_z.is_cuda:
                performance_embedding = torch.Tensor(initial_z).to(x.device).view(x.shape[0],initial_z.shape[-1])
            else:
                performance_embedding = initial_z.view(x.shape[0],initial_z.shape[-1])
            perform_mu, perform_var = 0, 0
        residual_info = self.residual_info_selector(x, note_locations)
        output, alter_out = self.performance_decoder(score_embedding, performance_embedding, residual_info, edges, note_locations)
        return output, perform_mu, perform_var, alter_out
    
    
    # def _read_input_from_xml(self, xml_path, composer, qpm_primo):
    #   input_keys = self.stats['input_keys']
    #   graph_keys = self.stats['graph_keys']
    #   score, input_tensor, edges, note_locations = get_input_from_xml(xml_path, composer, qpm_primo, input_keys, graph_keys)

    # def generate_perform_midi_from_xml(self, x, edges, note_locations, initial_z='zero'):
    #   output, _, _, _ = self.forward(x, None, edges, note_locations, initial_z=initial_z)
    #   self.midi_converter(output)


class ModelToMIDIConverter:
  def __init__(self, stats):
    self.stats = stats

  def _scale_model_prediction_to_original(self, prediction):
    '''
    prediction (torch.Tensor): output of model forward, N x T x C

    out: Output features in original scale 
    '''
    prediction_unnorm = torch.clone(prediction.cpu())
    for key in self.stats['output_keys']:
      idx = list(range(*self.stats['key_to_dim']['output'][key]))
      prediction_unnorm[...,idx]  *= self.stats['stats'][key]['stds']
      prediction_unnorm[...,idx]  += self.stats['stats'][key]['mean']
    return prediction_unnorm

  def _model_prediction_to_feature(self, prediction_unnorm):
    '''
    prediction_unnorm (torch.Tensor): T x C 
    '''
    assert prediction_unnorm.dim() == 2
    output_features = {}
    for key in self.stats['output_keys']:
      idx = list(range(*self.stats['key_to_dim']['output'][key]))
      output_features[key] = prediction_unnorm[:,idx]
    return output_features
  
  def __call__(self, model_output, score):
    '''
    model_output (torch.Tensor): N x T x C
    score (pyScoreParser.ScoreData)
    '''
    unnorm_output = self._scale_model_prediction_to_original(model_output)
    output_features = self._model_prediction_to_feature(unnorm_output)
    xml_notes, tempos = apply_tempo_perform_features(score, output_features, start_time=0.5, predicted=True, return_tempo=True)
    output_midi, midi_pedals = xml_notes_to_midi(xml_notes, False, ignore_overlapped=True)
    save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_path,
                                  bool_pedal=bool_pedal, disklavier=disklavier, tempo_clock=clock_notes)

    return


class TrillRNN(nn.Module):
    def __init__(self, net_params, device):
        super(TrillRNN, self).__init__()
        self.hidden_size = net_params.note.size
        self.num_layers = net_params.note.layer
        self.input_size = net_params.input_size
        self.output_size = net_params.output_size
        self.device = device
        self.is_graph = False
        self.loss_type = 'MSE'

        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        # self.fc = nn.Linear(hidden_size * 2, num_output)  # 2 for

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.note_lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(self.hidden_size, self.output_size),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=0):
        note_contracted = self.note_fc(x)
        hidden_out, _ = self.note_lstm(note_contracted)

        out = self.out_fc(hidden_out)

        if self.loss_type == 'MSE':
            up_trill = self.sigmoid(out[:,:,-1])
            out[:,:,-1] = up_trill
        else:
            out = self.sigmoid(out)
        return out, False, False, torch.zeros(1)


class TrillGraph(nn.Module):
    def __init__(self, net_params, trill_index, loss_type, device):
        super(TrillGraph, self).__init__()
        self.loss_type = loss_type
        self.hidden_size = net_params.note.size
        self.num_layers = net_params.note.layer
        self.input_size = net_params.input_size
        self.output_size = net_params.output_size
        self.num_edge_types = net_params.num_edge_types
        self.is_trill_index = trill_index
        self.device = device

        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        # self.fc = nn.Linear(hidden_size * 2, num_output)  # 2 for

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.graph = GatedGraph(self.hidden_size, self.num_edge_types)

        self.out_fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(self.hidden_size, self.output_size),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edges):
        # hidden = self.init_hidden(x.size(0))
        # hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        if edges.shape[0] != self.num_edge_types:
            edges = edges[:self.num_edge_types, :, :]

        # Decode the hidden state of the last time step
        is_trill_mat = x[:, :, self.is_trill_index]
        is_trill_mat = is_trill_mat.view(1,-1,1).repeat(1,1,self.output_size).view(1,-1,self.output_size)
        is_trill_mat = Variable(is_trill_mat, requires_grad=False)

        note_contracted = self.note_fc(x)
        note_after_graph = self.graph(note_contracted, edges, iteration=5)
        out = self.out_fc(note_after_graph)

        if self.loss_type == 'MSE':
            up_trill = self.sigmoid(out[:,:,-1])
            out[:,:,-1] = up_trill
        else:
            out = self.sigmoid(out)
        # out = out * is_trill_mat
        return out


