import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .model_utils import make_higher_node, reparameterize, masking_half, encode_with_net, cal_length_from_padded_beat_numbers
from .module import GatedGraph, SimpleAttention, ContextAttention, GatedGraphX, GatedGraphXBias, GraphConvStack



class HanPerfEncoder(nn.Module):
    def __init__(self, net_params) -> None:
        super(HanPerfEncoder, self).__init__()
        self.performance_embedding_size = net_params.performance.size
        self.encoder_size = net_params.encoder.size
        self.encoded_vector_size = net_params.encoded_vector_size
        self.encoder_input_size = net_params.encoder.input
        self.encoder_layer_num = net_params.encoder.layer
        self.num_attention_head = net_params.num_attention_head

        self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True)
        if self.encoder_size % self.num_attention_head == 0:
            self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        else:
            self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.encoder_size * 2)
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
        self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)


    def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        total_note_cat = score_embedding['total_note_cat']

        expanded_y = self.performance_embedding_layer(y)
        # perform_concat = torch.cat((note_out, beat_out_spanned, measure_out_spanned, expanded_y), 2)
        perform_concat = torch.cat((total_note_cat, expanded_y), 2)
        perform_concat = masking_half(perform_concat)
        perform_contracted = self.performance_contractor(perform_concat)
        perform_note_encoded, _ = self.performance_note_encoder(perform_contracted)

        perform_measure = make_higher_node(perform_note_encoded, self.performance_measure_attention,
                                                beat_numbers, measure_numbers, lower_is_note=True)
        perform_measure = pack_padded_sequence(perform_measure, perform_measure.shape[1] - (perform_measure.sum(-1)==0).sum(dim=1), True, False )

        perform_style_encoded, _ = self.performance_encoder(perform_measure)
        perform_style_encoded, _ = pad_packed_sequence(perform_style_encoded, True)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)
        perform_z, perform_mu, perform_var = \
            encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        if return_z:
            return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_z, perform_mu, perform_var


class IsgnPerfEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnPerfEncoder, self).__init__()
        self.performance_contractor = nn.Sequential(
            nn.Linear(net_params.encoder.input, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.performance_embedding_layer = nn.Sequential(
            nn.Linear(net_params.output_size, net_params.performance.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.performance_graph_encoder = GatedGraph(net_params.encoder.size, net_params.num_edge_types)
        self.performance_measure_attention = ContextAttention(net_params.encoder.size, net_params.num_attention_head)

        self.performance_encoder = nn.LSTM(net_params.encoder.size, net_params.encoder.size, num_layers=net_params.encoder.layer,
                                           batch_first=True, bidirectional=True)

        self.performance_final_attention = ContextAttention(net_params.encoder.size * 2, net_params.num_attention_head)
        self.performance_encoder_mean = nn.Linear(net_params.encoder.size * 2, net_params.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(net_params.encoder.size * 2, net_params.encoded_vector_size)

    def get_perform_style_from_input(self, perform_concat, edges, measure_numbers):
        perform_style_contracted = self.performance_contractor(perform_concat)
        perform_style_graphed = self.performance_graph_encoder(perform_style_contracted, edges)
        performance_measure_nodes = make_higher_node(perform_style_graphed, self.performance_measure_attention, measure_numbers,
                                                measure_numbers, lower_is_note=True)
        perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)
        perform_z, perform_mu, perform_var = \
            encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        return perform_z, perform_mu, perform_var

    def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
        measure_numbers = note_locations['measure']
        note_out = score_embedding['total_note_cat']

        expanded_y = self.performance_embedding_layer(y)

        perform_concat = torch.cat((note_out.repeat(y.shape[0], 1, 1), expanded_y), 2)
        perform_z, perform_mu, perform_var = self.get_perform_style_from_input(perform_concat, edges, measure_numbers)
        if return_z:
            return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_z, perform_mu, perform_var


class IsgnPerfEncoderX(IsgnPerfEncoder):
    def __init__(self, net_params):
        super(IsgnPerfEncoderX, self).__init__(net_params)
        self.performance_graph_encoder = GatedGraphX(net_params.encoder.size, net_params.encoder.size, net_params.num_edge_types)
        self.performance_measure_attention = ContextAttention(net_params.encoder.size, net_params.num_attention_head)

        self.performance_encoder = nn.LSTM(net_params.encoder.size, net_params.encoder.size, num_layers=net_params.encoder.layer,
                                           batch_first=True, bidirectional=True)

        self.performance_final_attention = ContextAttention(net_params.encoder.size * 2, net_params.num_attention_head)
        self.performance_encoder_mean = nn.Linear(net_params.encoder.size * 2, net_params.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(net_params.encoder.size * 2, net_params.encoded_vector_size)

    def get_perform_style_from_input(self, perform_concat, edges, measure_numbers):
        perform_style_contracted = self.performance_contractor(perform_concat)
        hidden = torch.zeros_like(perform_style_contracted)
        perform_style_graphed = self.performance_graph_encoder(perform_style_contracted, hidden, edges)
        performance_measure_nodes = make_higher_node(perform_style_graphed, self.performance_measure_attention, measure_numbers,
                                                measure_numbers, lower_is_note=True)
        perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)
        perform_z, perform_mu, perform_var = \
            encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        return perform_z, perform_mu, perform_var


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
    total_perform_z = []
    for i in range(num):
        temp_z = reparameterize(perform_mu, perform_var)
        total_perform_z.append(temp_z)
    return total_perform_z