import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
import random
import numpy
import math
import model_constants as cons


DROP_OUT = 0.1

QPM_INDEX = 0
# VOICE_IDX = 11
PITCH_IDX = 12
TEMPO_IDX = PITCH_IDX + 13
DYNAMICS_IDX = TEMPO_IDX + 5
LEN_DYNAMICS_VEC = 4
QPM_PRIMO_IDX = 4
TEMPO_PRIMO_IDX = -2
NUM_VOICE_FEED_PARAM = 2

class GatedGraph(nn.Module):
    def  __init__(self, size, num_edge_style, device=0, secondary_size=0):
        super(GatedGraph, self).__init__()
        if secondary_size == 0:
            secondary_size = size
        # for i in range(num_edge_style):
        #     subgraph = self.subGraph(size)
        #     self.sub.append(subgraph)
        self.size = size
        self.secondary_size = secondary_size

        self.ba = torch.nn.Parameter(torch.Tensor(size))
        self.wz = torch.nn.Parameter(torch.Tensor(num_edge_style,size,secondary_size))
        self.wr = torch.nn.Parameter(torch.Tensor(num_edge_style,size,secondary_size))
        self.wh = torch.nn.Parameter(torch.Tensor(num_edge_style,size,secondary_size))
        self.uz = torch.nn.Parameter(torch.Tensor(size, secondary_size))
        # self.bz = torch.nn.Parameter(torch.Tensor(secondary_size))
        self.ur = torch.nn.Parameter(torch.Tensor(size, secondary_size))
        # self.br = torch.nn.Parameter(torch.Tensor(secondary_size))
        self.uh = torch.nn.Parameter(torch.Tensor(secondary_size, secondary_size))
        # self.bh = torch.nn.Parameter(torch.Tensor(secondary_size))

        nn.init.xavier_normal_(self.wz)
        nn.init.xavier_normal_(self.wr)
        nn.init.xavier_normal_(self.wh)
        nn.init.xavier_normal_(self.uz)
        nn.init.xavier_normal_(self.ur)
        nn.init.xavier_normal_(self.uh)
        nn.init.zeros_(self.ba)
        # nn.init.zeros_(self.bz)
        # nn.init.zeros_(self.br)
        # nn.init.zeros_(self.bh)

        self.sigmoid = torch.nn.Sigmoid()

        self.tanh = torch.nn.Tanh()

    def forward(self, input, edge_matrix, iteration=10):
        for i in range(iteration):
            activation = torch.matmul(edge_matrix.transpose(1,2), input) + self.ba
            temp_z = self.sigmoid( torch.bmm(activation, self.wz).sum(0) + torch.matmul(input, self.uz))
            temp_r = self.sigmoid( torch.bmm(activation, self.wr).sum(0) + torch.matmul(input, self.ur))

            if self.secondary_size == self.size:
                temp_hidden = self.tanh(
                    torch.bmm(activation, self.wh).sum(0) + torch.matmul(temp_r * input, self.uh))
                input = (1 - temp_z) * input + temp_r * temp_hidden
            else:
                temp_hidden = self.tanh(
                    torch.bmm(activation, self.wh).sum(0) + torch.matmul(temp_r * input[:,:,-self.secondary_size:], self.uh) )
                temp_result = (1 - temp_z) * input[:,:,-self.secondary_size:] + temp_r * temp_hidden
                input = torch.cat((input[:,:,:-self.secondary_size], temp_result), 2)

        return input

#
# class ContextAttention(nn.Module):
#     def __init__(self, size, num_head):
#         super(ContextAttention, self).__init__()
#         self.attention_net = nn.Sequential(
#             nn.Linear(size, size),
#             nn.ReLU(),
#             nn.Linear(size,size)
#         )
#         self.num_head = num_head
#
#         # if size % num_head != 0:
#         #     raise ValueError("size must be dividable by num_head", size, num_head)
#         # self.head_size = int(size/num_head)
#         # self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
#         # nn.init.uniform_(self.context_vector, a=-1, b=1)
#
#
#     def forward(self, x):
#         attention = self.attention_net(x)
#         # attention_tanh = F.tanh(attention)
#         # similarity = torch.matmul(attention_tanh, self.context_vector)
#         # if self.head_size != 1:
#         #     attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
#         #     similarity = torch.bmm(attention_split, self.context_vector)
#         #     softmax_weight = F.softmax(similarity, dim=1)
#         #     x_split = torch.cat(x.split(split_size=self.head_size, dim=2), dim=0)
#         #
#         #     weighted_mul = torch.bmm(softmax_weight.transpose(1,2), x_split)
#         #
#         #     restore_size = int(weighted_mul.size(0) / self.num_head)
#         #     attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
#         # else:
#         # softmax_weight = F.softmax(attention_tanh, dim=1)
#         softmax_weight = F.softmax(attention, dim=1)
#         attention = softmax_weight * x
#
#         sum_attention = torch.sum(attention, dim=1)
#         return sum_attention

class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size/num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = F.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split, self.context_vector)
            softmax_weight = F.softmax(similarity, dim=1)
            x_split = torch.cat(x.split(split_size=self.head_size, dim=2), dim=0)

            weighted_mul = torch.bmm(softmax_weight.transpose(1,2), x_split)

            restore_size = int(weighted_mul.size(0) / self.num_head)
            attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
        else:
            softmax_weight = F.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention


class ISGN(nn.Module):
    def __init__(self, network_parameters, device):
        super(ISGN, self).__init__()
        self.device = device
        self.num_graph_iteration = network_parameters.graph_iteration
        self.num_sequence_iteration = network_parameters.sequence_iteration
        self.is_graph = True
        self.is_baseline = network_parameters.is_baseline
        if hasattr(network_parameters, 'is_test_version') and network_parameters.is_test_version:
            self.is_test_version = True
        else:
            self.is_test_version = False

        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.num_layers = network_parameters.note.layer
        self.note_hidden_size = network_parameters.note.size
        self.num_measure_layers = network_parameters.measure.layer
        self.measure_hidden_size = network_parameters.measure.size
        self.final_hidden_size = network_parameters.final.size
        self.final_input = network_parameters.final.input
        self.encoder_size = network_parameters.encoder.size
        self.encoded_vector_size = network_parameters.encoded_vector_size
        self.encoder_input_size = network_parameters.encoder.input
        self.encoder_layer_num = network_parameters.encoder.layer
        self.time_regressive_size = network_parameters.time_reg.size
        self.time_regressive_layer = network_parameters.time_reg.layer
        self.num_edge_types = network_parameters.num_edge_types
        self.final_graph_margin_size = network_parameters.final.margin

        if self.is_baseline:
            self.final_graph_input_size = self.final_input + self.encoder_size + 8 + self.output_size + self.final_graph_margin_size + self.time_regressive_size * 2
            self.final_beat_hidden_idx = self.final_input + self.encoder_size + 8 # tempo info
        else:
            self.final_graph_input_size = self.final_input + self.encoder_size + self.output_size + self.final_graph_margin_size + self.time_regressive_size * 2
            self.final_beat_hidden_idx = self.final_input + self.encoder_size

        self.num_attention_head = network_parameters.num_attention_head
        # self.num_attention_head = 4

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.note_hidden_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.note_hidden_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
        )

        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, self.device, secondary_size=self.note_hidden_size)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, self.device)

        self.measure_attention = ContextAttention(self.note_hidden_size * 2, self.num_attention_head)
        self.measure_rnn = nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)

        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU(),
            nn.Linear(self.encoder_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_graph_encoder = GatedGraph(self.encoder_size, self.num_edge_types, self.device)
        self.performance_measure_attention = ContextAttention(self.encoder_size, self.num_attention_head)

        self.performance_encoder = nn.LSTM(self.encoder_size, self.encoder_size, num_layers=self.encoder_layer_num,
                                           batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

        self.beat_tempo_contractor = nn.Sequential(
            nn.Linear(self.final_graph_input_size - self.time_regressive_size * 2, self.time_regressive_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )
        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )
        self.perform_style_to_measure = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size, self.encoder_size, num_layers=1, bidirectional=False)

        self.initial_result_fc = nn.Sequential(
            nn.Linear(self.final_input, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),

            nn.Linear(self.encoder_size, self.output_size),
            nn.ReLU()
        )

        self.final_graph = GatedGraph(self.final_graph_input_size, self.num_edge_types, self.device,
                                      self.output_size + self.final_graph_margin_size)
        if self.is_baseline:
            self.tempo_rnn = nn.LSTM(self.final_graph_margin_size + self.output_size, self.time_regressive_size,
                                     num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
            self.final_measure_attention = ContextAttention(self.output_size, 1)
            self.final_margin_attention = ContextAttention(self.final_graph_margin_size, self.num_attention_head)

            self.fc = nn.Sequential(
                nn.Linear(self.final_graph_input_size, self.final_graph_margin_size),
                nn.Dropout(DROP_OUT),
                nn.ReLU(),
                nn.Linear(self.final_graph_margin_size, self.output_size),
            )
        # elif self.is_test_version:
        else:
            self.tempo_rnn = nn.LSTM(self.final_graph_margin_size + self.output_size + 8, self.time_regressive_size,
                                     num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
            self.final_beat_attention = ContextAttention(self.output_size, 1)
            self.final_margin_attention = ContextAttention(self.final_graph_margin_size, self.num_attention_head)
            self.tempo_fc = nn.Linear(self.time_regressive_size * 2, 1)

            self.fc = nn.Sequential(
                nn.Linear(self.final_graph_input_size, self.final_graph_margin_size),
                nn.Dropout(DROP_OUT),
                nn.ReLU(),
                nn.Linear(self.final_graph_margin_size, self.output_size-1),
            )
        # else:
        #     self.tempo_rnn = nn.LSTM(self.time_regressive_size + 3 + 5, self.time_regressive_size, num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
        #     self.final_beat_attention = ContextAttention(self.final_graph_input_size - self.time_regressive_size * 2, 1)
        #     self.tempo_fc = nn.Linear(self.time_regressive_size * 2, 1)
        #     # self.fc = nn.Linear(self.final_input + self.encoder_size + self.output_size, self.output_size - 1)
        #     self.fc = nn.Sequential(
        #         nn.Linear(self.final_graph_input_size + 1, self.encoder_size),
        #         nn.Dropout(DROP_OUT),
        #         nn.ReLU(),
        #         nn.Linear(self.encoder_size, self.output_size - 1),
        #     )

        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=False, return_z=False):
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        section_numbers = [x.section for x in note_locations]
        num_notes = x.size(1)

        note_out, measure_hidden_out = self.run_graph_network(x, edges, measure_numbers, start_index)
        if type(initial_z) is not bool:
            if type(initial_z) is str and initial_z == 'zero':
                # zero_mean = torch.zeros(self.encoded_vector_size)
                # one_std = torch.ones(self.encoded_vector_size)
                # perform_z = self.reparameterize(zero_mean, one_std).to(self.device)
                perform_z = torch.Tensor(numpy.random.normal(size=self.encoded_vector_size)).to(self.device)
            # if type(initial_z) is list:
            #     perform_z = self.reparameterize(torch.Tensor(initial_z), torch.Tensor(initial_z)).to(self.device)
            elif not initial_z.is_cuda:
                perform_z = torch.Tensor(initial_z).to(self.device).view(1,-1)
            else:
                perform_z = initial_z.view(1,-1)
            perform_mu = 0
            perform_var = 0
        else:
            perform_concat = torch.cat((note_out, y), 2).view(-1, self.encoder_input_size)
            perform_style_contracted = self.performance_contractor(perform_concat).view(1, num_notes, -1)
            perform_style_graphed = self.performance_graph_encoder(perform_style_contracted, edges)
            performance_measure_nodes = self.make_higher_node(perform_style_graphed, self.performance_measure_attention, beat_numbers,
                                                  measure_numbers, start_index, lower_is_note=True)
            perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
            perform_style_vector = self.performance_final_attention(perform_style_encoded)

            # perform_style_reduced = perform_style_reduced.view(-1,self.encoder_input_size)
            # perform_style_node = self.sum_with_attention(perform_style_reduced, self.perform_attention)
            # perform_style_vector = perform_style_encoded[:, -1, :]  # need check
            perform_z, perform_mu, perform_var = \
                self.encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        if return_z:
            total_perform_z = [perform_z]
            for i in range(10):
                temp_z = self.reparameterize(perform_mu, perform_var)
                total_perform_z.append(temp_z)
            total_perform_z = torch.stack(total_perform_z)
            mean_perform_z = torch.mean(total_perform_z, 0, True)

            # mean_perform_z = torch.Tensor(numpy.random.normal(loc=perform_mu, scale=perform_var, size=self.encoded_vector_size)).to(self.device)

            return mean_perform_z

        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)

        initial_output = self.initial_result_fc(note_out)
        num_measures = measure_numbers[start_index+num_notes-1] - measure_numbers[start_index] + 1
        # perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        # perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_hidden_out), 2)
        # measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        # measure_perform_style_spanned = self.span_beat_to_note_num(measure_perform_style, measure_numbers, num_notes, start_index)

        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.time_regressive_size * 2)).to(self.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(self.device)


        num_beats = beat_numbers[start_index + num_notes - 1] - beat_numbers[start_index] + 1
        qpm_primo = x[:, :, QPM_PRIMO_IDX].view(1, -1, 1)
        tempo_primo = x[:, :, TEMPO_PRIMO_IDX:].view(1, -1, 2)
        # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
        beat_qpm_primo = qpm_primo[0, 0, 0].repeat((1, num_beats, 1))
        beat_tempo_primo = tempo_primo[0, 0, :].repeat((1, num_beats, 1))
        beat_tempo_vector = self.note_tempo_infos_to_beat(x, beat_numbers, start_index, TEMPO_IDX)

        total_iterated_output = [initial_output]

        # for i in range(2):

        if self.is_baseline:
            tempo_vector = x[:, :, TEMPO_IDX:TEMPO_IDX + 5].view(1, -1, 5)
            tempo_info_in_note = torch.cat((qpm_primo, tempo_primo, tempo_vector), 2)

            out_with_result = torch.cat(
                (note_out, perform_z_batched, tempo_info_in_note, initial_beat_hidden, initial_output, initial_margin), 2)

            for i in range(self.num_sequence_iteration):
                out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
                initial_out = out_with_result[:, :, -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
                changed_margin = out_with_result[:,:, -self.final_graph_margin_size:]

                margin_in_measure = self.make_higher_node(changed_margin, self.final_margin_attention, measure_numbers,
                                                 measure_numbers, start_index, lower_is_note=True)
                out_in_measure = self.make_higher_node(initial_out, self.final_measure_attention, measure_numbers,
                                                 measure_numbers, start_index, lower_is_note=True)

                out_measure_cat = torch.cat((margin_in_measure, out_in_measure), 2)

                out_beat_rnn_result, _ = self.tempo_rnn(out_measure_cat)
                out_beat_spanned = self.span_beat_to_note_num(out_beat_rnn_result, measure_numbers, num_notes, start_index)
                out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                             out_beat_spanned,
                                             out_with_result[:, :, -self.output_size - self.final_graph_margin_size:]),
                                            2)
                final_out = self.fc(out_with_result)
                out_with_result = torch.cat((out_with_result[:, :, :-self.output_size - self.final_graph_margin_size],
                                             final_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
                # out = torch.cat((out, trill_out), 2)
                total_iterated_output.append(final_out)
        else:
            out_with_result = torch.cat(
                # (note_out, measure_perform_style_spanned, initial_beat_hidden, initial_output, initial_margin), 2)
                (note_out, perform_z_batched, initial_beat_hidden, initial_output, initial_margin), 2)

            for i in range(self.num_sequence_iteration):
                out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
                initial_out = out_with_result[:, :,
                              -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
                changed_margin = out_with_result[:, :, -self.final_graph_margin_size:]

                margin_in_beat = self.make_higher_node(changed_margin, self.final_margin_attention, beat_numbers,
                                                          beat_numbers, start_index, lower_is_note=True)
                out_in_beat = self.make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                       beat_numbers, start_index, lower_is_note=True)
                out_beat_cat = torch.cat((out_in_beat, margin_in_beat, beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), 2)
                out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
                tempo_out = self.tempo_fc(out_beat_rnn_result)
                tempos_spanned = self.span_beat_to_note_num(tempo_out, beat_numbers, num_notes, start_index)
                out_beat_spanned = self.span_beat_to_note_num(out_beat_rnn_result, beat_numbers, num_notes, start_index)
                out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                             out_beat_spanned,
                                             out_with_result[:, :, -self.output_size - self.final_graph_margin_size:]),
                                            2)
                other_out = self.fc(out_with_result)

                final_out = torch.cat((tempos_spanned, other_out), 2)
                out_with_result = torch.cat((out_with_result[:, :, :-self.output_size - self.final_graph_margin_size],
                                             final_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
                total_iterated_output.append(final_out)

        return final_out, perform_mu, perform_var, total_iterated_output

    def run_graph_network(self, nodes, adjacency_matrix, measure_numbers, start_index):
        # 1. Run feed-forward network by note level
        num_notes = nodes.shape[1]
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(self.device)
        notes_and_measure_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden = self.graph_1st(notes_and_measure_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden)
            notes_hidden_second = self.graph_2nd(notes_between, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
                                          notes_hidden_second[:,:, -self.note_hidden_size:]), -1)

            measure_nodes = self.make_higher_node(notes_hidden_cat, self.measure_attention, measure_numbers, measure_numbers,
                                                  start_index, lower_is_note=True)
            measure_hidden, _ = self.measure_rnn(measure_nodes)
            measure_hidden_spanned = self.span_beat_to_note_num(measure_hidden, measure_numbers, num_notes, start_index)
            notes_hidden = torch.cat((measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_hidden, notes_hidden_second),-1)
        return final_out, measure_hidden

    def encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = self.reparameterize(mu, var)
        return z, mu, var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # def decode_with_net(self, z, decode_network):
    #     decode_network
    #     return

    def sum_with_attention(self, hidden, attention_net):
        attention = attention_net(hidden)
        attention = self.softmax(attention)
        upper_node = hidden * attention
        upper_node_sum = torch.sum(upper_node, dim=0)

        return upper_node_sum

    def make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:, lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.stack(higher_nodes).view(1, -1, lower_hidden_size)

        return higher_nodes

    def span_beat_to_note_num(self, beat_out, beat_number, num_notes, start_index):
        start_beat = beat_number[start_index]
        num_beat = beat_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = beat_out.shape[2]
        for i in range(num_notes):
            beat_index = beat_number[start_index+i] - start_beat
            if beat_index >= num_beat:
                beat_index = num_beat-1
            span_mat[0,i,beat_index] = 1
        span_mat = span_mat.to(self.device)

        spanned_beat = torch.bmm(span_mat, beat_out)
        return spanned_beat

    def note_tempo_infos_to_beat(self, y, beat_numbers, start_index, index=None):
        beat_tempos = []
        num_notes = y.size(1)
        prev_beat = -1
        for i in range(num_notes):
            cur_beat = beat_numbers[start_index+i]
            if cur_beat > prev_beat:
                if index is None:
                    beat_tempos.append(y[0,i,:])
                if index == TEMPO_IDX:
                    beat_tempos.append(y[0,i,TEMPO_IDX:TEMPO_IDX+5])
                else:
                    beat_tempos.append(y[0,i,index])
                prev_beat = cur_beat
        num_beats = len(beat_tempos)
        beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
        return beat_tempos


class HAN_Integrated(nn.Module):
    def __init__(self, network_parameters, device, step_by_step=False):
        super(HAN_Integrated, self).__init__()
        self.device = device
        self.step_by_step = step_by_step
        self.is_graph = network_parameters.is_graph
        self.is_teacher_force = network_parameters.is_teacher_force
        self.is_baseline = network_parameters.is_baseline
        self.num_graph_iteration = network_parameters.graph_iteration
        self.hierarchy = network_parameters.hierarchy_level
        if hasattr(network_parameters, 'is_test_version') and network_parameters.is_test_version:
            self.test_version = True
        else:
            self.test_version = False
        # self.is_simplified_note = network_parameters.is_simplified

        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.num_layers = network_parameters.note.layer
        self.hidden_size = network_parameters.note.size
        self.num_beat_layers = network_parameters.beat.layer
        self.beat_hidden_size = network_parameters.beat.size
        self.num_measure_layers = network_parameters.measure.layer
        self.measure_hidden_size = network_parameters.measure.size

        self.final_hidden_size = network_parameters.final.size
        self.num_voice_layers = network_parameters.voice.layer
        self.voice_hidden_size = network_parameters.voice.size
        self.final_input = network_parameters.final.input
        if self.test_version:
            self.final_input -= 1
        self.encoder_size = network_parameters.encoder.size
        self.encoded_vector_size = network_parameters.encoded_vector_size
        self.encoder_input_size = network_parameters.encoder.input
        self.encoder_layer_num = network_parameters.encoder.layer
        self.num_attention_head = network_parameters.num_attention_head
        self.num_edge_types = network_parameters.num_edge_types

        if self.is_graph:
            self.graph_1st = GatedGraph(self.hidden_size, self.num_edge_types)
            self.graph_between = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(DROP_OUT),
                # nn.BatchNorm1d(self.note_hidden_size),
                nn.ReLU()
            )
            self.graph_2nd = GatedGraph(self.hidden_size, self.num_edge_types)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)

        if not self.is_baseline:
            if self.is_graph:
                self.beat_attention = ContextAttention(self.hidden_size * 2, self.num_attention_head)
                self.beat_rnn = nn.LSTM(self.hidden_size * 2, self.beat_hidden_size,
                                        self.num_beat_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
            else:
                self.voice_net = nn.LSTM(self.hidden_size, self.voice_hidden_size, self.num_voice_layers,
                                         batch_first=True, bidirectional=True, dropout=DROP_OUT)
                self.beat_attention = ContextAttention((self.hidden_size + self.voice_hidden_size) * 2,
                                                       self.num_attention_head)
                self.beat_rnn = nn.LSTM((self.hidden_size + self.voice_hidden_size) * 2, self.beat_hidden_size, self.num_beat_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
            self.measure_attention = ContextAttention(self.beat_hidden_size*2, self.num_attention_head)
            self.measure_rnn = nn.LSTM(self.beat_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
            self.perform_style_to_measure = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size, self.encoder_size, num_layers=1, bidirectional=False)

            if self.hierarchy == 'measure':
                self.output_lstm = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size + self.output_size, self.final_hidden_size, num_layers=1, batch_first=True)
            elif self.hierarchy == 'beat':
                self.output_lstm = nn.LSTM((self.beat_hidden_size + self.measure_hidden_size) * 2 + self.encoder_size + self.output_size, self.final_hidden_size, num_layers=1, batch_first=True)
            else:
                if self.step_by_step:
                    if self.test_version:
                        self.beat_tempo_forward = nn.LSTM(
                            (self.beat_hidden_size + self.measure_hidden_size) * 2 + 2 - 1 + self.output_size + self.encoder_size, self.beat_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=False)
                    else:
                        self.beat_tempo_forward = nn.LSTM(
                            (self.beat_hidden_size + self.measure_hidden_size) * 2 + 5 + 3 + self.output_size + self.encoder_size, self.beat_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=False)
                    self.result_for_tempo_attention = ContextAttention(self.output_size - 1, 1)
                else:
                    self.beat_tempo_forward = nn.LSTM((self.beat_hidden_size + self.measure_hidden_size) * 2 + 3 + 3 + self.encoder_size,
                                                      self.beat_hidden_size, num_layers=1, batch_first=True,
                                                      bidirectional=False)
                self.beat_tempo_fc = nn.Linear(self.beat_hidden_size, 1)

        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
        )

        if self.hierarchy:
            self.fc = nn.Linear(self.final_hidden_size, self.output_size)
        else:
            self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
            if self.is_baseline:
                self.fc = nn.Linear(self.final_hidden_size, self.output_size)
            else:
                self.fc = nn.Linear(self.final_hidden_size, self.output_size - 1)

        self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True)
        if self.encoder_size % self.num_attention_head == 0:
            self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        else:
            self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.encoder_size * 2)
        self.performance_embedding_layer = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )
        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, start_index, initial_z=False, rand_threshold=0.2, return_z=False):
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]

        num_notes = x.size(1)

        if self.is_baseline:
            note_out = self.note_fc(x)
            note_out, _ = self.lstm(note_out)
        else:
            note_out, beat_hidden_out, measure_hidden_out = \
                self.run_offline_score_model(x, edges, beat_numbers, measure_numbers, voice_numbers, start_index)
            beat_out_spanned = self.span_beat_to_note_num(beat_hidden_out, beat_numbers, num_notes, start_index)
            measure_out_spanned = self.span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes,
                                                             start_index)
        if type(initial_z) is not bool:
            if type(initial_z) is str and initial_z == 'zero':
                zero_mean = torch.zeros(self.encoded_vector_size)
                one_std = torch.zeros(self.encoded_vector_size)
                perform_z = self.reparameterize(zero_mean, one_std).to(self.device)
            # if type(initial_z) is list:
            #     perform_z = self.reparameterize(torch.Tensor(initial_z), torch.Tensor(initial_z)).to(self.device)
            elif not initial_z.is_cuda:
                perform_z = torch.Tensor(initial_z).to(self.device).view(1,-1)
            else:
                perform_z = initial_z.view(1,-1)
            perform_mu = 0
            perform_var = 0
        else:
            expanded_y = self.performance_embedding_layer(y)
            if self.is_baseline:
                perform_concat = torch.cat((note_out, expanded_y), 2)
            else:
                perform_concat = torch.cat((note_out, beat_out_spanned, measure_out_spanned, expanded_y), 2)
            perform_concat = self.masking_half(perform_concat)
            perform_contracted = self.performance_contractor(perform_concat)
            perform_note_encoded, _ = self.performance_note_encoder(perform_contracted)

            perform_measure = self.make_higher_node(perform_note_encoded, self.performance_measure_attention,
                                                    beat_numbers, measure_numbers, start_index, lower_is_note=True)
            perform_style_encoded, _ = self.performance_encoder(perform_measure)
            perform_style_vector = self.performance_final_attention(perform_style_encoded)
            perform_z, perform_mu, perform_var = \
                self.encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        if return_z:
            total_perform_z = [perform_z]
            for i in range(10):
                temp_z = self.reparameterize(perform_mu, perform_var)
                # temp_z = torch.Tensor(numpy.random.normal(loc=perform_mu, scale=math.exp(0.5*perform_var))).to(self.device)
                total_perform_z.append(temp_z)
            # total_perform_z = torch.stack(total_perform_z)
            # mean_perform_z = torch.mean(total_perform_z, 0, True)
            # var = torch.exp(0.5 * perform_var)
            # mean_perform_z = torch.Tensor(numpy.random.normal(loc=perform_mu, scale=var)).to(self.device)

            # return mean_perform_z
            return total_perform_z

        # perform_z = self.performance_decoder(perform_z)
        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)
        perform_z = perform_z.view(-1)

        if not self.is_baseline:
            tempo_hidden = self.init_hidden(1,1,x.size(0), self.beat_hidden_size)
            num_beats = beat_hidden_out.size(1)
            result_nodes = torch.zeros(num_beats, self.output_size - 1).to(self.device)

            num_measures = measure_numbers[start_index + num_notes - 1] - measure_numbers[start_index] + 1
            perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1, num_measures, -1)
            perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_hidden_out), 2)
            measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
            measure_perform_style_spanned = self.span_beat_to_note_num(measure_perform_style, measure_numbers,
                                                                       num_notes, start_index)

        if self.hierarchy:
            if self.hierarchy == 'measure':
                hierarchy_numbers = measure_numbers
                hierarchy_nodes = measure_hidden_out
            elif self.hierarchy == 'beat':
                hierarchy_numbers = beat_numbers
                beat_measure_concated = torch.cat((beat_out_spanned, measure_out_spanned),2)
                hierarchy_nodes = self.note_tempo_infos_to_beat(beat_measure_concated, hierarchy_numbers, start_index)
            num_hierarchy_nodes = hierarchy_nodes.shape[1]
            if self.test_version:
                hierarchy_nodes_latent_combined = torch.cat((hierarchy_nodes, measure_perform_style), 2)
            else:
                perform_z_batched = perform_z.repeat(num_hierarchy_nodes, 1).view(1, num_hierarchy_nodes, -1)
                hierarchy_nodes_latent_combined = torch.cat((hierarchy_nodes, perform_z_batched),2)

            out_hidden_state = self.init_hidden(1,1,x.size(0), self.final_hidden_size)
            prev_out = torch.zeros(1,1,self.output_size).to(self.device)
            out_total = torch.zeros(1, num_hierarchy_nodes, self.output_size).to(self.device)

            for i in range(num_hierarchy_nodes):
                # print(hierarchy_nodes_latent_combined.shape, prev_out.shape)
                out_combined = torch.cat((hierarchy_nodes_latent_combined[:,i:i+1,:], prev_out),2)
                out, out_hidden_state = self.output_lstm(out_combined, out_hidden_state)
                out = self.fc(out)
                out_total[:,i,:] = out
                prev_out = out.view(1,1,-1)
            return out_total, perform_mu, perform_var, note_out

        else:
            final_hidden = self.init_hidden(1, 1, x.size(0), self.final_hidden_size)
            if self.step_by_step:
                qpm_primo = x[:, 0, QPM_PRIMO_IDX]
                tempo_primo = x[0, 0, TEMPO_PRIMO_IDX:]

                if self.is_teacher_force:
                    true_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)

                # prev_out = y[0, 0, :]
                # prev_tempo = y[:, 0, QPM_INDEX]
                prev_out = torch.zeros(self.output_size).to(self.device)
                prev_tempo = prev_out[QPM_INDEX:QPM_INDEX+1]
                prev_beat = -1
                prev_beat_end = 0
                out_total = torch.zeros(num_notes, self.output_size).to(self.device)
                prev_out_list = []
                # if args.beatTempo:
                #     prev_out[0] = tempos_spanned[0, 0, 0]
                has_ground_truth = y.size(1) > 1
                if self.is_baseline:
                    for i in range(num_notes):
                        out_combined = torch.cat((note_out[0, i, :], prev_out, qpm_primo, tempo_primo, perform_z)).view(1, 1, -1)
                        out, final_hidden = self.output_lstm(out_combined, final_hidden)

                        out = out.view(-1)
                        out = self.fc(out)

                        prev_out_list.append(out)
                        prev_out = out
                        out_total[i, :] = out
                    out_total = out_total.view(1, num_notes, -1)
                    return out_total, perform_mu, perform_var, note_out
                else:
                    for i in range(num_notes):
                        current_beat = beat_numbers[start_index + i] - beat_numbers[start_index]
                        current_measure = measure_numbers[start_index + i] - measure_numbers[start_index]
                        if current_beat > prev_beat:  # beat changed
                            if i - prev_beat_end > 0:  # if there are outputs to consider
                                corresp_result = torch.stack(prev_out_list).unsqueeze_(0)
                            else:  # there is no previous output
                                corresp_result = torch.zeros((1,1,self.output_size-1)).to(self.device)
                            result_node = self.result_for_tempo_attention(corresp_result)
                            prev_out_list = []
                            result_nodes[current_beat, :] = result_node

                            if self.is_teacher_force and current_beat > 0 and random.random() < rand_threshold:
                                prev_tempo = true_tempos[0,current_beat-1,:]

                            tempos = torch.zeros(1, num_beats, 1).to(self.device)
                            if self.test_version:
                                beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :],
                                                            measure_hidden_out[0, current_measure, :], prev_tempo, x[0,i,self.input_size-2:self.input_size-1],
                                                            result_nodes[current_beat, :],
                                                            measure_perform_style[0, current_measure, :])).view(1, 1, -1)
                            else:
                                beat_tempo_vec = x[0, i, TEMPO_IDX:TEMPO_IDX + 5]
                                beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :], measure_hidden_out[0, current_measure,:], prev_tempo,
                                                            qpm_primo, tempo_primo, beat_tempo_vec,
                                                            result_nodes[current_beat, :], perform_z)).view(1, 1, -1)
                            beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)

                            tmp_tempos = self.beat_tempo_fc(beat_forward)

                            prev_beat_end = i
                            prev_tempo = tmp_tempos.view(1)
                            prev_beat = current_beat

                        tmp_voice = voice_numbers[start_index + i] - 1

                        if self.is_teacher_force and i > 0 and random.random() < rand_threshold:
                            prev_out = torch.cat((prev_tempo, y[0, i - 1, 1:]))

                        if self.test_version:
                            out_combined = torch.cat(
                                (note_out[0, i, :], beat_hidden_out[0, current_beat, :],
                                 measure_hidden_out[0, current_measure, :],
                                 prev_out, x[0,i,self.input_size-2:], measure_perform_style[0, current_measure,:])).view(1, 1, -1)
                        else:
                            out_combined = torch.cat(
                                (note_out[0, i, :], beat_hidden_out[0, current_beat, :],
                                 measure_hidden_out[0, current_measure, :],
                                 prev_out, qpm_primo, tempo_primo, perform_z)).view(1, 1, -1)
                        out, final_hidden = self.output_lstm(out_combined, final_hidden)
                        # out = torch.cat((out, out_combined), 2)
                        out = out.view(-1)
                        out = self.fc(out)

                        prev_out_list.append(out)
                        out = torch.cat((prev_tempo, out))

                        prev_out = out
                        out_total[i, :] = out

                    out_total = out_total.view(1, num_notes, -1)
                    hidden_total = torch.cat((note_out, beat_out_spanned, measure_out_spanned), 2)
                    return out_total, perform_mu, perform_var, hidden_total
            else:  # non autoregressive
                qpm_primo = x[:,:,QPM_PRIMO_IDX].view(1,-1,1)
                tempo_primo = x[:,:,TEMPO_PRIMO_IDX:].view(1,-1,2)
                # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
                beat_qpm_primo = qpm_primo[0,0,0].repeat((1, num_beats, 1))
                beat_tempo_primo = tempo_primo[0,0,:].repeat((1, num_beats, 1))
                beat_tempo_vector = self.note_tempo_infos_to_beat(x, beat_numbers, start_index, TEMPO_IDX)
                if 'beat_hidden_out' not in locals():
                    beat_hidden_out = beat_out_spanned
                num_beats = beat_hidden_out.size(1)
                # score_z_beat_spanned = score_z.repeat(num_beats,1).view(1,num_beats,-1)
                perform_z_beat_spanned = perform_z.repeat(num_beats,1).view(1,num_beats,-1)
                beat_tempo_cat = torch.cat((beat_hidden_out, beat_qpm_primo, beat_tempo_primo, beat_tempo_vector, perform_z_beat_spanned), 2)
                beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)
                tempos = self.beat_tempo_fc(beat_forward)
                num_notes = note_out.size(1)
                tempos_spanned = self.span_beat_to_note_num(tempos, beat_numbers, num_notes, start_index)
                # y[0, :, 0] = tempos_spanned.view(-1)

                # mean_velocity_info = x[:, :, mean_vel_start_index:mean_vel_start_index+4].view(1,-1,4)
                # dynamic_info = torch.cat((x[:, :, mean_vel_start_index + 4].view(1,-1,1),
                #                           x[:, :, vel_vec_start_index:vel_vec_start_index + 4]), 2).view(1,-1,5)

                out_combined = torch.cat((
                    note_out, beat_out_spanned, measure_out_spanned,
                    # qpm_primo, tempo_primo, mean_velocity_info, dynamic_info,
                    perform_z_batched), 2)

                out, final_hidden = self.output_lstm(out_combined, final_hidden)

                out = self.fc(out)
                # out = torch.cat((out, trill_out), 2)

                out = torch.cat((tempos_spanned, out), 2)
                score_combined = torch.cat((
                    note_out, beat_out_spanned, measure_out_spanned), 2)

                return out, perform_mu, perform_var, score_combined

    def run_offline_score_model(self, x, edges, beat_numbers, measure_numbers, voice_numbers, start_index):
        hidden = self.init_hidden(self.num_layers, 2, x.size(0), self.hidden_size)
        beat_hidden = self.init_hidden(self.num_beat_layers, 2, x.size(0), self.beat_hidden_size)
        measure_hidden = self.init_hidden(self.num_measure_layers, 2, x.size(0), self.measure_hidden_size)

        x = self.note_fc(x)

        if self.is_graph:
            hidden_out = self.run_graph_network(x, edges)
        else:
            temp_voice_numbers = voice_numbers[start_index:start_index + x.size(1)]
            if temp_voice_numbers == []:
                temp_voice_numbers = voice_numbers[start_index:]
            max_voice = max(temp_voice_numbers)
            voice_hidden = self.init_voice_layer(1, max_voice)
            voice_out, voice_hidden = self.run_voice_net(x, voice_hidden, temp_voice_numbers, max_voice)
            hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
            hidden_out = torch.cat((hidden_out,voice_out), 2)

        beat_nodes = self.make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, start_index, lower_is_note=True)
        beat_hidden_out, beat_hidden = self.beat_rnn(beat_nodes, beat_hidden)
        measure_nodes = self.make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers, start_index)
        measure_hidden_out, measure_hidden = self.measure_rnn(measure_nodes, measure_hidden)

        return hidden_out, beat_hidden_out, measure_hidden_out

    def run_graph_network(self, nodes, graph_matrix):
        # 1. Run feed-forward network by note level
        notes_hidden = self.graph_1st(nodes, graph_matrix, iteration=self.num_graph_iteration)
        notes_between = self.graph_between(notes_hidden)
        notes_hidden_second = self.graph_2nd(notes_between, graph_matrix, iteration=self.num_graph_iteration)
        notes_hidden = torch.cat((notes_hidden, notes_hidden_second),-1)

        return notes_hidden


    def encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = self.reparameterize(mu, var)
        return z, mu, var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # def decode_with_net(self, z, decode_network):
    #     decode_network
    #     return

    def sum_with_attention(self, hidden, attention_net):
        attention = attention_net(hidden)
        attention = self.softmax(attention)
        upper_node = hidden * attention
        upper_node_sum = torch.sum(upper_node, dim=0)

        return upper_node_sum


    def make_higher_node(self, lower_out, attention_weights, lower_indexes, higher_indexes, start_index, lower_is_note=False):
        higher_nodes = []
        prev_higher_index = higher_indexes[start_index]
        lower_node_start = 0
        lower_node_end = 0
        num_lower_nodes = lower_out.shape[1]
        start_lower_index = lower_indexes[start_index]
        lower_hidden_size = lower_out.shape[2]
        for low_index in range(num_lower_nodes):
            absolute_low_index = start_lower_index + low_index
            if lower_is_note:
                current_note_index = start_index + low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[:, lower_node_start:lower_node_end, :]
                higher = attention_weights(corresp_lower_out)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[:, lower_node_start:, :]
        higher = attention_weights(corresp_lower_out)
        higher_nodes.append(higher)

        higher_nodes = torch.cat(higher_nodes, dim=1).view(1,-1,lower_hidden_size)

        return higher_nodes

    def span_beat_to_note_num(self, beat_out, beat_number, num_notes, start_index):
        start_beat = beat_number[start_index]
        num_beat = beat_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = beat_out.shape[2]
        for i in range(num_notes):
            beat_index = beat_number[start_index+i] - start_beat
            if beat_index >= num_beat:
                beat_index = num_beat-1
            span_mat[0,i,beat_index] = 1
        span_mat = span_mat.to(self.device)

        spanned_beat = torch.bmm(span_mat, beat_out)
        return spanned_beat

    def note_tempo_infos_to_beat(self, y, beat_numbers, start_index, index=None):
        beat_tempos = []
        num_notes = y.size(1)
        prev_beat = -1
        for i in range(num_notes):
            cur_beat = beat_numbers[start_index+i]
            if cur_beat > prev_beat:
                if index is None:
                    beat_tempos.append(y[0,i,:])
                elif index == TEMPO_IDX:
                    beat_tempos.append(y[0,i,TEMPO_IDX:TEMPO_IDX+5])
                else:
                    beat_tempos.append(y[0,i,index])
                prev_beat = cur_beat
        num_beats = len(beat_tempos)
        beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
        return beat_tempos

    def run_voice_net(self, batch_x, voice_hidden, voice_numbers, max_voice):
        num_notes = batch_x.size(1)
        output = torch.zeros(1, batch_x.size(1), self.voice_hidden_size * 2).to(self.device)
        voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1,max_voice+1):
            voice_x_bool = voice_numbers == i
            num_voice_notes = torch.sum(voice_x_bool)
            if num_voice_notes > 0:
                span_mat = torch.zeros(num_notes, num_voice_notes)
                note_index_in_voice = 0
                for j in range(num_notes):
                    if voice_x_bool[j] ==1:
                        span_mat[j, note_index_in_voice] = 1
                        note_index_in_voice += 1
                span_mat = span_mat.view(1,num_notes,-1).to(self.device)
                voice_x = batch_x[0,voice_x_bool,:].view(1,-1, self.hidden_size)
                ith_hidden = voice_hidden[i-1]

                ith_voice_out, ith_hidden = self.voice_net(voice_x, ith_hidden)
                # ith_voice_out, ith_hidden = self.lstm(voice_x, ith_hidden)
                output += torch.bmm(span_mat, ith_voice_out)
        return output, voice_hidden

    def masking_half(self, y):
        num_notes = y.shape[1]
        y = y[:,:num_notes//2,:]
        return y

    def init_hidden(self, num_layer, num_direction, batch_size, hidden_size):
        h0 = torch.zeros(num_layer * num_direction, batch_size, hidden_size).to(self.device)
        return (h0, h0)

    def init_voice_layer(self, batch_size, max_voice):
        layers = []
        for i in range(max_voice):
            # h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(device)
            h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(self.device)
            layers.append((h0, h0))
        return layers



class TrillRNN(nn.Module):
    def __init__(self, network_parameters, device):
        super(TrillRNN, self).__init__()
        self.hidden_size = network_parameters.note.size
        self.num_layers = network_parameters.note.layer
        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
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
    def __init__(self, network_parameters, trill_index, loss_type, device):
        super(TrillGraph, self).__init__()
        self.loss_type = loss_type
        self.hidden_size = network_parameters.note.size
        self.num_layers = network_parameters.note.layer
        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.num_edge_types = network_parameters.num_edge_types
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
