import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
import random
import numpy
import math

from torch.nn.modules import dropout

from . import model_constants as cons
from .model_utils import make_higher_node, reparameterize, span_beat_to_note_num
from . import model_utils as utils
from .module import GatedGraph, SimpleAttention, ContextAttention
from .score_encoder import IsgnResEncoder, IsgnResEncoderV2

QPM_INDEX = 0
# VOICE_IDX = 11
PITCH_IDX = 12
TEMPO_IDX = PITCH_IDX + 13
DYNAMICS_IDX = TEMPO_IDX + 5
LEN_DYNAMICS_VEC = 4
QPM_PRIMO_IDX = 4
TEMPO_PRIMO_IDX = -2
NUM_VOICE_FEED_PARAM = 2



class VirtuosoNet(nn.Module):
    def __init__(self, network_params, device):
        super(VirtuosoNet, self).__init__()
        self.network_params = network_params

        self.score_encoder
        self.performance_encoder



class ISGN(nn.Module):
    def __init__(self, net_params, device):
        super(ISGN, self).__init__()
        self.device = device
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        self.is_graph = True
        self.network_params = net_params
        self.is_baseline = net_params.is_baseline

        self.input_size = net_params.input_size
        self.output_size = net_params.output_size
        self.num_layers = net_params.note.layer
        self.note_hidden_size = net_params.note.size
        self.num_measure_layers = net_params.measure.layer
        self.measure_hidden_size = net_params.measure.size
        self.final_hidden_size = net_params.final.size
        self.final_input = net_params.final.input
        self.encoder_size = net_params.encoder.size
        self.encoded_vector_size = net_params.encoded_vector_size
        self.encoder_input_size = net_params.encoder.input
        self.encoder_layer_num = net_params.encoder.layer
        self.time_regressive_size = net_params.time_reg.size
        self.time_regressive_layer = net_params.time_reg.layer
        self.num_edge_types = net_params.num_edge_types
        self.final_graph_margin_size = net_params.final.margin
        self.drop_out = net_params.drop_out
        self.num_attention_head = net_params.num_attention_head


        if self.is_baseline:
            self.final_graph_input_size = self.final_input + self.encoder_size + 8 + self.output_size + self.final_graph_margin_size + self.time_regressive_size * 2
            self.final_beat_hidden_idx = self.final_input + self.encoder_size + 8 # tempo info
        else:
            self.final_graph_input_size = self.final_input + self.encoder_size + self.output_size + self.final_graph_margin_size + self.time_regressive_size * 2
            self.final_beat_hidden_idx = self.final_input + self.encoder_size

        # self.num_attention_head = 4


        '''
        self.note_fc = nn.Sequential(
            nn.Linear(self.input_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
        )

        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, secondary_size=self.note_hidden_size)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(self.drop_out),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types)

        if net_params.use_simple_attention:
            self.measure_attention = SimpleAttention(self.note_hidden_size * 2)
        else:
            self.measure_attention = ContextAttention(self.note_hidden_size * 2, self.num_attention_head)
        self.measure_rnn = nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
        '''
        self.score_encoder = IsgnResEncoderV2(net_params)

        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(self.drop_out),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU(),
            nn.Linear(self.encoder_size, self.encoder_size),
            nn.Dropout(self.drop_out),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_graph_encoder = GatedGraph(self.encoder_size, self.num_edge_types)
        self.performance_measure_attention = ContextAttention(self.encoder_size, self.num_attention_head)

        self.performance_encoder = nn.LSTM(self.encoder_size, self.encoder_size, num_layers=self.encoder_layer_num,
                                           batch_first=True, bidirectional=True)

        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

        self.beat_tempo_contractor = nn.Sequential(
            nn.Linear(self.final_graph_input_size - self.time_regressive_size * 2, self.time_regressive_size),
            nn.Dropout(self.drop_out),
            nn.ReLU()
        )
        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(self.drop_out),
            nn.ReLU()
        )
        self.perform_style_to_measure = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size, self.encoder_size, num_layers=1, bidirectional=False)

        self.initial_result_fc = nn.Sequential(
            nn.Linear(self.final_input, self.encoder_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),

            nn.Linear(self.encoder_size, self.output_size),
            nn.ReLU()
        )

        self.final_graph = GatedGraph(self.final_graph_input_size, self.num_edge_types,
                                      self.output_size + self.final_graph_margin_size)
        # if self.is_baseline:
        #     self.tempo_rnn = nn.LSTM(self.final_graph_margin_size + self.output_size, self.time_regressive_size,
        #                              num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)
        #     self.final_measure_attention = ContextAttention(self.output_size, 1)
        #     self.final_margin_attention = ContextAttention(self.final_graph_margin_size, self.num_attention_head)

        #     self.fc = nn.Sequential(
        #         nn.Linear(self.final_graph_input_size, self.final_graph_margin_size),
        #         nn.Dropout(self.drop_out),
        #         nn.ReLU(),
        #         nn.Linear(self.final_graph_margin_size, self.output_size),
        #     )
        # # elif self.is_test_version:
        # else:
        self.tempo_rnn = nn.LSTM(self.final_graph_margin_size + self.output_size + 8, self.time_regressive_size,
                                    num_layers=self.time_regressive_layer, batch_first=True, bidirectional=True)

        self.final_beat_attention = ContextAttention(self.output_size, 1)
        self.final_margin_attention = ContextAttention(self.final_graph_margin_size, self.num_attention_head)
        self.tempo_fc = nn.Linear(self.time_regressive_size * 2, 1)

        self.fc = nn.Sequential(
            nn.Linear(self.final_graph_input_size, self.final_graph_margin_size),
            nn.Dropout(self.drop_out),
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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, initial_z=False, return_z=False):
        times = [] 
        times.append(time.perf_counter())
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        section_numbers = note_locations['section']
        num_notes = x.size(1)

        # note_out, measure_hidden_out = self.run_graph_network(x, edges, measure_numbers)
        note_out, measure_hidden_out = self.score_encoder(x, edges, measure_numbers)
        times.append(time.perf_counter())

        if type(initial_z) is not bool:
            if type(initial_z) is str and initial_z == 'zero':
                # zero_mean = torch.zeros(self.encoded_vector_size)
                # one_std = torch.ones(self.encoded_vector_size)
                # perform_z = reparameterize(zero_mean, one_std).to(self.device)
                perform_z = torch.Tensor(numpy.random.normal(size=self.encoded_vector_size)).to(x.device)
            # if type(initial_z) is list:
            #     perform_z = reparameterize(torch.Tensor(initial_z), torch.Tensor(initial_z)).to(self.device)
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
            performance_measure_nodes = make_higher_node(perform_style_graphed, self.performance_measure_attention, beat_numbers,
                                                  measure_numbers, lower_is_note=True)
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
                temp_z = reparameterize(perform_mu, perform_var)
                total_perform_z.append(temp_z)
            total_perform_z = torch.stack(total_perform_z)
            # mean_perform_z = torch.mean(total_perform_z, 0, True)

            # mean_perform_z = torch.Tensor(numpy.random.normal(loc=perform_mu, scale=perform_var, size=self.encoded_vector_size)).to(self.device)
            return total_perform_z

        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)

        initial_output = self.initial_result_fc(note_out)
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        # perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        # perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_hidden_out), 2)
        # measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        # measure_perform_style_spanned = self.span_beat_to_note_num(measure_perform_style, measure_numbers, num_notes, start_index)

        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.time_regressive_size * 2)).to(self.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(self.device)

        num_beats = beat_numbers[-1] - beat_numbers[0] + 1
        qpm_primo = x[:, :, QPM_PRIMO_IDX].view(1, -1, 1)
        tempo_primo = x[:, :, TEMPO_PRIMO_IDX:].view(1, -1, 2)
        # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
        beat_qpm_primo = qpm_primo[0, 0, 0].repeat((1, num_beats, 1))
        beat_tempo_primo = tempo_primo[0, 0, :].repeat((1, num_beats, 1))
        beat_tempo_vector = self.note_tempo_infos_to_beat(x, beat_numbers, TEMPO_IDX)

        total_iterated_output = [initial_output]

        if self.is_baseline:
            tempo_vector = x[:, :, TEMPO_IDX:TEMPO_IDX + 5].view(1, -1, 5)
            tempo_info_in_note = torch.cat((qpm_primo, tempo_primo, tempo_vector), 2)

            out_with_result = torch.cat(
                (note_out, perform_z_batched, tempo_info_in_note, initial_beat_hidden, initial_output, initial_margin), 2)

            for i in range(self.num_sequence_iteration):
                out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
                initial_out = out_with_result[:, :, -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
                changed_margin = out_with_result[:,:, -self.final_graph_margin_size:]

                margin_in_measure = make_higher_node(changed_margin, self.final_margin_attention, measure_numbers,
                                                 measure_numbers, lower_is_note=True)
                out_in_measure = make_higher_node(initial_out, self.final_measure_attention, measure_numbers,
                                                 measure_numbers, lower_is_note=True)

                out_measure_cat = torch.cat((margin_in_measure, out_in_measure), 2)

                out_beat_rnn_result, _ = self.tempo_rnn(out_measure_cat)
                out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, measure_numbers, num_notes)
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
                times=[]
                out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)

                initial_out = out_with_result[:, :,
                              -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
                changed_margin = out_with_result[:, :, -self.final_graph_margin_size:]

                margin_in_beat = make_higher_node(changed_margin, self.final_margin_attention, beat_numbers,
                                                          beat_numbers, lower_is_note=True)
                out_in_beat = make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                       beat_numbers, lower_is_note=True)
                out_beat_cat = torch.cat((out_in_beat, margin_in_beat, beat_qpm_primo, beat_tempo_primo, beat_tempo_vector), 2)
                out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
                tempo_out = self.tempo_fc(out_beat_rnn_result)

                tempos_spanned = span_beat_to_note_num(tempo_out, beat_numbers, num_notes)
                out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, beat_numbers, num_notes)

                out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                             out_beat_spanned,
                                             out_with_result[:, :, -self.output_size - self.final_graph_margin_size:]),
                                            2)
                other_out = self.fc(out_with_result)

                final_out = torch.cat((tempos_spanned, other_out), 2)
                out_with_result = torch.cat((out_with_result[:, :, :-self.output_size - self.final_graph_margin_size],
                                             final_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
                total_iterated_output.append(final_out)
                # print([times[i]-times[i-1] for i in range(1, len(times))])
        return final_out, perform_mu, perform_var, total_iterated_output

    def run_graph_network(self, nodes, adjacency_matrix, measure_numbers):
        # 1. Run feed-forward network by note level
        num_notes = nodes.shape[1]
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(self.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden)
            notes_hidden_second = self.graph_2nd(notes_between, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
                                          notes_hidden_second[:,:, -self.note_hidden_size:]), -1)

            measure_nodes = make_higher_node(notes_hidden_cat, self.measure_attention, measure_numbers, measure_numbers,
                                                  lower_is_note=True)
            measure_hidden, _ = self.measure_rnn(measure_nodes)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers, num_notes)
            notes_hidden = torch.cat((measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_hidden, notes_hidden_second),-1)
        return final_out, measure_hidden

    def encode_with_net(self, score_input, mean_net, var_net):
        mu = mean_net(score_input)
        var = var_net(score_input)

        z = reparameterize(mu, var)
        return z, mu, var


    def note_tempo_infos_to_beat(self, y, beat_numbers, index=None):
        beat_tempos = []
        num_notes = y.size(1)
        prev_beat = -1
        for i in range(num_notes):
            cur_beat = beat_numbers[i]
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
    def __init__(self, net_params, device, step_by_step=False):
        super(HAN_Integrated, self).__init__()
        self.device = device
        self.step_by_step = step_by_step
        self.is_graph = net_params.is_graph
        self.is_teacher_force = net_params.is_teacher_force
        self.is_baseline = net_params.is_baseline
        self.num_graph_iteration = net_params.graph_iteration
        self.hierarchy = net_params.hierarchy_level
        self.drop_out = net_params.drop_out
        self.network_params = net_params

        if hasattr(net_params, 'is_test_version') and net_params.is_test_version:
            self.test_version = True
        else:
            self.test_version = False
        # self.is_simplified_note = net_params.is_simplified

        self.input_size = net_params.input_size
        self.output_size = net_params.output_size
        self.num_layers = net_params.note.layer
        self.hidden_size = net_params.note.size
        self.num_beat_layers = net_params.beat.layer
        self.beat_hidden_size = net_params.beat.size
        self.num_measure_layers = net_params.measure.layer
        self.measure_hidden_size = net_params.measure.size
        self.performance_embedding_size = net_params.performance.size

        self.final_hidden_size = net_params.final.size
        self.num_voice_layers = net_params.voice.layer
        self.voice_hidden_size = net_params.voice.size
        self.final_input = net_params.final.input
        if self.test_version:
            self.final_input -= 1
        self.encoder_size = net_params.encoder.size
        self.encoded_vector_size = net_params.encoded_vector_size
        self.encoder_input_size = net_params.encoder.input
        self.encoder_layer_num = net_params.encoder.layer
        self.num_attention_head = net_params.num_attention_head
        self.num_edge_types = net_params.num_edge_types

        if self.is_graph:
            self.graph_1st = GatedGraph(self.hidden_size, self.num_edge_types)
            self.graph_between = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(self.drop_out),
                # nn.BatchNorm1d(self.note_hidden_size),
                nn.ReLU()
            )
            self.graph_2nd = GatedGraph(self.hidden_size, self.num_edge_types)
        else:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=self.drop_out)

        if not self.is_baseline:
            if self.is_graph:
                self.beat_attention = ContextAttention(self.hidden_size * 2, self.num_attention_head)
                self.beat_rnn = nn.LSTM(self.hidden_size * 2, self.beat_hidden_size,
                                        self.num_beat_layers, batch_first=True, bidirectional=True, dropout=self.drop_out)
            else:
                self.voice_net = nn.LSTM(self.hidden_size, self.voice_hidden_size, self.num_voice_layers,
                                         batch_first=True, bidirectional=True, dropout=self.drop_out)
                self.beat_attention = ContextAttention((self.hidden_size + self.voice_hidden_size) * 2,
                                                       self.num_attention_head)
                self.beat_rnn = nn.LSTM((self.hidden_size + self.voice_hidden_size) * 2, self.beat_hidden_size, self.num_beat_layers, batch_first=True, bidirectional=True, dropout=self.drop_out)
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
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.drop_out),
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
            nn.Linear(self.output_size, self.performance_embedding_size),
            nn.Dropout(self.drop_out),
            nn.ReLU(),
            nn.Linear(self.performance_embedding_size, self.performance_embedding_size),
            nn.Dropout(self.drop_out),
            nn.ReLU()
        )
        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(self.drop_out),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )
        self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(self.drop_out),
            nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, edges, note_locations, initial_z=False, rand_threshold=0.2, return_z=False):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        voice_numbers = note_locations['voice']

        num_notes = x.size(1)

        if self.is_baseline:
            note_out = self.note_fc(x)
            note_out, _ = self.lstm(note_out)
        else:
            note_out, beat_hidden_out, measure_hidden_out = \
                self.run_offline_score_model(x, edges, beat_numbers, measure_numbers, voice_numbers)
            beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers, num_notes)
            measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes)
        if type(initial_z) is not bool:
            if type(initial_z) is str and initial_z == 'zero':
                zero_mean = torch.zeros(self.encoded_vector_size)
                one_std = torch.zeros(self.encoded_vector_size)
                perform_z = reparameterize(zero_mean, one_std).to(self.device)
            # if type(initial_z) is list:
            #     perform_z = reparameterize(torch.Tensor(initial_z), torch.Tensor(initial_z)).to(self.device)
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

            perform_measure = make_higher_node(perform_note_encoded, self.performance_measure_attention,
                                                    beat_numbers, measure_numbers, lower_is_note=True)
            perform_style_encoded, _ = self.performance_encoder(perform_measure)
            perform_style_vector = self.performance_final_attention(perform_style_encoded)
            perform_z, perform_mu, perform_var = \
                self.encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        if return_z:
            total_perform_z = [perform_z]
            for i in range(10):
                temp_z = reparameterize(perform_mu, perform_var)
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

            # num_measures = measure_numbers[start_index + num_notes - 1] - measure_numbers[start_index] + 1
            num_measures = measure_numbers[-1] - measure_numbers[0] + 1
            perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1, num_measures, -1)
            if perform_z_measure_spanned.shape[1] != measure_hidden_out.shape[1]:
                print(measure_numbers)
            perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_hidden_out), 2)
            measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
            measure_perform_style_spanned = span_beat_to_note_num(measure_perform_style, measure_numbers,
                                                                       num_notes)

        if self.hierarchy:
            if self.hierarchy == 'measure':
                hierarchy_numbers = measure_numbers
                hierarchy_nodes = measure_hidden_out
            elif self.hierarchy == 'beat':
                hierarchy_numbers = beat_numbers
                beat_measure_concated = torch.cat((beat_out_spanned, measure_out_spanned),2)
                hierarchy_nodes = self.note_tempo_infos_to_beat(beat_measure_concated, hierarchy_numbers)
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
                    true_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, 0, QPM_INDEX)

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
                        current_beat = beat_numbers[i] - beat_numbers[0]
                        current_measure = measure_numbers[i] - measure_numbers[0]
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

                        tmp_voice = voice_numbers[i] - 1

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
                tempos_spanned = span_beat_to_note_num(tempos, beat_numbers, num_notes, start_index)
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

    def run_offline_score_model(self, x, edges, beat_numbers, measure_numbers, voice_numbers):
        hidden = self.init_hidden(self.num_layers, 2, x.size(0), self.hidden_size)
        beat_hidden = self.init_hidden(self.num_beat_layers, 2, x.size(0), self.beat_hidden_size)
        measure_hidden = self.init_hidden(self.num_measure_layers, 2, x.size(0), self.measure_hidden_size)

        x = self.note_fc(x)

        if self.is_graph:
            hidden_out = self.run_graph_network(x, edges)
        else:
            # temp_voice_numbers = voice_numbers[start_index:start_index + x.size(1)]
            # if temp_voice_numbers == []:
            #     temp_voice_numbers = voice_numbers[start_index:]
            max_voice = max(voice_numbers)
            voice_hidden = self.init_voice_layer(1, max_voice)
            voice_out, voice_hidden = self.run_voice_net(x, voice_hidden, voice_numbers, max_voice)
            hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
            hidden_out = torch.cat((hidden_out,voice_out), 2)

        beat_nodes = make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, lower_is_note=True)
        beat_hidden_out, beat_hidden = self.beat_rnn(beat_nodes, beat_hidden)
        measure_nodes = make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers)
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

        z = reparameterize(mu, var)
        return z, mu, var


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
        # voice_numbers = torch.Tensor(voice_numbers)
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


