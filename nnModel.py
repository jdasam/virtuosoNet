import torch
import torch.nn as nn
import time
from torch.autograd import Variable



DROP_OUT = 0.25

QPM_INDEX = 0
TEMPO_IDX = 27
PITCH_IDX = 14
QPM_PRIMO_IDX = 5
TEMPO_PRIMO_IDX = -2
GRAPH_KEYS = ['onset', 'forward', 'melisma', 'rest', 'voice', 'boundary', 'closest']
N_EDGE_TYPE = len(GRAPH_KEYS) * 2



class GatedGraph(nn.Module):
    class subGraph():
        def __init__(self, size, device):
            self.wz = torch.nn.Parameter(torch.Tensor(size, size)).to(device)
            self.wr = torch.nn.Parameter(torch.Tensor(size, size)).to(device)
            self.wh = torch.nn.Parameter(torch.Tensor(size, size)).to(device)
            nn.init.xavier_normal_(self.wz)
            nn.init.xavier_normal_(self.wr)
            nn.init.xavier_normal_(self.wh)

    def  __init__(self, size, num_edge_style, device):
        super(GatedGraph, self).__init__()
        self.sub = []
        # for i in range(num_edge_style):
        #     subgraph = self.subGraph(size)
        #     self.sub.append(subgraph)

        self.wz = torch.nn.Parameter(torch.Tensor(num_edge_style,size,size))
        self.wr = torch.nn.Parameter(torch.Tensor(num_edge_style,size,size))
        self.wh = torch.nn.Parameter(torch.Tensor(num_edge_style,size,size))
        self.uz = torch.nn.Parameter(torch.Tensor(size, size)).to(device)
        self.bz = torch.nn.Parameter(torch.Tensor(size)).to(device)
        self.ur = torch.nn.Parameter(torch.Tensor(size, size)).to(device)
        self.br = torch.nn.Parameter(torch.Tensor(size)).to(device)
        self.uh = torch.nn.Parameter(torch.Tensor(size, size)).to(device)
        self.bh = torch.nn.Parameter(torch.Tensor(size)).to(device)

        nn.init.xavier_normal_(self.wz)
        nn.init.xavier_normal_(self.wr)
        nn.init.xavier_normal_(self.wh)
        nn.init.xavier_normal_(self.uz)
        nn.init.xavier_normal_(self.ur)
        nn.init.xavier_normal_(self.uh)
        nn.init.zeros_(self.bz)
        nn.init.zeros_(self.br)
        nn.init.zeros_(self.bh)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, input, edge_matrix, iteration=20):

        for i in range(iteration):
            activation = torch.matmul(edge_matrix.transpose(1,2), input)
            temp_z = self.sigmoid( torch.bmm(activation, self.wz).sum(0) + torch.matmul(input, self.uz) + self.bz)
            temp_r = self.sigmoid( torch.bmm(activation, self.wr).sum(0) + torch.matmul(input, self.ur) + self.br)
            temp_hidden = self.tanh(torch.bmm(activation, self.wh).sum(0) + torch.matmul(temp_r * input, self.uh) + self.bh)

            input = (1 - temp_z) * input + temp_r * temp_hidden

        return input


class GGNN_HAN(nn.Module):
    def __init__(self, network_parameters, device, num_trill_param=5):
        super(GGNN_HAN, self).__init__()
        self.device = device
        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.num_layers = network_parameters.note.layer
        self.note_hidden_size = network_parameters.note.size
        self.num_beat_layers = network_parameters.beat.layer
        self.beat_hidden_size = network_parameters.beat.size
        self.num_measure_layers = network_parameters.measure.layer
        self.measure_hidden_size = network_parameters.measure.size
        self.final_hidden_size = network_parameters.final.size
        self.num_voice_layers = network_parameters.voice.layer
        self.voice_hidden_size = network_parameters.voice.size
        self.final_input = network_parameters.final.input
        self.encoder_size = network_parameters.encoder.size
        self.encoder_input_size = network_parameters.encoder.input
        self.encoder_layer_num = network_parameters.encoder.layer
        self.onset_hidden_size = network_parameters.onset.size
        self.num_onset_layers = network_parameters.onset.layer

        self.beat_attention = nn.Linear(self.note_hidden_size * 2, self.note_hidden_size * 2)
        self.beat_rnn = nn.LSTM(self.note_hidden_size * 2, self.beat_hidden_size, self.num_beat_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        self.measure_attention = nn.Linear(self.beat_hidden_size*2, self.beat_hidden_size*2)
        self.measure_rnn = nn.LSTM(self.beat_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
        # self.tempo_attention = nn.Linear(self.output_size-1, self.output_size-1)

        self.beat_tempo_forward = nn.LSTM(self.beat_hidden_size*2 + 3+ 3 + self.encoder_size, self.beat_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.beat_tempo_fc = nn.Linear(self.beat_hidden_size,  1)

        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.final_hidden_size, self.output_size - 1)

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

        self.graph_1st = GatedGraph(self.note_hidden_size, N_EDGE_TYPE, self.device)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size, self.note_hidden_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size, N_EDGE_TYPE, self.device)

        self.performance_graph_encoder = GatedGraph(self.note_hidden_size, N_EDGE_TYPE, self.device)
        self.performance_measure_attention = nn.Linear(self.encoder_size, self.encoder_size)

        self.performance_contractor = nn.Sequential(
            nn.Linear(self.encoder_input_size, self.encoder_size),
            nn.Dropout(DROP_OUT),
            # nn.BatchNorm1d(self.encoder_size),
            nn.ReLU()
        )

        self.performance_encoder = nn.LSTM(self.encoder_size, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=False)
        self.performance_encoder_mean = nn.Linear(self.encoder_size, self.encoder_size)
        self.performance_encoder_var = nn.Linear(self.encoder_size, self.encoder_size)

        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x, y, edges, note_locations, start_index, step_by_step = False, initial_z=False, rand_threshold=0.7):
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]
        onset_numbers = [x.onset for x in note_locations]
        num_notes = x.size(1)

        note_out, beat_hidden_out, measure_hidden_out = \
            self.run_offline_score_model(x, edges, onset_numbers, beat_numbers, measure_numbers, voice_numbers, start_index)
        beat_out_spanned = self.span_beat_to_note_num(beat_hidden_out, beat_numbers, num_notes, start_index)
        measure_out_spanned = self.span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes, start_index)
        if initial_z:
            perform_z = torch.Tensor(initial_z).to(self.device).view(1,-1)
            perform_mu = 0
            perform_var = 0
        else:
            perform_concat = torch.cat((note_out, beat_out_spanned, measure_out_spanned, y), 2).view(-1, self.encoder_input_size)
            perform_style_contracted = self.performance_contractor(perform_concat).view(1, num_notes, -1)
            perform_style_graphed = self.performance_graph_encoder(perform_style_contracted, edges)
            performance_measure_nodes = self.make_higher_node(perform_style_graphed, self.performance_measure_attention, beat_numbers,
                                                  measure_numbers, start_index, lower_is_note=True)
            perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)

            # perform_style_reduced = perform_style_reduced.view(-1,self.encoder_input_size)
            # perform_style_node = self.sum_with_attention(perform_style_reduced, self.perform_attention)
            perform_style_vector = perform_style_encoded[:, -1, :]  # need check
            perform_z, perform_mu, perform_var = \
                self.encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)

        # perform_z = self.performance_decoder(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)
        num_notes = x.size(1)

        tempo_hidden = self.init_beat_tempo_forward(x.size(0))
        final_hidden = self.init_final_layer(x.size(0))

        num_notes = x.size(1)
        num_beats = beat_hidden_out.size(1)

        # non autoregressive

        qpm_primo = x[:,:, QPM_PRIMO_IDX].view(1, -1, 1)
        tempo_primo = x[:,:, TEMPO_PRIMO_IDX:].view(1, -1, 2)
        # beat_tempos = self.note_tempo_infos_to_beat(y, beat_numbers, start_index, QPM_INDEX)
        beat_qpm_primo = qpm_primo[0,0,0].repeat((1, num_beats, 1))
        beat_tempo_primo = tempo_primo[0,0,:].repeat((1, num_beats, 1))
        beat_tempo_vector = self.note_tempo_infos_to_beat(x, beat_numbers, start_index, TEMPO_IDX)
        # measure_out_in_beat
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



        return out, perform_mu, perform_var, note_out

    def run_offline_score_model(self, x, edges, onset_numbers, beat_numbers, measure_numbers, voice_numbers, start_index):
        x = x[0,:,:]
        beat_hidden = self.init_beat_layer(1)
        measure_hidden = self.init_measure_layer(1)

        note_out = self.run_graph_network(x, edges, start_index)
        note_out = note_out.view(1,note_out.shape[0], note_out.shape[1])
        # note_out, onset_out = self.run_onset_rnn(x, voice_out, onset_numbers, start_index)
        # hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # beat_nodes = self.make_higher_node(onset_out, self.beat_attention, onset_numbers, beat_numbers, start_index)
        beat_nodes = self.make_beat_node(note_out, beat_numbers, start_index)
        beat_hidden_out, beat_hidden = self.beat_rnn(beat_nodes, beat_hidden)
        measure_nodes = self.make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers, start_index)
        # measure_nodes = self.make_measure_node(beat_hidden_out, measure_numbers, beat_numbers, start_index)
        measure_hidden_out, measure_hidden = self.measure_rnn(measure_nodes, measure_hidden)

        return note_out, beat_hidden_out, measure_hidden_out

    def run_graph_network(self, nodes, graph_matrix, start_index):
        # 1. Run feed-forward network by note level

        notes_hidden = self.note_fc(nodes)
        num_notes = notes_hidden.size(1)

        notes_hidden = self.graph_1st(notes_hidden, graph_matrix)
        time3 = time.time()

        notes_between = self.graph_between(notes_hidden)

        notes_hidden_second = self.graph_2nd(notes_between, graph_matrix)

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
                current_note_index = absolute_low_index
            else:
                current_note_index = lower_indexes.index(absolute_low_index)

            if higher_indexes[current_note_index] > prev_higher_index:
                # new beat start
                lower_node_end = low_index
                corresp_lower_out = lower_out[0, lower_node_start:lower_node_end, :]
                higher = self.sum_with_attention(corresp_lower_out, attention_weights)
                higher_nodes.append(higher)

                lower_node_start = low_index
                prev_higher_index = higher_indexes[current_note_index]

        corresp_lower_out = lower_out[0, lower_node_start:, :]
        higher = self.sum_with_attention(corresp_lower_out, attention_weights)
        higher_nodes.append(higher)

        higher_nodes = torch.stack(higher_nodes).view(1, -1, lower_hidden_size)

        return higher_nodes


    def make_beat_node(self, hidden_out, beat_number, start_index):
        beat_nodes = []
        prev_beat = beat_number[start_index]
        beat_notes_start = 0
        beat_notes_end = 0
        num_notes = hidden_out.shape[1]
        for note_index in range(num_notes):
            actual_index = start_index + note_index
            if beat_number[actual_index] > prev_beat:
                #new beat start
                beat_notes_end = note_index
                corresp_hidden = hidden_out[0, beat_notes_start:beat_notes_end, :]
                beat = self.sum_with_attention(corresp_hidden, self.beat_attention)
                beat_nodes.append(beat)

                beat_notes_start = note_index
                prev_beat = beat_number[actual_index]

        last_hidden =  hidden_out[0, beat_notes_end:, :]
        beat = self.sum_with_attention(last_hidden, self.beat_attention)
        beat_nodes.append(beat)

        beat_nodes = torch.stack(beat_nodes).view(1, -1, self.note_hidden_size * 2)
        # beat_nodes = torch.Tensor(beat_nodes)

        return beat_nodes

    def make_measure_node(self, beat_out, measure_number, beat_number, start_index):
        measure_nodes = []
        prev_measure = measure_number[start_index]
        measure_beats_start = 0
        measure_beats_end = 0
        num_beats = beat_out.shape[1]
        start_beat = beat_number[start_index]
        for beat_index in range(num_beats):
            current_beat = start_beat + beat_index
            current_note_index = beat_number.index(current_beat)

            if measure_number[current_note_index] > prev_measure:
                # new beat start
                measure_beats_end = beat_index
                corresp_hidden = beat_out[0, measure_beats_start:measure_beats_end, :]
                measure = self.sum_with_attention(corresp_hidden, self.measure_attention)
                measure_nodes.append(measure)

                measure_beats_start = beat_index
                prev_measure = measure_number[beat_index]

        last_hidden = beat_out[0, measure_beats_end:, :]
        measure = self.sum_with_attention(last_hidden, self.measure_attention)
        measure_nodes.append(measure)

        measure_nodes = torch.stack(measure_nodes).view(1,-1,self.beat_hidden_size*2)

        return measure_nodes

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
                    beat_tempos.append(y[0,i,TEMPO_IDX:TEMPO_IDX+3])
                else:
                    beat_tempos.append(y[0,i,index])
                prev_beat = cur_beat
        num_beats = len(beat_tempos)
        beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
        return beat_tempos

    def measure_to_beat_span(self, measure_out, measure_numbers, beat_numbers, start_index):
        pass


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
                voice_x = batch_x[0,voice_x_bool,:].view(1,-1, self.input_size)
                ith_hidden = voice_hidden[i-1]

                ith_voice_out, ith_hidden = self.voice_net(voice_x, ith_hidden)
                # ith_voice_out, ith_hidden = self.lstm(voice_x, ith_hidden)
                output += torch.bmm(span_mat, ith_voice_out)
        return output, voice_hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.note_hidden_size).to(self.device)
        return (h0, h0)

    def init_final_layer(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.final_hidden_size).to(self.device)
        return (h0, h0)

    def init_onset_layer(self, batch_size):
        h0 = torch.zeros(self.num_onset_layers * 2, batch_size, self.onset_hidden_size).to(self.device)
        return (h0, h0)

    def init_beat_layer(self, batch_size):
        h0 = torch.zeros(self.num_beat_layers * 2, batch_size, self.beat_hidden_size).to(self.device)
        return (h0, h0)

    def init_measure_layer(self, batch_size):
        h0 = torch.zeros(self.num_measure_layers * 2, batch_size, self.measure_hidden_size).to(self.device)
        return (h0, h0)

    def init_beat_tempo_forward(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.beat_hidden_size).to(self.device)
        return (h0, h0)

    def init_voice_layer(self, batch_size, max_voice):
        layers = []
        for i in range(max_voice):
            # h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(device)
            h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.note_hidden_size).to(self.device)
            layers.append((h0, h0))
        return layers

    def init_onset_encoder(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.onset_hidden_size).to(self.device)
        return (h0, h0)



class TrillRNN(nn.Module):
    def __init__(self, network_parameters, trill_index):
        super(TrillRNN, self).__init__()
        self.hidden_size = network_parameters.note.size
        self.num_layers = network_parameters.note.layer
        self.input_size = network_parameters.input_size
        self.output_size = network_parameters.output_size
        self.is_trill_index = trill_index

        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        # self.fc = nn.Linear(hidden_size * 2, num_output)  # 2 for bidirection
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, note_hidden):
        # hidden = self.init_hidden(x.size(0))
        # hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        note_hidden = torch.nn.Parameter(note_hidden, requires_grad=False)
        is_trill_mat = x[:, :, self.is_trill_index]
        is_trill_mat = is_trill_mat.view(1,-1,1).repeat(1,1,self.output_size).view(1,-1,self.output_size)
        is_trill_mat = Variable(is_trill_mat, requires_grad=False)
        out = self.fc(note_hidden)
        up_trill = self.sigmoid(out[:,:,-1])
        out[:,:,-1] = up_trill
        out = out * is_trill_mat
        return out

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, h0)
