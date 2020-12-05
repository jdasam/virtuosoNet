import torch
import torch.nn as nn
from .model_utils import make_higher_node, reparameterize, span_beat_to_note_num
from .module import GatedGraph, SimpleAttention, ContextAttention


class IsgnResEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnResEncoder, self).__init__()
        net_params.input_size = net_params.input_size
        self.num_layers = net_params.note.layer
        self.note_hidden_size = net_params.note.size
        self.num_measure_layers = net_params.measure.layer
        self.measure_hidden_size = net_params.measure.size
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        self.num_edge_types = net_params.num_edge_types
        self.num_attention_head = net_params.num_attention_head

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(self.note_hidden_size, self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.Tanh(),
        )
        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types)
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types)
        self.graph_3rd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types)

        self.measure_attention = ContextAttention(self.note_hidden_size + self.measure_hidden_size * 2, self.num_attention_head)
        self.measure_expander = nn.Sequential(
            nn.Linear(self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(net_params.drop_out),
            nn.Tanh(),
        )
        self.attention = ContextAttention(self.note_hidden_size * 2, self.num_attention_head)
        self.lstm =  nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
        
    def forward(self, nodes, adjacency_matrix, note_locations):
        num_notes = nodes.shape[1]
        measure_numbers = note_locations['measure']

        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(nodes.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden_1 = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_2 = self.graph_2nd(notes_hidden_1, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden3 = self.graph_3rd(notes_hidden_2, adjacency_matrix, iteration=self.num_graph_iteration)
            # notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
            #                               notes_hidden_second[:,:, -self.note_hidden_size:]), -1)
            measure_nodes = make_higher_node(notes_hidden3, self.attention, measure_numbers, measure_numbers,
                                                lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden = self.measure_expander(measure_hidden)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            # notes_hidden3[:, :, :self.measure_hidden_size*2] = notes_hidden3[:, :, :self.measure_hidden_size*2] + measure_hidden_spanned
            notes_hidden3 = notes_hidden3 + notes_hidden_1 + notes_hidden_2 + measure_hidden_spanned

            notes_hidden = self.encoder_fc(notes_hidden3)
        return notes_hidden, measure_hidden

class IsgnResEncoderV2(IsgnResEncoder):
    def __init__(self, net_params):
        super(IsgnResEncoderV2, self).__init__(net_params)
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        
    def forward(self, nodes, adjacency_matrix, note_locations):
        measure_numbers = note_locations['measure']
        num_notes = nodes.shape[1]
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(nodes.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden_1 = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_2 = self.graph_2nd(notes_hidden_1, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_3 = self.graph_2nd(notes_hidden_2, adjacency_matrix, iteration=self.num_graph_iteration)
            # notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
            #                               notes_hidden_second[:,:, -self.note_hidden_size:]), -1)
            measure_nodes = make_higher_node(notes_hidden_3, self.attention, measure_numbers, measure_numbers,
                                                lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            # notes_hidden3[:, :, :self.measure_hidden_size*2] = notes_hidden3[:, :, :self.measure_hidden_size*2] + measure_hidden_spanned
            notes_hidden = notes_hidden_1 + notes_hidden_2 + notes_hidden_3
            notes_hidden = torch.cat( [notes_hidden[:,:,:self.measure_hidden_size*2] + measure_hidden_spanned, 
                                       notes_hidden[:,:,self.measure_hidden_size*2:] + notes_dense_hidden], dim=-1)

            notes_hidden = self.encoder_fc(notes_hidden)
        return notes_hidden, measure_hidden

class IsgnOldEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnOldEncoder, self).__init__()
        net_params.input_size = net_params.input_size
        self.note_hidden_size = net_params.note.size
        self.num_measure_layers = net_params.measure.layer
        self.measure_hidden_size = net_params.measure.size
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        net_params.drop_out = net_params.drop_out
        self.num_edge_types = net_params.num_edge_types
        self.num_attention_head = net_params.num_attention_head

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.Tanh(),
        )
        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, secondary_size=self.note_hidden_size)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(net_params.drop_out),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types, secondary_size=self.note_hidden_size)
        self.attention = ContextAttention(self.note_hidden_size * 2, self.num_attention_head)
        self.lstm =  nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
        
    def forward(self, nodes, adjacency_matrix, note_locations):
        measure_numbers = note_locations['measure']
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(nodes.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden)
            notes_hidden_second = self.graph_2nd(notes_between, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
                                          notes_hidden_second[:,:, -self.note_hidden_size:]), -1)

            measure_nodes = make_higher_node(notes_hidden_cat, self.attention, measure_numbers, measure_numbers,
                                                  lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_hidden = torch.cat((measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_hidden, notes_hidden_second),-1)
        return final_out, measure_hidden

class IsgnOldGraphSingleEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnOldGraphSingleEncoder, self).__init__()
        net_params.input_size = net_params.input_size
        self.num_layers = net_params.note.layer
        self.note_hidden_size = net_params.note.size
        self.num_measure_layers = net_params.measure.layer
        self.measure_hidden_size = net_params.measure.size
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        net_params.drop_out = net_params.drop_out
        self.num_edge_types = net_params.num_edge_types
        self.num_attention_head = net_params.num_attention_head

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.note_hidden_size),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(net_params.drop_out),
            # nn.BatchNorm1d(self.note_hidden_size),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, self.num_edge_types)
        self.attention = ContextAttention(self.note_hidden_size * 2, self.num_attention_head)
        self.lstm =  nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
        
    def forward(self, nodes, adjacency_matrix, measure_numbers):
        num_notes = nodes.shape[1]
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(nodes.device)
        note_measure_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
        # for i in range(3):
            notes_hidden = self.graph_1st(note_measure_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden)
            notes_hidden_second = self.graph_2nd(notes_between, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
                                          notes_hidden_second[:,:, -self.note_hidden_size:]), -1)

            measure_nodes = make_higher_node(notes_hidden_cat, self.attention, measure_numbers, measure_numbers,
                                                  lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_hidden = torch.cat((measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_hidden, notes_hidden_second),-1)
        return final_out, measure_hidden




class HanEncoder(nn.Module):
    def __init__(self, net_params):
        super(HanEncoder, self).__init__()

        net_params.input_size = net_params.input_size
        self.num_layers = net_params.note.layer
        self.hidden_size = net_params.note.size
        self.num_beat_layers = net_params.beat.layer
        self.beat_hidden_size = net_params.beat.size
        self.num_measure_layers = net_params.measure.layer
        self.measure_hidden_size = net_params.measure.size

        self.num_voice_layers = net_params.voice.layer
        self.voice_hidden_size = net_params.voice.size
        self.num_attention_head = net_params.num_attention_head

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=net_params.drop_out)

        self.voice_net = nn.LSTM(self.hidden_size, self.voice_hidden_size, self.num_voice_layers,
                                    batch_first=True, bidirectional=True, dropout=net_params.drop_out)
        self.beat_attention = ContextAttention((self.hidden_size + self.voice_hidden_size) * 2,
                                                self.num_attention_head)
        self.beat_rnn = nn.LSTM((self.hidden_size + self.voice_hidden_size) * 2, self.beat_hidden_size, self.num_beat_layers, batch_first=True, bidirectional=True, dropout=net_params.drop_out)
        self.measure_attention = ContextAttention(self.beat_hidden_size*2, self.num_attention_head)
        self.measure_rnn = nn.LSTM(self.beat_hidden_size * 2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
    
    
    def forward(self, x, edges, note_locations):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        voice_numbers = note_locations['voice']

        # hidden = self.init_hidden(self.num_layers, 2, x.size(0), self.hidden_size)
        # beat_hidden = self.init_hidden(self.num_beat_layers, 2, x.size(0), self.beat_hidden_size)
        # measure_hidden = self.init_hidden(self.num_measure_layers, 2, x.size(0), self.measure_hidden_size)

        x = self.note_fc(x)
        max_voice = max(voice_numbers)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out = torch.cat((hidden_out,voice_out), 2)

        beat_nodes = make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, lower_is_note=True)
        beat_hidden_out, _ = self.beat_rnn(beat_nodes)
        measure_nodes = make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers)
        measure_hidden_out, _ = self.measure_rnn(measure_nodes)

        beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers)
        measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers)

        return hidden_out, beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned


    def run_voice_net(self, batch_x, voice_numbers, max_voice):
        num_notes = batch_x.size(1)
        output = torch.zeros(batch_x.shape[0], batch_x.shape[1], self.voice_hidden_size * 2).to(batch_x.device)
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
                span_mat = span_mat.view(1,num_notes,-1).repeat(batch_x.shape[0],1,1).to(batch_x.device)
                voice_x = batch_x[:,voice_x_bool,:]
                # voice_x = batch_x[0,voice_x_bool,:].view(1,-1, self.hidden_size)
                # ith_hidden = voice_hidden[i-1]

                ith_voice_out, _ = self.voice_net(voice_x)
                # ith_voice_out, ith_hidden = self.lstm(voice_x, ith_hidden)
                output += torch.bmm(span_mat, ith_voice_out)
        return output