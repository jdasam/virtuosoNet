import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.modules.dropout import Dropout
from .model_utils import make_higher_node, span_beat_to_note_num
from .module import GatedGraph, SimpleAttention, ContextAttention, GatedGraphX, GatedGraphXBias, GraphConvStack


class IsgnEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnEncoder, self).__init__() 
        self.note_hidden_size = net_params.note.size
        self.measure_hidden_size = net_params.measure.size
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.Tanh(),
        )
        self.graph_1st = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, net_params.num_edge_types, secondary_size=self.note_hidden_size)
        self.graph_between = nn.Sequential(
            nn.Linear(self.note_hidden_size + self.measure_hidden_size * 2, self.note_hidden_size + self.measure_hidden_size * 2),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, net_params.num_edge_types, secondary_size=self.note_hidden_size)
        self.attention = ContextAttention(self.note_hidden_size * 2, net_params.num_attention_head)
        self.lstm =  nn.LSTM(self.note_hidden_size * 2, self.measure_hidden_size, net_params.measure.layer, batch_first=True, bidirectional=True)
    
    def run_iteration(self, notes_hidden, adjacency_matrix, note_locations):
        measure_numbers = note_locations['measure']
        for i in range(self.num_sequence_iteration):
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

        final_out = torch.cat((notes_hidden, notes_hidden_second[:,:, -self.note_hidden_size:]),-1)
        return {'total_note_cat': final_out, 
                'note': torch.cat([notes_hidden[:,:, -self.note_hidden_size:], notes_hidden_second[:,:,-self.note_hidden_size:]], dim=-1),
                'beat': None,
                'measure':measure_hidden, 
                'beat_spanned':None, 
                'measure_spanned': measure_hidden_spanned}


    def forward(self, nodes, adjacency_matrix, note_locations):
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(nodes.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)

        out_dict = self.run_iteration(notes_hidden, adjacency_matrix, note_locations)
        return out_dict
        # return final_out, measure_hidden
        


class IsgnResEncoder(IsgnEncoder):
    def __init__(self, net_params):
        super(IsgnResEncoder, self).__init__(IsgnResEncoder)
        self.graph_3rd = GatedGraph(self.note_hidden_size + self.measure_hidden_size * 2, net_params.num_edge_types)
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

    def run_iteration(self, notes_hidden, adjacency_matrix, note_locations):
        measure_numbers = note_locations['measure']
        for i in range(self.num_sequence_iteration):
            notes_hidden_1 = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_2 = self.graph_2nd(notes_hidden_1, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_3 = self.graph_3rd(notes_hidden_2, adjacency_matrix, iteration=self.num_graph_iteration)
            measure_nodes = make_higher_node(notes_hidden_3, self.attention, measure_numbers, measure_numbers,
                                                lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden = self.measure_expander(measure_hidden)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_hidden_3 = notes_hidden_3 + notes_hidden_1 + notes_hidden_2 + measure_hidden_spanned

            notes_hidden = self.encoder_fc(notes_hidden_3)


class IsgnResEncoderV2(IsgnResEncoder):
    def __init__(self, net_params):
        super(IsgnResEncoderV2, self).__init__(net_params)

    def run_iteration(self, notes_hidden, adjacency_matrix, note_locations):
        measure_numbers = note_locations['measure']
        notes_dense_hidden = notes_hidden[:,:,-self.note_hidden_size:]
        for i in range(self.num_sequence_iteration):
            notes_hidden_1 = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_2 = self.graph_2nd(notes_hidden_1, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_3 = self.graph_3rd(notes_hidden_2, adjacency_matrix, iteration=self.num_graph_iteration)
            measure_nodes = make_higher_node(notes_hidden_3, self.attention, measure_numbers, measure_numbers,
                                                lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden = self.measure_expander(measure_hidden)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            # notes_hidden_3 = notes_hidden_3 + notes_hidden_1 + notes_hidden_2 + measure_hidden_spanned
            notes_hidden = notes_hidden_1 + notes_hidden_2 + notes_hidden_3
            notes_hidden = torch.cat( [notes_hidden[:,:,:self.measure_hidden_size*2] + measure_hidden_spanned, 
                                       notes_hidden[:,:,self.measure_hidden_size*2:] + notes_dense_hidden], dim=-1)

            notes_hidden = self.encoder_fc(notes_hidden_3)
'''
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
'''

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

class IsgnOldEncoder(IsgnEncoder):
    def __init__(self, net_params):
        super(IsgnOldEncoder, self).__init__(net_params)


class IsgnBeatMeasEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnBeatMeasEncoder, self).__init__()
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        self.num_edge_types = net_params.num_edge_types
        self.num_attention_head = net_params.num_attention_head
        self.note_hidden_size = net_params.note.size

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, net_params.note.size),
            nn.Dropout(net_params.drop_out),
            nn.Tanh(),
        )
        self.graph_1st = GatedGraph(net_params.note.size + (net_params.measure.size + net_params.beat.size)* 2, self.num_edge_types, secondary_size=net_params.note.size)
        self.graph_between = nn.Sequential(
            nn.Linear(net_params.note.size + (net_params.measure.size + net_params.beat.size) * 2, net_params.note.size + (net_params.measure.size + net_params.beat.size) * 2),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(net_params.note.size + (net_params.measure.size + net_params.beat.size)* 2, self.num_edge_types, secondary_size=net_params.note.size)
        
        self.beat_attention = ContextAttention(net_params.note.size * 2, self.num_attention_head)
        self.beat_lstm =  nn.LSTM(net_params.note.size * 2, net_params.beat.size, net_params.beat.layer, batch_first=True, bidirectional=True)
        self.measure_attention = ContextAttention(net_params.beat.size * 2, self.num_attention_head)
        self.measure_lstm =  nn.LSTM(net_params.beat.size * 2, net_params.measure.size, net_params.measure.layer, batch_first=True, bidirectional=True)
    
    def forward(self, nodes, adjacency_matrix, note_locations):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), (self.beat_lstm.hidden_size + self.measure_lstm.hidden_size) * 2)).to(nodes.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)
        for i in range(self.num_sequence_iteration):
            notes_hidden = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden)
            notes_hidden_second = self.graph_2nd(notes_between, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_cat = torch.cat((notes_hidden[:,:, -self.note_hidden_size:],
                                          notes_hidden_second[:,:, -self.note_hidden_size:]), -1)
            beat_nodes = make_higher_node(notes_hidden_cat, self.beat_attention, beat_numbers, beat_numbers,
                                                  lower_is_note=True)
            beat_hidden, _ = self.beat_lstm(beat_nodes)
            measure_nodes = make_higher_node(beat_hidden, self.measure_attention, beat_numbers, measure_numbers,
                                                  lower_is_note=False)
            measure_hidden, _ = self.measure_lstm(measure_nodes)
            beat_hidden_spanned = span_beat_to_note_num(beat_hidden, beat_numbers)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_hidden = torch.cat((beat_hidden_spanned, measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_hidden, notes_hidden_second[:,:, -self.note_hidden_size:]),-1)
        return {'total_note_cat': final_out, 
        'note': torch.cat([notes_hidden[:,:, -self.note_hidden_size:], notes_hidden_second[:,:,-self.note_hidden_size:]], dim=-1),
        'beat': beat_hidden,
        'measure':measure_hidden, 
        'beat_spanned':beat_hidden_spanned, 
        'measure_spanned': measure_hidden_spanned}
        # return final_out, (beat_hidden, measure_hidden)

class IsgnBeatMeasNewEncoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnBeatMeasNewEncoder, self).__init__()
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        self.num_edge_types = net_params.num_edge_types
        self.num_attention_head = net_params.num_attention_head
        self.note_hidden_size = net_params.note.size

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, net_params.note.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.graph_1st = GatedGraph((net_params.note.size + net_params.measure.size + net_params.beat.size)* 2, self.num_edge_types, secondary_size=net_params.note.size)
        self.graph_between = nn.Sequential(
            nn.Linear(net_params.note.size, net_params.note.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.graph_2nd = GatedGraph(net_params.note.size * 2, self.num_edge_types, secondary_size=net_params.note.size)
        
        self.beat_attention = ContextAttention(net_params.note.size, self.num_attention_head)
        self.beat_lstm =  nn.LSTM(net_params.note.size, net_params.beat.size, net_params.beat.layer, batch_first=True, bidirectional=True)
        self.measure_attention = ContextAttention(net_params.beat.size * 2, self.num_attention_head)
        self.measure_lstm =  nn.LSTM(net_params.beat.size * 2, net_params.measure.size, net_params.measure.layer, batch_first=True, bidirectional=True)
    
    def forward(self, nodes, adjacency_matrix, note_locations):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        initial_input = self.note_fc(nodes)
        initial_hidden = torch.zeros((initial_input.size(0), initial_input.size(1), (self.beat_lstm.hidden_size + self.measure_lstm.hidden_size) * 2 + self.note_hidden_size)).to(nodes.device)
        notes_hidden = torch.cat((initial_input, initial_hidden), 2)
        notes_hidden_second = torch.zeros((initial_input.size(0), initial_input.size(1), self.note_hidden_size * 2)).to(nodes.device)
        for i in range(self.num_sequence_iteration):
            notes_hidden = self.graph_1st(notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_between = self.graph_between(notes_hidden[:,:,-self.note_hidden_size:])
            notes_hidden_second = torch.cat([notes_between, notes_hidden_second[:,:,-self.note_hidden_size:]], -1)
            notes_hidden_second = self.graph_2nd(notes_hidden_second, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_final_hidden = notes_hidden_second[:,:,-self.note_hidden_size:]
            beat_nodes = make_higher_node(notes_final_hidden, self.beat_attention, beat_numbers, beat_numbers,
                                                  lower_is_note=True)
            beat_hidden, _ = self.beat_lstm(beat_nodes)
            measure_nodes = make_higher_node(beat_hidden, self.measure_attention, beat_numbers, measure_numbers,
                                                  lower_is_note=False)
            measure_hidden, _ = self.measure_lstm(measure_nodes)
            beat_hidden_spanned = span_beat_to_note_num(beat_hidden, beat_numbers)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_hidden = torch.cat((initial_input, beat_hidden_spanned, measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)

        final_out = torch.cat((notes_final_hidden, beat_hidden_spanned, measure_hidden_spanned),-1)
        # return final_out, (beat_hidden, measure_hidden)
        return {'total_note_cat': final_out, 'note':notes_final_hidden, 'beat':beat_hidden, 'measure':measure_hidden, 'beat_spanned':beat_hidden_spanned, 'measure_spanned': measure_hidden_spanned}


class IsgnBeatMeasNewEncoderX(nn.Module):
    def __init__(self, net_params):
        super(IsgnBeatMeasNewEncoderX, self).__init__()
        self.num_graph_iteration = net_params.graph_iteration
        self.num_sequence_iteration = net_params.sequence_iteration
        self.num_edge_types = net_params.num_edge_types
        self.num_attention_head = net_params.num_attention_head
        self.note_hidden_size = net_params.note.size

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, net_params.note.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.graph_1st = GatedGraphX(net_params.note.size + (net_params.measure.size + net_params.beat.size)* 2, net_params.note.size, self.num_edge_types)
        self.graph_2nd = GatedGraphX(net_params.note.size, net_params.note.size, self.num_edge_types)
        
        self.beat_attention = ContextAttention(net_params.note.size, self.num_attention_head)
        self.beat_lstm =  nn.LSTM(net_params.note.size, net_params.beat.size, net_params.beat.layer, batch_first=True, bidirectional=True)
        self.measure_attention = ContextAttention(net_params.beat.size * 2, self.num_attention_head)
        self.measure_lstm =  nn.LSTM(net_params.beat.size * 2, net_params.measure.size, net_params.measure.layer, batch_first=True, bidirectional=True)
    
    def forward(self, nodes, adjacency_matrix, note_locations):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        initial_input = self.note_fc(nodes)
        initial_hidden = torch.zeros((initial_input.size(0), initial_input.size(1), (self.beat_lstm.hidden_size + self.measure_lstm.hidden_size) * 2 )).to(nodes.device)
        notes_input = torch.cat((initial_input, initial_hidden), 2)
        notes_hidden = torch.zeros(nodes.shape[0], nodes.shape[1], self.note_hidden_size).to(nodes.device)
        notes_hidden_second = torch.zeros(nodes.shape[0], nodes.shape[1], self.note_hidden_size).to(nodes.device)

        for i in range(self.num_sequence_iteration):
            notes_hidden = self.graph_1st(notes_input, notes_hidden, adjacency_matrix, iteration=self.num_graph_iteration)
            notes_hidden_second = self.graph_2nd(notes_hidden, notes_hidden_second, adjacency_matrix, iteration=self.num_graph_iteration)
            beat_nodes = make_higher_node(notes_hidden_second, self.beat_attention, beat_numbers, beat_numbers,
                                                lower_is_note=True)
            beat_hidden, _ = self.beat_lstm(beat_nodes)
            measure_nodes = make_higher_node(beat_hidden, self.measure_attention, beat_numbers, measure_numbers,
                                                lower_is_note=False)
            measure_hidden, _ = self.measure_lstm(measure_nodes)
            beat_hidden_spanned = span_beat_to_note_num(beat_hidden, beat_numbers)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_input = torch.cat((initial_input, beat_hidden_spanned, measure_hidden_spanned),-1)

        final_out = torch.cat((notes_hidden_second, beat_hidden_spanned, measure_hidden_spanned),-1)
        # return final_out, (beat_hidden, measure_hidden)
        return {'total_note_cat': final_out, 'note':notes_hidden_second, 'beat':beat_hidden, 'measure':measure_hidden, 'beat_spanned':beat_hidden_spanned, 'measure_spanned': measure_hidden_spanned}


class IsgnBeatMeasNewEncoderXBias(IsgnBeatMeasNewEncoderX):
    def __init__(self, net_params):
        super(IsgnBeatMeasNewEncoderXBias, self).__init__(net_params)
        self.graph_1st = GatedGraphXBias(net_params.note.size + (net_params.measure.size + net_params.beat.size)* 2, net_params.note.size, self.num_edge_types)
        self.graph_2nd = GatedGraphXBias(net_params.note.size, net_params.note.size, self.num_edge_types)



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


class GcnEncoder(nn.Module):
    def __init__(self, net_params):
        super().__init__() 
        self.note_hidden_size = net_params.note.size
        self.measure_hidden_size = net_params.measure.size
        self.num_sequence_iteration = net_params.sequence_iteration
        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, self.note_hidden_size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.graph = GraphConvStack(net_params.note.size + net_params.measure.size*2, net_params.note.size, num_edge_style=net_params.num_edge_types, num_layers=net_params.note.layer, drop_out=net_params.drop_out)

        self.attention = ContextAttention(self.note_hidden_size, net_params.num_attention_head)
        self.lstm =  nn.LSTM(self.note_hidden_size, self.measure_hidden_size, net_params.measure.layer, batch_first=True, bidirectional=True)
    
    def run_iteration(self, notes_hidden, adjacency_matrix, note_locations):
        measure_numbers = note_locations['measure']
        for i in range(self.num_sequence_iteration):
            temp_notes_hidden = self.graph(notes_hidden, adjacency_matrix)
            measure_nodes = make_higher_node(temp_notes_hidden, self.attention, measure_numbers, measure_numbers,
                                                  lower_is_note=True)
            measure_hidden, _ = self.lstm(measure_nodes)
            measure_hidden_spanned = span_beat_to_note_num(measure_hidden, measure_numbers)
            notes_hidden = torch.cat((measure_hidden_spanned, notes_hidden[:,:,-self.note_hidden_size:]),-1)
        notes_hidden = torch.cat((measure_hidden_spanned, temp_notes_hidden),-1)
        return {'total_note_cat': notes_hidden, 
                'note': notes_hidden[:,:, -self.note_hidden_size:],
                'beat': None,
                'measure':measure_hidden, 
                'beat_spanned':None, 
                'measure_spanned': measure_hidden_spanned}


    def forward(self, nodes, adjacency_matrix, note_locations):
        notes_dense_hidden = self.note_fc(nodes)
        initial_measure = torch.zeros((notes_dense_hidden.size(0), notes_dense_hidden.size(1), self.measure_hidden_size * 2)).to(nodes.device)
        notes_hidden = torch.cat((initial_measure, notes_dense_hidden), 2)

        out_dict = self.run_iteration(notes_hidden, adjacency_matrix, note_locations)
        return out_dict
        # return final_out, measure_hidden

class HanEncoder(nn.Module):
    def __init__(self, net_params):
        super(HanEncoder, self).__init__()

        self.note_fc = nn.Sequential(
            nn.Linear(net_params.input_size, net_params.note.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(net_params.note.size, net_params.note.size, net_params.note.layer, batch_first=True, bidirectional=True, dropout=net_params.drop_out)

        self.voice_net = nn.LSTM(net_params.note.size, net_params.voice.size, net_params.voice.layer,
                                    batch_first=True, bidirectional=True, dropout=net_params.drop_out)
        self.beat_attention = ContextAttention((net_params.note.size + net_params.voice.size) * 2,
                                                net_params.num_attention_head)
        self.beat_rnn = nn.LSTM((net_params.note.size + net_params.voice.size) * 2, net_params.beat.size, net_params.beat.layer, batch_first=True, bidirectional=True, dropout=net_params.drop_out)
        self.measure_attention = ContextAttention(net_params.beat.size*2, net_params.num_attention_head)
        self.measure_rnn = nn.LSTM(net_params.beat.size * 2, net_params.measure.size, net_params.measure.layer, batch_first=True, bidirectional=True)

    
    def forward(self, x, edges, note_locations):
        voice_numbers = note_locations['voice']
        x = self.note_fc(x)

        x = nn.utils.rnn.pack_padded_sequence(x, x.shape[1] - (voice_numbers==0).sum(dim=1), True, False )

        max_voice = torch.max(voice_numbers)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out, _ = pad_packed_sequence(hidden_out, True)
        hidden_out = torch.cat((hidden_out,voice_out), 2)

        beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned = self.run_beat_and_measure(hidden_out, note_locations)
        # return hidden_out, beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned
        return {'note': hidden_out, 'beat': beat_hidden_out, 'measure': measure_hidden_out, 'beat_spanned':beat_out_spanned, 'measure_spanned':measure_out_spanned,
                'total_note_cat': torch.cat([hidden_out, beat_out_spanned, measure_out_spanned], dim=-1) }

    def run_voice_net(self, batch_x, voice_numbers, max_voice):
        if isinstance(batch_x, torch.nn.utils.rnn.PackedSequence):
          batch_x, _ = nn.utils.rnn.pad_packed_sequence(batch_x, True)
        num_notes = batch_x.size(1)
        output = torch.zeros(batch_x.shape[0], batch_x.shape[1], self.voice_net.hidden_size * 2).to(batch_x.device)
        # voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1,max_voice+1):
          voice_x_bool = voice_numbers == i
          num_voice_notes = torch.sum(voice_x_bool)
          num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)

          if num_voice_notes > 0:
            voice_notes = [batch_x[i, voice_x_bool[i]] if torch.sum(voice_x_bool[i])>0 else torch.zeros(1,batch_x.shape[-1]).to(batch_x.device) for i in range(len(batch_x)) ]
            voice_x = pad_sequence(voice_notes, True)
            pack_voice_x = pack_padded_sequence(voice_x, [len(x) for x in voice_notes], True, False)
            ith_voice_out, _ = self.voice_net(pack_voice_x)
            ith_voice_out, _ = pad_packed_sequence(ith_voice_out, True)
            
            span_mat = torch.zeros(batch_x.shape[0], num_notes, voice_x.shape[1])
            voice_where = torch.nonzero(voice_x_bool)
            span_mat[voice_where[:,0], voice_where[:,1], torch.cat([torch.arange(num_batch_voice_notes[i]) for i in range(len(batch_x))])] = 1

            output += torch.bmm(span_mat, ith_voice_out)
        return output

    def run_beat_and_measure(self, hidden_out, note_locations):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        beat_nodes = make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, lower_is_note=True)
        beat_nodes = pack_padded_sequence(beat_nodes, beat_nodes.shape[1] - (beat_nodes.sum(-1)==0).sum(dim=1), True, False )
        beat_hidden_out, _ = self.beat_rnn(beat_nodes)
        beat_hidden_out, _ = pad_packed_sequence(beat_hidden_out, True)
        measure_nodes = make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers)
        measure_hidden_out, _ = self.measure_rnn(measure_nodes)

        beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers)
        measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers)

        return beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned

    def get_attention_weights(self, x, edges, note_locations):
        voice_numbers = note_locations['voice']
        x = self.note_fc(x)
        max_voice = max(voice_numbers)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out = torch.cat((hidden_out,voice_out), 2)
        beat_hidden_out, _, _, _ = self.run_beat_and_measure(hidden_out, note_locations)

        weights = self.beat_attention.get_attention(hidden_out).squeeze()
        weights_mean = torch.mean(weights, axis=1).unsqueeze(1).repeat(1,weights.shape[1])
        weights_std = torch.std(weights, axis=1).unsqueeze(1).repeat(1,weights.shape[1])

        beat_weights = self.measure_attention.get_attention(beat_hidden_out).squeeze()
        beat_weights_mean = torch.mean(beat_weights, axis=1).unsqueeze(1).repeat(1,beat_weights.shape[1])
        beat_weights_std = torch.std(beat_weights, axis=1).unsqueeze(1).repeat(1,beat_weights.shape[1])

        norm_weights =  (weights-weights_mean)/weights_std
        norm_beat_weights = (beat_weights-beat_weights_mean)/beat_weights_std
        return {'note':norm_weights.permute(1,0).cpu().numpy(), 'beat':norm_beat_weights.permute(1,0).cpu().numpy()}

    

class HanGraphEncoder(HanEncoder):
    def __init__(self, net_params):
        super(HanGraphEncoder, self).__init__(net_params)
        self.note_hidden_summarize = nn.Sequential(
            nn.Linear((net_params.note.size + net_params.voice.size) * 2, net_params.graph_note.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.note_graph = GatedGraph(net_params.graph_note.size * 2, self.num_edge_types, secondary_size=self.graph_note.size)
        self.num_graph_iteration = net_params.num_graph_iteration

        self.beat_attention = ContextAttention( (net_params.note.size + net_params.voice.size) * 2 + net_params.graph_note.size,
                                                net_params.num_attention_head)
        self.beat_rnn = nn.LSTM((net_params.note.size + net_params.voice.size) * 2 + net_params.graph_note.size, net_params.beat.size, net_params.beat.layer, batch_first=True, bidirectional=True, dropout=net_params.drop_out)


    def forward(self, x, edges, note_locations):
        voice_numbers = note_locations['voice']

        x = self.note_fc(x)
        max_voice = max(voice_numbers)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out = torch.cat((hidden_out,voice_out), dim=-1)

        graph_input = self.note_hidden_summarize(hidden_out)
        graph_output = self.note_graph(graph_input, iteration=self.num_graph_iteration)

        hidden_out = torch.cat([hidden_out, graph_output], dim=-1)

        beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned = self.run_beat_and_measure(hidden_out, note_locations)

        return {'note': hidden_out, 'beat': beat_hidden_out, 'measure': measure_hidden_out, 'beat_spanned':beat_out_spanned, 'measure_spanned':measure_out_spanned,
                'total_note_cat': torch.cat([hidden_out, beat_out_spanned, measure_out_spanned], dim=-1) }


class GraphHanEncoder(HanEncoder):
    def __init__(self, net_params):
        super(GraphHanEncoder, self).__init__(net_params)
        self.note_hidden_size = net_params.note.size
        self.note_graph = GatedGraph(net_params.note.size * 2, net_params.num_edge_types, secondary_size=net_params.note.size)
        self.num_graph_iteration = net_params.graph_iteration

        self.beat_attention = ContextAttention( (net_params.note.size + net_params.voice.size) * 2,
                                                net_params.num_attention_head)
        self.beat_rnn = nn.LSTM((net_params.note.size + net_params.voice.size) * 2 , net_params.beat.size, net_params.beat.layer, batch_first=True, bidirectional=True, dropout=net_params.drop_out)


    def forward(self, x, edges, note_locations):
        voice_numbers = note_locations['voice']

        x = self.note_fc(x)
        max_voice = max(voice_numbers)
        initial_hidden = torch.zeros_like(x)
        x = self.note_graph(torch.cat([x, initial_hidden], dim=-1), edges, iteration=self.num_graph_iteration)[:,:,-self.note_hidden_size:]

        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out = torch.cat((hidden_out,voice_out), dim=-1)
        beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned = self.run_beat_and_measure(hidden_out, note_locations)

        return {'note': hidden_out, 'beat': beat_hidden_out, 'measure': measure_hidden_out, 'beat_spanned':beat_out_spanned, 'measure_spanned':measure_out_spanned,
                'total_note_cat': torch.cat([hidden_out, beat_out_spanned, measure_out_spanned], dim=-1) }
