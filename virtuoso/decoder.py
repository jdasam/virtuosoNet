import torch
import torch.nn as nn
from .model_utils import make_higher_node, reparameterize, run_hierarchy_lstm_with_pack, span_beat_to_note_num, get_beat_corresp_out
from .utils import note_feature_to_beat_mean, get_is_padded_for_sequence
from .module import GatedGraph, GraphConv, SimpleAttention, ContextAttention, GatedGraphX, GatedGraphXBias, GraphConvStack
from .model_constants import QPM_INDEX, QPM_PRIMO_IDX

class IsgnDecoder(nn.Module):
    def __init__(self, net_params):
        super(IsgnDecoder, self).__init__()
        self.output_size = net_params.output_size
        self.num_sequence_iteration = net_params.sequence_iteration
        self.num_graph_iteration = net_params.graph_iteration
        self.final_graph_margin_size = net_params.final.margin
        
        self.style_vector_expandor = nn.Sequential(
            nn.Linear(net_params.encoded_vector_size, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.initial_result_fc = nn.Sequential(
            nn.Linear(net_params.final.input
            - net_params.encoder.size
            - net_params.time_reg.size * 2
            - net_params.output_size
            - net_params.final.margin, net_params.encoder.size ),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.encoder.size, net_params.output_size),
        )
        self.final_graph = GatedGraph(net_params.final.input, net_params.num_edge_types,
                                      net_params.output_size + net_params.final.margin)

        self.tempo_rnn = nn.LSTM(net_params.final.margin + net_params.output_size + 8, net_params.time_reg.size,
                                    num_layers=net_params.time_reg.layer, batch_first=True, bidirectional=True)

        self.final_beat_attention = ContextAttention(net_params.output_size, 1)
        self.final_margin_attention = ContextAttention(net_params.final.margin, net_params.num_attention_head)
        self.tempo_fc = nn.Linear(net_params.time_reg.size * 2, 1)
        

        self.fc = nn.Sequential(
            nn.Linear(net_params.final.input, net_params.final.margin),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.final.margin, net_params.output_size-1),
        )
    def handle_perform_z(self, score_embedding, perf_embedding, edges, note_locations):
        return

    def concat_tempo_rnn_input(self, out_in_beat, margin_in_beat, res_info):
        return torch.cat((out_in_beat, margin_in_beat, res_info), 2)
    
    def run_final_isgn(self, note_out, perf_embedding, res_info, edges, note_locations):
        num_notes = note_out.shape[1]
        beat_numbers = note_locations['beat']

        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.tempo_rnn.hidden_size * 2)).to(note_out.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(note_out.device)

        initial_output = self.initial_result_fc(note_out)
        total_iterated_output = [initial_output]

        out_with_result = torch.cat(
            (note_out, perf_embedding, initial_beat_hidden, initial_output, initial_margin), 2)
            # (note_out, perform_z_batched, initial_beat_hidden, initial_output, initial_margin), 2)

        for i in range(self.num_sequence_iteration):
            out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)

            initial_out = out_with_result[:, :,
                            -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
            changed_margin = out_with_result[:, :, -self.final_graph_margin_size:]

            margin_in_beat = make_higher_node(changed_margin, self.final_margin_attention, beat_numbers,
                                                        beat_numbers, lower_is_note=True)
            out_in_beat = make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                    beat_numbers, lower_is_note=True)
            out_beat_cat = self.concat_tempo_rnn_input(out_in_beat, margin_in_beat, res_info)
            out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
            tempo_out = self.tempo_fc(out_beat_rnn_result)

            tempos_spanned = span_beat_to_note_num(tempo_out, beat_numbers)
            out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, beat_numbers)

            out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                            out_beat_spanned,
                                            out_with_result[:, :, -self.output_size - self.final_graph_margin_size:]),
                                        2)
            other_out = self.fc(out_with_result)

            final_out = torch.cat((tempos_spanned, other_out), 2)
            out_with_result = torch.cat((out_with_result[:, :, :-self.output_size - self.final_graph_margin_size],
                                            final_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
            total_iterated_output.append(final_out)
        return final_out, total_iterated_output
    
    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        perform_z = self.handle_perform_z(score_embedding, perf_embedding, edges, note_locations)
        final_out, total_iterated_output = self.run_final_isgn(score_embedding, perform_z, res_info, edges, note_locations)
        
        return final_out, total_iterated_output


class IsgnNoteDecoder(IsgnDecoder):
    def __init__(self, net_params):
        super(IsgnNoteDecoder, self).__init__(net_params)
        self.final_beat_hidden_idx = net_params.note.size * 2 + net_params.measure.size * 2 + net_params.encoder.size

        self.perform_style_to_measure = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size, net_params.encoder.size, num_layers=1, bidirectional=False)
        self.measure_perf_fc = nn.Linear(net_params.encoder.size, net_params.encoder.size)
        
    def handle_perform_z(self, score_embedding, perf_embedding, edges, note_locations):
        # _, measure_out = score_embedding
        measure_out = score_embedding['measure']
        measure_numbers = note_locations['measure']
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        
        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_out), 2)
        measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        measure_perform_style = self.measure_perf_fc(measure_perform_style)
        measure_perform_style_spanned = span_beat_to_note_num(measure_perform_style, measure_numbers)

        return measure_perform_style_spanned

class IsgnMeasNoteDecoder(IsgnDecoder):
    def __init__(self, net_params):
        super(IsgnMeasNoteDecoder, self).__init__(net_params)
        self.perform_style_to_measure = None
        self.measure_perf_fc = None
        self.measure_out_lstm = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size + 2, net_params.measure.size, bidirectional=True, batch_first=True)
        self.measure_out_fc = nn.Linear(net_params.measure.size * 2, 2)
        self.final_beat_hidden_idx = net_params.note.size * 2 + net_params.measure.size * 2 + net_params.encoder.size + 2
        
    def handle_perform_z(self, score_embedding, perf_embedding, edges, note_locations):
        # note_out, _ = score_embedding
        perform_z = self.style_vector_expandor(perf_embedding)
        return perform_z

    def get_measure_level_output(self, score_embedding, perf_embedding, res_info, note_locations):
        # _, measure_out = score_embedding
        measure_out = score_embedding['measure']
        num_measures = measure_out.shape[1]
        measure_numbers = note_locations['measure']
        perf_emb_reshpaed = perf_embedding.unsqueeze(1)

        measure_hidden = (torch.zeros(2*self.measure_out_lstm.num_layers, 1, measure_out.shape[-1]//2).to(measure_out.device), 
                          torch.zeros(2*self.measure_out_lstm.num_layers, 1, measure_out.shape[-1]//2).to(measure_out.device))
        prev_out = torch.zeros(measure_out.shape[0], 1, 2).to(measure_out.device)
        measure_tempo_vel = torch.zeros(measure_out.shape[0], num_measures, 2).to(measure_out.device)
        for i in range(num_measures):
            cur_input = torch.cat([perf_emb_reshpaed, measure_out[:,i:i+1], prev_out], dim=-1)
            cur_tempo_vel, measure_hidden = self.measure_out_lstm(cur_input, measure_hidden)
            measure_tempo_vel[:,i:i+1,:] = self.measure_out_fc(cur_tempo_vel)
            prev_out = measure_tempo_vel[:,i:i+1,:]

        measure_tempo_vel_broadcasted = span_beat_to_note_num(measure_tempo_vel, measure_numbers)
        return measure_tempo_vel_broadcasted, measure_tempo_vel
        
    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        # note_out, measure_out = score_embedding
        note_out = score_embedding['total_note_cat']
        num_notes = note_out.shape[1]

        perform_z = self.handle_perform_z(score_embedding, perf_embedding, edges, note_locations)
        measure_tempo_vel_broadcasted, measure_tempo_vel = self.get_measure_level_output(score_embedding, perform_z, res_info, note_locations)
        
        note_out = torch.cat([note_out, measure_tempo_vel_broadcasted], dim=-1)
        final_out, total_iterated_output = self.run_final_isgn(note_out, perform_z.repeat(1,num_notes,1), res_info, edges, note_locations)
        return final_out, {'iter_out': total_iterated_output, 'meas_out':measure_tempo_vel}


class IsgnMeasNoteDecoderV2(IsgnMeasNoteDecoder):
    # connect res info to measure-level decoder
    def __init__(self, net_params):
        super(IsgnMeasNoteDecoderV2, self).__init__(net_params)
        self.measure_out_lstm = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size + 2 + 8, net_params.measure.size, bidirectional=True, batch_first=True)
        self.measure_out_fc = nn.Linear(net_params.measure.size * 2, 2)
        self.final_beat_hidden_idx = net_params.final.input - net_params.time_reg.size * 2 - net_params.output_size - net_params.final.margin
        self.initial_result_fc = nn.Sequential(
            nn.Linear(net_params.final.input
            - net_params.encoder.size
            - net_params.time_reg.size * 2
            - net_params.output_size
            - net_params.final.margin, net_params.encoder.size ),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.encoder.size, net_params.output_size),
        )
        self.tempo_rnn = nn.LSTM(net_params.final.margin + net_params.output_size, net_params.time_reg.size,
                            num_layers=net_params.time_reg.layer, batch_first=True, bidirectional=True)

    def init_measure_hidden(self, device):
        return (torch.zeros(2*self.measure_out_lstm.num_layers, 1, self.measure_out_lstm.hidden_size).to(device), 
                    torch.zeros(2*self.measure_out_lstm.num_layers, 1, self.measure_out_lstm.hidden_size).to(device))

    def concat_tempo_rnn_input(self, out_in_beat, margin_in_beat, res_info):
        return torch.cat((out_in_beat, margin_in_beat), 2)

    def handle_perform_z(self, score_embedding, perf_embedding, edges, note_locations):
        note_out = score_embedding['total_note_cat']
        measure_numbers = note_locations['measure']
        num_notes = note_out.shape[1]
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        perform_z = perform_z.repeat(num_notes, 1).unsqueeze(0)

        return perform_z, perform_z_measure_spanned

    def get_measure_level_output(self, score_embedding, perform_z_measure_spanned, res_info, note_locations):
        measure_out = score_embedding['measure']
        num_measures = measure_out.shape[1]
        measure_numbers = note_locations['measure']

        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_out, res_info), 2)
        measure_hidden = self.init_measure_hidden(measure_out.device)
        prev_out = torch.zeros(measure_out.shape[0], 1, 2).to(measure_out.device)
        measure_tempo_vel = torch.zeros(measure_out.shape[0], num_measures, 2).to(measure_out.device)
        for i in range(num_measures):
            cur_input = torch.cat([perform_z_measure_cat[:,i:i+1,:], prev_out], dim=-1)
            cur_tempo_vel, measure_hidden =  self.measure_out_lstm(cur_input, measure_hidden)
            measure_tempo_vel[:,i:i+1,:] = self.measure_out_fc(cur_tempo_vel)
            prev_out = measure_tempo_vel[:,i:i+1,:]

        measure_tempo_vel_broadcasted = span_beat_to_note_num(measure_tempo_vel, measure_numbers)

        return measure_tempo_vel, measure_tempo_vel_broadcasted

        
    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        # note_out, measure_out = score_embedding
        note_out = score_embedding['total_note_cat']

        perform_z, perform_z_measure_spanned = self.handle_perform_z(score_embedding, perf_embedding, edges, note_locations)
        measure_tempo_vel, measure_tempo_vel_broadcasted = self.get_measure_level_output(score_embedding, perform_z_measure_spanned, res_info, note_locations)
        note_out = torch.cat([note_out, measure_tempo_vel_broadcasted], dim=-1)
        final_out, total_iterated_output = self.run_final_isgn(note_out, perform_z, res_info, edges, note_locations)

        return final_out, {'iter_out': total_iterated_output, 'meas_out':measure_tempo_vel}


class IsgnMeasNoteDecoderV3(IsgnMeasNoteDecoderV2):
    def __init__(self, net_params):
        super(IsgnMeasNoteDecoderV3, self).__init__(net_params)
        self.perform_style_to_measure = None
        self.measure_perf_fc = None
        self.measure_out_lstm = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size + 2 + 8, net_params.measure.size, num_layers=net_params.measure.layer, batch_first=True)
        self.measure_out_fc = nn.Linear(net_params.measure.size, 2)

    def init_measure_hidden(self, batch_size,  device):
        return (torch.zeros(self.measure_out_lstm.num_layers, batch_size, self.measure_out_lstm.hidden_size).to(device), 
                    torch.zeros(self.measure_out_lstm.num_layers, batch_size, self.measure_out_lstm.hidden_size).to(device))



class IsgnBeatMeasDecoder(IsgnMeasNoteDecoderV3):
    def __init__(self, net_params):
        super(IsgnBeatMeasDecoder, self).__init__(net_params)
        self.beat_out_lstm = nn.LSTM(net_params.beat.size * 2 + net_params.encoder.size + 4, net_params.beat.size, num_layers=net_params.beat.layer, batch_first=True)
        self.beat_out_fc = nn.Linear(net_params.beat.size, 2)
        self.tempo_fc = None
        self.final_beat_hidden_idx += 1
        self.initial_result_fc = nn.Sequential(
            nn.Linear(net_params.final.input
            - net_params.encoder.size
            - net_params.time_reg.size * 2
            - net_params.output_size + 1
            - net_params.final.margin, net_params.encoder.size ),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.encoder.size, net_params.output_size - 1),
        )


    def init_beat_hidden(self, batch_size, device):
        return (torch.zeros(self.beat_out_lstm.num_layers, batch_size, self.beat_out_lstm.hidden_size).to(device), 
                    torch.zeros(self.beat_out_lstm.num_layers, batch_size, self.beat_out_lstm.hidden_size).to(device))

    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        # note_out, (beat_out, measure_out) = score_embedding
        note_out = score_embedding['total_note_cat']
        beat_out = score_embedding['beat']
        measure_out = score_embedding['measure']
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        n_batch = note_out.shape[0]
        num_notes = note_out.shape[1]
        # num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        num_measures = measure_out.shape[1]
        num_beats = beat_out.shape[1]

        is_padded_measure = get_is_padded_for_sequence(measure_out)
        is_padded_beat = get_is_padded_for_sequence(beat_out)


        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z_measure_spanned = perform_z.unsqueeze(1).repeat(1,num_measures, 1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_out, res_info), 2)

        measure_hidden = self.init_measure_hidden(n_batch, measure_out.device)
        prev_out = torch.zeros(measure_out.shape[0], 1, 2).to(perform_z.device)
        measure_tempo_vel = torch.zeros(measure_out.shape[0], num_measures, 2).to(note_out.device)
        for i in range(num_measures):
            cur_input = torch.cat([perform_z_measure_cat[:,i:i+1,:], prev_out], dim=-1)
            cur_tempo_vel, measure_hidden =  self.measure_out_lstm(cur_input, measure_hidden)
            measure_tempo_vel[:,i:i+1,:] = self.measure_out_fc(cur_tempo_vel)
            prev_out = measure_tempo_vel[:,i:i+1,:]
        measure_tempo_vel[is_padded_measure] = 0

        measure_tempo_vel_broadcasted = span_beat_to_note_num(measure_tempo_vel, measure_numbers)
        measure_tempo_vel_in_beat = note_feature_to_beat_mean(measure_tempo_vel_broadcasted, beat_numbers, use_mean=False)

        beat_hidden = self.init_beat_hidden(n_batch, beat_out.device)
        perform_z_beat_spanned = perform_z.unsqueeze(1).repeat(1,num_beats, 1)
        perform_z_beat_cat = torch.cat((perform_z_beat_spanned, beat_out, measure_tempo_vel_in_beat), 2)
        prev_out = torch.zeros(beat_out.shape[0], 1, 2).to(perform_z.device)
        beat_tempo_vel = torch.zeros(beat_out.shape[0], num_beats, 2).to(beat_out.device)
        for i in range(num_beats):
            cur_input = torch.cat([perform_z_beat_cat[:,i:i+1,:], prev_out], dim=-1)
            cur_tempo_vel, beat_hidden =  self.beat_out_lstm(cur_input, beat_hidden)
            beat_tempo_vel[:,i:i+1,:] = self.beat_out_fc(cur_tempo_vel)
            prev_out = beat_tempo_vel[:,i:i+1,:]
        beat_tempo_vel[is_padded_beat] = 0
        beat_tempo_vel_broadcasted = span_beat_to_note_num(beat_tempo_vel, beat_numbers)

        note_out = torch.cat([note_out, measure_tempo_vel_broadcasted, beat_tempo_vel_broadcasted], dim=-1)
        perform_z = perform_z.unsqueeze(1).repeat(1, num_notes, 1)

        initial_output = self.initial_result_fc(note_out)
        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.tempo_rnn.hidden_size * 2)).to(note_out.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(note_out.device)

        total_iterated_output = [torch.cat([beat_tempo_vel_broadcasted[:,:,0:1], initial_output], dim=-1) ]

        out_with_result = torch.cat(
            (note_out, perform_z, initial_beat_hidden, initial_output, initial_margin), 2)
            # (note_out, perform_z_batched, initial_beat_hidden, initial_output, initial_margin), 2)

        final_out, total_iterated_output = self.run_note_level_decoder(out_with_result, edges, beat_numbers, beat_tempo_vel_broadcasted, total_iterated_output)
        return final_out, {'iter_out': total_iterated_output, 'meas_out':measure_tempo_vel, 'beat_out':beat_tempo_vel}

    def run_note_level_decoder(self, out_with_result, edges, beat_numbers, beat_tempo_vel_broadcasted, total_iterated_output):
        for i in range(self.num_sequence_iteration):
            out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
            initial_out = out_with_result[:, :,
                            -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
            changed_margin = out_with_result[:, :, -self.final_graph_margin_size:]

            margin_in_beat = make_higher_node(changed_margin, self.final_margin_attention, beat_numbers,
                                                        beat_numbers, lower_is_note=True)
            out_in_beat = make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                    beat_numbers, lower_is_note=True)
            out_beat_cat = torch.cat((out_in_beat, margin_in_beat), 2)
            out_beat_rnn_result = run_hierarchy_lstm_with_pack(out_beat_cat, self.tempo_rnn)
            # out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
            out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, beat_numbers)

            out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                            out_beat_spanned,
                                            out_with_result[:, :, -self.output_size + 1 - self.final_graph_margin_size:]),
                                        2)
            other_out = self.fc(out_with_result)

            # final_out = torch.cat((tempos_spanned, other_out), 2)
            out_with_result = torch.cat((out_with_result[:, :, :-self.output_size  + 1 - self.final_graph_margin_size],
                                            other_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
            final_out = torch.cat((beat_tempo_vel_broadcasted[:,:,0:1], other_out), -1)
            total_iterated_output.append(final_out)
        return final_out, total_iterated_output


class IsgnBeatMeasNewDecoder(IsgnBeatMeasDecoder):
    def __init__(self, net_params):
        super(IsgnBeatMeasNewDecoder, self).__init__(net_params)
        self.fc = nn.Sequential(
            nn.Linear(net_params.final.margin + net_params.time_reg.size * 2, net_params.final.margin),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.final.margin, net_params.output_size-1),
        )

    def init_beat_hidden(self, device):
        return (torch.zeros(self.beat_out_lstm.num_layers, 1, self.beat_out_lstm.hidden_size).to(device), 
                    torch.zeros(self.beat_out_lstm.num_layers, 1, self.beat_out_lstm.hidden_size).to(device))


    def run_note_level_decoder(self, out_with_result, edges, beat_numbers, beat_tempo_vel_broadcasted, total_iterated_output):
        for i in range(self.num_sequence_iteration):
            out_with_result = self.final_graph(out_with_result, edges, iteration=self.num_graph_iteration)
            initial_out = out_with_result[:, :,
                            -self.output_size - self.final_graph_margin_size: -self.final_graph_margin_size]
            changed_margin = out_with_result[:, :, -self.final_graph_margin_size:]

            margin_in_beat = make_higher_node(changed_margin, self.final_margin_attention, beat_numbers,
                                                        beat_numbers, lower_is_note=True)
            out_in_beat = make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                    beat_numbers, lower_is_note=True)
            out_beat_cat = torch.cat((out_in_beat, margin_in_beat), 2)
            out_beat_rnn_result = run_hierarchy_lstm_with_pack(out_beat_cat, self.tempo_rnn)
            # out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
            out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, beat_numbers)

            out_with_result = torch.cat((out_with_result[:, :, :self.final_beat_hidden_idx],
                                            out_beat_spanned,
                                            out_with_result[:, :, -self.output_size + 1 - self.final_graph_margin_size:]),
                                        2)
            other_out = self.fc(torch.cat([out_beat_spanned, out_with_result[:,:,-self.final_graph_margin_size:]], dim=-1))

            # final_out = torch.cat((tempos_spanned, other_out), 2)
            out_with_result = torch.cat((out_with_result[:, :, :-self.output_size  + 1 - self.final_graph_margin_size],
                                            other_out, out_with_result[:, :, -self.final_graph_margin_size:]), 2)
            final_out = torch.cat((beat_tempo_vel_broadcasted[:,:,0:1], other_out), -1)
            total_iterated_output.append(final_out)
        return final_out, total_iterated_output



class IsgnBeatMeasDecoderX(IsgnBeatMeasDecoder):
    def __init__(self, net_params):        
        super(IsgnBeatMeasDecoderX, self).__init__(net_params)
        self.final_graph = GatedGraphX(net_params.final.input - net_params.final.margin, net_params.final.margin, net_params.num_edge_types)
        self.fc = nn.Sequential(
            nn.Linear(net_params.final.margin + net_params.time_reg.size * 2, net_params.final.margin),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.final.margin, net_params.output_size-1),
        )

    def run_note_level_decoder(self, out_with_result, edges, beat_numbers, beat_tempo_vel_broadcasted, total_iterated_output):
        out_with_result = out_with_result[:,:,:-self.final_graph_margin_size]
        hidden = torch.zeros(out_with_result.shape[0], out_with_result.shape[1], self.final_graph_margin_size).to(out_with_result.device)
        for i in range(self.num_sequence_iteration):
            hidden = self.final_graph(out_with_result, hidden, edges, iteration=self.num_graph_iteration)
            initial_out = out_with_result[:, :,
                            -self.output_size: ]
            margin_in_beat = make_higher_node(hidden, self.final_margin_attention, beat_numbers,
                                                        beat_numbers, lower_is_note=True)
            out_in_beat = make_higher_node(initial_out, self.final_beat_attention, beat_numbers,
                                                    beat_numbers, lower_is_note=True)
            out_beat_cat = torch.cat((out_in_beat, margin_in_beat), 2)
            # out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
            out_beat_rnn_result = run_hierarchy_lstm_with_pack(out_beat_cat, self.tempo_rnn)
            out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, beat_numbers)

            other_out = self.fc(torch.cat([out_beat_spanned, hidden], dim=-1))

            # final_out = torch.cat((tempos_spanned, other_out), 2)
            out_with_result = torch.cat((out_with_result[:, :, :-self.output_size  + 1],
                                            other_out), 2)
            final_out = torch.cat((beat_tempo_vel_broadcasted[:,:,0:1], other_out), -1)
            total_iterated_output.append(final_out)
        return final_out, total_iterated_output


class IsgnBeatMeasDecoderXBias(IsgnBeatMeasDecoderX):
    def __init__(self, net_params):        
        super(IsgnBeatMeasDecoderXBias, self).__init__(net_params)
        self.final_graph = GatedGraphXBias(net_params.final.input - net_params.final.margin, net_params.final.margin, net_params.num_edge_types)


class IsgnConvDecoder(IsgnMeasNoteDecoderV3):
    def __init__(self, net_params):
        super().__init__(net_params)
        self.final_graph = GraphConvStack(net_params.final.input, net_params.final.margin, net_params.num_edge_types,
                            num_layers=net_params.final.layer,
                            drop_out=net_params.drop_out)
        self.initial_result_fc = nn.Sequential(
            nn.Linear(net_params.final.input
            - net_params.encoder.size
            - net_params.time_reg.size * 2
            - net_params.output_size, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.encoder.size, net_params.output_size),
        )

        self.fc = nn.Sequential(
            nn.Linear(net_params.final.margin + net_params.time_reg.size*2 + net_params.output_size, net_params.final.margin),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.final.margin, net_params.output_size-1),
        )

       
    def run_final_isgn(self, note_out, perf_embedding, res_info, edges, note_locations):
        num_notes = note_out.shape[1]
        beat_numbers = note_locations['beat']

        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.tempo_rnn.hidden_size * 2)).to(note_out.device)

        initial_output = self.initial_result_fc(note_out)
        total_iterated_output = [initial_output]

        out_with_result = torch.cat(
            (note_out, perf_embedding, initial_beat_hidden, initial_output), 2)
            # (note_out, perform_z_batched, initial_beat_hidden, initial_output, initial_margin), 2)
    
        for i in range(self.num_sequence_iteration):
            final_hidden = self.final_graph(out_with_result, edges)
            out_in_beat = make_higher_node(initial_output, self.final_beat_attention, beat_numbers,
                                        beat_numbers, lower_is_note=True)
            hidden_in_beat = make_higher_node(final_hidden, self.final_margin_attention, beat_numbers,
                                                        beat_numbers, lower_is_note=True)
            out_beat_cat = self.concat_tempo_rnn_input(out_in_beat, hidden_in_beat, res_info)
            out_beat_rnn_result, _ = self.tempo_rnn(out_beat_cat)
            tempo_out = self.tempo_fc(out_beat_rnn_result)

            tempos_spanned = span_beat_to_note_num(tempo_out, beat_numbers)
            out_beat_spanned = span_beat_to_note_num(out_beat_rnn_result, beat_numbers)
            # out_with_result = torch.cat([note_out, perf_embedding, out_beat_spanned, initial_output])
            final_hidden = torch.cat([final_hidden, out_beat_spanned, tempos_spanned, initial_output[:,:,1:]], 2)
            other_out = self.fc(final_hidden)

            final_out = torch.cat((tempos_spanned, other_out), 2)
            out_with_result = torch.cat([note_out, perf_embedding, out_beat_spanned, final_out], 2)
            total_iterated_output.append(final_out)
        return final_out, total_iterated_output

    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        # note_out, measure_out = score_embedding
        note_out = score_embedding['total_note_cat']

        perform_z, perform_z_measure_spanned = self.handle_perform_z(score_embedding, perf_embedding, edges, note_locations)
        measure_tempo_vel, measure_tempo_vel_broadcasted = self.get_measure_level_output(score_embedding, perform_z_measure_spanned, res_info, note_locations)
        note_out = torch.cat([note_out, measure_tempo_vel_broadcasted], dim=-1)
        final_out, total_iterated_output = self.run_final_isgn(note_out, perform_z, res_info, edges, note_locations)

        return final_out, {'iter_out': total_iterated_output, 'meas_out':measure_tempo_vel}



class HanDecoder(nn.Module):
    def __init__(self, net_params):
        super(HanDecoder, self).__init__()
        self.num_attention_head = net_params.num_attention_head
        self.final_hidden_size = net_params.final.size

        self.result_for_tempo_attention = ContextAttention(net_params.output_size - 1, 1)
        self.beat_tempo_fc = nn.Linear(net_params.beat.size, 1)
        self.fc = nn.Linear(self.final_hidden_size, net_params.output_size - 1)

    def handle_style_vector(self, perf_emb, score_emb, note_locations):
        return None

    def concat_beat_rnn_input(self, batch_ids, beat_emb, measure_emb, perf_emb, res_info, prev_tempo, beat_results, note_index, beat_index, measure_index):
        return None

    def concat_final_rnn_input(self, note_emb, beat_emb, measure_emb, perf_emb, res_info, prev_out, note_index, beat_index, measure_index):
        return None

    def init_hidden(self, num_layer, num_direction, batch_size, hidden_size, device):
        h0 = torch.zeros(num_layer * num_direction, batch_size, hidden_size).to(device)
        return (h0, h0.clone())

    def run_beat_and_note_regressive_decoding(self, score_embedding, perf_emb, res_info, note_locations):
        # note_emb, beat_emb, measure_emb, _, _ = score_embedding
        note_emb = score_embedding['note']
        beat_emb = score_embedding['beat']
        measure_emb = score_embedding['measure']
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        num_batch = note_emb.size(0)
        num_notes = note_emb.shape[1]
        num_beats = beat_emb.shape[1]

        beat_results = torch.zeros(num_batch, num_beats, self.fc.out_features).to(note_emb.device)
        final_hidden = self.init_hidden(1, 1, num_batch, self.final_hidden_size, note_emb.device)
        tempo_hidden = self.init_hidden(1, 1, num_batch, self.beat_tempo_fc.in_features, note_emb.device)

        prev_out = torch.zeros(num_batch, self.fc.out_features + 1).to(note_emb.device) # +1 for tempo, becasue tempo is calculated with beat
        out_total = torch.zeros(num_batch, num_notes, self.fc.out_features + 1).to(note_emb.device)
        zero_shifted_beat_numbers = beat_numbers - beat_numbers[:,0:1]
        zero_shifted_measure_numbers = measure_numbers - measure_numbers[:,0:1]

        zero_shifted_beat_numbers.clamp_min_(0)
        zero_shifted_measure_numbers.clamp_min_(0)

        diff_beat_numbers = torch.diff(beat_numbers)
        import time
        for i in range(num_notes):
          if i > 0:
            beat_changed = diff_beat_numbers[:,i-1]
          else:
            beat_changed = diff_beat_numbers[:,0].clone()
            beat_changed[:] = 1
          if torch.sum(beat_changed) > 0: # at least one sample's beat has changed
            selected_batch_ids = torch.where(beat_changed)[0]
            if i ==0:
              corresp_result = torch.zeros((num_batch,1,self.fc.out_features)).to(note_emb.device)
            else:
              corresp_result = get_beat_corresp_out(out_total, beat_numbers, selected_batch_ids, i)
              corresp_result = corresp_result[:,:,QPM_INDEX+1:]
            result_node = self.result_for_tempo_attention(corresp_result)
            current_beat = zero_shifted_beat_numbers[:,i]
            current_measure = zero_shifted_measure_numbers[:,i]
            selected_batch_ids = torch.where(beat_changed)[0]
            beat_results[selected_batch_ids, current_beat[selected_batch_ids]] = result_node
            beat_tempo_cat = self.concat_beat_rnn_input(selected_batch_ids, beat_emb, measure_emb, perf_emb, res_info, prev_out[:, QPM_INDEX:QPM_INDEX+1], beat_results, i, current_beat, current_measure)
            selected_batch_tempo_hidden = (tempo_hidden[0][:, selected_batch_ids], tempo_hidden[1][:, selected_batch_ids])
            beat_forward, temp_tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, selected_batch_tempo_hidden)
            tempo_hidden[0][:, selected_batch_ids], tempo_hidden[1][:, selected_batch_ids] = temp_tempo_hidden
            tmp_tempos = self.beat_tempo_fc(beat_forward)
            prev_out[selected_batch_ids, QPM_INDEX:QPM_INDEX+1] = tmp_tempos[:,0]
          out_combined = self.concat_final_rnn_input(note_emb, beat_emb, measure_emb, perf_emb, res_info, prev_out, i, current_beat, current_measure)
          out, final_hidden = self.output_lstm(out_combined, final_hidden)
          out = self.fc(out)[:,0]
          prev_out[:, QPM_INDEX+1:] = out
          out_total[:, i] = prev_out

        out_total[note_emb.sum(-1)==0] = 0
        return out_total

    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        perf_emb = self.handle_style_vector(perf_embedding, score_embedding, note_locations)
        return self.run_beat_and_note_regressive_decoding(score_embedding, perf_emb, res_info, note_locations), []



class HanMeasureZDecoder(HanDecoder):
    def __init__(self, net_params):
        super(HanMeasureZDecoder, self).__init__(net_params)
        self.final_hidden_size = net_params.final.size
        self.final_input = net_params.final.input
        self.num_attention_head = net_params.num_attention_head
        self.final_input = net_params.final.input


        self.style_vector_expandor = nn.Sequential(
            nn.Linear(net_params.encoded_vector_size, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.perform_style_to_measure = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size, net_params.encoder.size, num_layers=1, bidirectional=False)
        self.measure_perf_fc = nn.Linear(net_params.encoder.size, net_params.encoder.size)

        self.beat_tempo_forward = nn.LSTM(
                (net_params.beat.size + net_params.measure.size) * 2 + 5 + 3 + net_params.output_size + net_params.encoder.size, net_params.beat.size,
                num_layers=1, batch_first=True, bidirectional=False)
        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)

    def handle_style_vector(self, perf_emb, score_emb, note_locations):
        measure_numbers = note_locations['measure']
        _, _, measure_emb, _, _ = score_emb
        
        perform_z = self.style_vector_expandor(perf_emb)
        perform_z = perform_z.view(-1)
        
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1, num_measures, -1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_emb), 2)
        measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        measure_perform_style = self.measure_perf_fc(measure_perform_style)

        return measure_perform_style

    def concat_beat_rnn_input(self, beat_emb, measure_emb, perf_emb, res_info, prev_tempo, beat_results, note_index, beat_index, measure_index):
        return torch.cat((beat_emb[0, beat_index, :],
                                            measure_emb[0, measure_index, :], prev_tempo, res_info[0, beat_index, :],
                                            beat_results[beat_index, :],
                                            perf_emb[0, measure_index, :])).view(1, 1, -1)

    def concat_final_rnn_input(self, note_emb, beat_emb, measure_emb, perf_emb, res_info, prev_out, note_index, beat_index, measure_index):
        return torch.cat(
                (note_emb[0, note_index, :], beat_emb[0, beat_index, :],
                    measure_emb[0, measure_index, :],
                    prev_out, perf_emb[0, measure_index,:])).view(1, 1, -1)



class HanDecoderSingleZ(HanDecoder):
    def __init__(self, net_params) -> None:
        super(HanDecoderSingleZ, self).__init__(net_params)
        self.final_hidden_size = net_params.final.size
        self.final_input = net_params.final.input
        self.num_attention_head = net_params.num_attention_head

        self.style_vector_expandor = nn.Sequential(
            nn.Linear(net_params.encoded_vector_size, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )

        self.beat_tempo_forward = nn.LSTM(
                (net_params.beat.size + net_params.measure.size) * 2 + 5 + 3 + net_params.output_size + net_params.encoder.size, net_params.beat.size,
                num_layers=1, batch_first=True, bidirectional=False)
        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)


    def handle_style_vector(self, perf_emb, score_emb, note_locations):
        perform_z = self.style_vector_expandor(perf_emb)
        perform_z = perform_z.view(-1)

        return perform_z

    def concat_beat_rnn_input(self, beat_emb, measure_emb, perf_emb, res_info, prev_tempo, beat_results, note_index, beat_index, measure_index):
        return torch.cat((beat_emb[0, beat_index, :],
                                            measure_emb[0, measure_index, :], prev_tempo, res_info[0, beat_index, :],
                                            beat_results[beat_index, :],
                                            perf_emb)).view(1, 1, -1)

    def concat_final_rnn_input(self, note_emb, beat_emb, measure_emb, perf_emb, res_info, prev_out, note_index, beat_index, measure_index):
        return torch.cat(
                (note_emb[0, note_index, :], beat_emb[0, beat_index, :],
                    measure_emb[0, measure_index, :],
                    prev_out, perf_emb)).view(1, 1, -1)


class HanMeasNoteDecoder(HanDecoder):
    def __init__(self, net_params):
        super(HanMeasNoteDecoder, self).__init__(net_params)
        self.final_hidden_size = net_params.final.size
        self.final_input = net_params.final.input + 2
        self.num_attention_head = net_params.num_attention_head

        self.style_vector_expandor = nn.Sequential(
            nn.Linear(net_params.encoded_vector_size, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.measure_out_lstm = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size + 2 + 8, net_params.measure.size, batch_first=True)
        self.measure_out_fc =  nn.Linear(net_params.measure.size , 2)

        self.beat_tempo_forward = nn.LSTM(
                (net_params.beat.size + net_params.measure.size) * 2 + net_params.output_size + net_params.encoder.size + 2, net_params.beat.size,
                num_layers=1, batch_first=True, bidirectional=False)
        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)


    def handle_style_vector(self, perf_emb):
        perform_z = self.style_vector_expandor(perf_emb)
        if len(perf_emb.shape) > 1:
            perform_z = perform_z.unsqueeze(1)
        else:
            perform_z = perform_z.view(1, 1, -1)
        return perform_z
    
    def concat_final_rnn_input(self, note_emb, beat_emb, measure_emb, perf_emb, res_info, prev_out, note_index, beat_index, measure_index):
        return torch.cat([
          note_emb[:, note_index],
          beat_emb[torch.arange(len(beat_index)), beat_index],
          measure_emb[torch.arange(len(measure_index)), measure_index],
          prev_out,
          perf_emb[:, 0]
          ], dim=-1).unsqueeze(1)
        
        # return torch.cat(
        #         (note_emb[0, note_index, :], beat_emb[0, beat_index, :],
        #             measure_emb[0, measure_index, :],
        #             prev_out, perf_emb[0,0,:])).view(1, 1, -1)
    
    def run_measure_level(self, score_embedding, perform_z, res_info, note_locations):
        # _, _, measure_out, _, _ = score_embedding
        measure_out = score_embedding['measure']
        measure_numbers = note_locations['measure']
        num_measures = measure_out.shape[1]

        prev_out = torch.zeros(measure_out.shape[0], 1, 2).to(measure_out.device)
        measure_tempo_vel = torch.zeros(measure_out.shape[0], num_measures, 2).to(measure_out.device)
        measure_hidden = self.init_hidden(self.measure_out_lstm.num_layers, 1, measure_out.shape[0], self.measure_out_lstm.hidden_size, measure_out.device)
        perform_z_view = perform_z.view(measure_out.shape[0], 1 , -1).repeat(1, measure_out.shape[1], 1)
        measure_cat = torch.cat([measure_out, perform_z_view, res_info], dim=-1)
        for i in range(num_measures):
            cur_input = torch.cat([measure_cat[:,i:i+1,:], prev_out], dim=-1)
            cur_tempo_vel, measure_hidden = self.measure_out_lstm(cur_input, measure_hidden)
            measure_tempo_vel[:,i:i+1,:] = self.measure_out_fc(cur_tempo_vel)
            prev_out = measure_tempo_vel[:,i:i+1,:]
        measure_tempo_vel[measure_out.sum(-1)==0] = 0
        measure_tempo_vel_broadcasted = span_beat_to_note_num(measure_tempo_vel, measure_numbers)
        return measure_tempo_vel_broadcasted, measure_tempo_vel


    def concat_beat_rnn_input(self, batch_ids, beat_emb, measure_emb, perf_emb, res_info, prev_tempo, beat_results, note_index, beat_index, measure_index):
        '''
        out.
        
        '''
        return torch.cat([
          beat_emb[batch_ids, beat_index[batch_ids]],
          measure_emb[batch_ids, measure_index[batch_ids]],
          prev_tempo[batch_ids],
          beat_results[batch_ids, beat_index[batch_ids]],
          perf_emb[batch_ids, 0]
          ], dim=-1).unsqueeze(1)
        
        # return torch.cat((beat_emb[0, beat_index, :],
        #                                     measure_emb[0, measure_index, :], prev_tempo,
        #                                     beat_results[beat_index, :],
        #                                     perf_emb[0, 0, :])).view(1, 1, -1)


    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        perform_z = self.handle_style_vector(perf_embedding)
        _, measure_tempo_vel = self.run_measure_level(score_embedding, perform_z, res_info, note_locations)
        # add measure-level tempo and velocity to score embedding.
        # new_score_embedding = (score_embedding[0], score_embedding[1], torch.cat([score_embedding[2], measure_tempo_vel], dim=-1), score_embedding[3], score_embedding[4])  
        score_embedding['measure'] = torch.cat([score_embedding['measure'], measure_tempo_vel], dim=-1)
        return self.run_beat_and_note_regressive_decoding(score_embedding, perform_z, _, note_locations), {'meas_out':measure_tempo_vel, 'iter_out':[]}

class HanHierDecoder(nn.Module):
    def __init__(self, net_params) -> None:
        super(HanDecoder, self).__init__()

    def forward(self, score_embedding, perf_embedding, note_location):
        _, _, measure_emb, _, _ = score_embedding
        measure_numbers = note_location['measure']
        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z = perform_z.view(-1)

        # num_measures = measure_numbers[start_index + num_notes - 1] - measure_numbers[start_index] + 1
        num_measures = measure_emb.sbape[1]
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1, num_measures, -1)
        if perform_z_measure_spanned.shape[1] != measure_emb.shape[1]:
            print(measure_numbers)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_emb), 2)
        measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        hierarchy_nodes = measure_emb
        num_hierarchy_nodes = hierarchy_nodes.shape[1]
        hierarchy_nodes_latent_combined = torch.cat((hierarchy_nodes, measure_perform_style), 2)

        out_hidden_state = self.init_hidden(1,1,x.size(0), self.final_hidden_size)
        prev_out = torch.zeros(1,1,net_params.output_size).to(self.device)
        out_total = torch.zeros(1, num_hierarchy_nodes, net_params.output_size).to(self.device)

        for i in range(num_hierarchy_nodes):
            out_combined = torch.cat((hierarchy_nodes_latent_combined[:,i:i+1,:], prev_out),2)
            out, out_hidden_state = self.output_lstm(out_combined, out_hidden_state)
            out = self.fc(out)
            out_total[:,i,:] = out
            prev_out = out.view(1,1,-1)
        return out_total, perform_mu, perform_var, note_out