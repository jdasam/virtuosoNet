import torch
import torch.nn as nn
from .model_utils import make_higher_node, reparameterize, span_beat_to_note_num
from .module import GatedGraph, SimpleAttention, ContextAttention
from .model_constants import QPM_INDEX, QPM_PRIMO_IDX

class IsgnDecoder(nn.Module):
    def __init__(self, net_params) -> None:
        super(IsgnDecoder, self).__init__()
        self.output_size = net_params.output_size
        self.num_sequence_iteration = net_params.sequence_iteration
        self.num_graph_iteration = net_params.graph_iteration
        self.final_graph_margin_size = net_params.final.margin
        self.final_beat_hidden_idx = net_params.note.size + net_params.measure.size * 2 + net_params.encoder.size

        self.beat_tempo_contractor = nn.Sequential(
            nn.Linear(net_params.final.input - net_params.time_reg.size * 2, net_params.time_reg.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.style_vector_expandor = nn.Sequential(
            nn.Linear(net_params.encoded_vector_size, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU()
        )
        self.perform_style_to_measure = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size, net_params.encoder.size, num_layers=1, bidirectional=False)
        self.measure_perf_fc = nn.Linear(net_params.encoder.size, net_params.encoder.size)

        self.initial_result_fc = nn.Sequential(
            nn.Linear(net_params.note.size + net_params.measure.size * 2, net_params.encoder.size),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.encoder.size, net_params.output_size),
            nn.ReLU()
        )

        self.final_graph = GatedGraph(net_params.final.input, net_params.num_edge_types,
                                      net_params.output_size + net_params.final.margin)

        self.tempo_rnn = nn.LSTM(net_params.final.margin + net_params.output_size + 8, net_params.time_reg.size,
                                    num_layers=net_params.time_reg.layer, batch_first=True, bidirectional=True)

        self.final_beat_attention = ContextAttention(net_params.output_size, 1)
        self.final_margin_attention = ContextAttention(net_params.final.margin, net_params.num_attention_head)
        self.tempo_fc = nn.Linear(net_params.time_reg.size * 2, 1)

        # self.fc = nn.Linear(net_params.final.input, net_params.output_size-1)
        self.fc = nn.Sequential(
            nn.Linear(net_params.final.input, net_params.final.margin),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.final.margin, net_params.output_size-1),
        )

        
    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        note_out, measure_out = score_embedding
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        num_notes = note_out.shape[1]

        perform_z = self.style_vector_expandor(perf_embedding)

        initial_output = self.initial_result_fc(note_out)
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_out), 2)
        measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        measure_perform_style = self.measure_perf_fc(measure_perform_style)
        measure_perform_style_spanned = span_beat_to_note_num(measure_perform_style, measure_numbers)

        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.tempo_rnn.hidden_size * 2)).to(note_out.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(note_out.device)


        total_iterated_output = [initial_output]

        out_with_result = torch.cat(
            (note_out, measure_perform_style_spanned, initial_beat_hidden, initial_output, initial_margin), 2)
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
            out_beat_cat = torch.cat((out_in_beat, margin_in_beat, res_info), 2)
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
            # print([times[i]-times[i-1] for i in range(1, len(times))])
        return final_out, total_iterated_output


class IsgnMeasNoteDecoder(IsgnDecoder):
    def __init__(self, net_params) -> None:
        super(IsgnMeasNoteDecoder, self).__init__(net_params)
        self.perform_style_to_measure = None
        self.measure_perf_fc = None
        self.measure_out_lstm = nn.LSTM(net_params.measure.size * 2 + net_params.encoder.size + 2, net_params.measure.size, bidirectional=True, batch_first=True)
        self.measure_out_fc = nn.Linear(net_params.measure.size * 2, 2)
        self.final_beat_hidden_idx += 2
        self.initial_result_fc = nn.Sequential(
            nn.Linear(net_params.note.size + net_params.measure.size * 2 + 2, net_params.encoder.size ),
            nn.Dropout(net_params.drop_out),
            nn.ReLU(),
            nn.Linear(net_params.encoder.size, net_params.output_size),
        )
    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        note_out, measure_out = score_embedding
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        num_notes = note_out.shape[1]
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1


        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_out), 2)
        perform_z = perform_z.repeat(num_notes, 1).unsqueeze(0)
        
        measure_hidden = (torch.zeros(2*self.measure_out_lstm.num_layers, 1, measure_out.shape[-1]//2).to(perform_z.device), 
                          torch.zeros(2*self.measure_out_lstm.num_layers, 1, measure_out.shape[-1]//2).to(perform_z.device))
        prev_out = torch.zeros(measure_out.shape[0], 1, 2).to(perform_z.device)
        measure_tempo_vel = torch.zeros(measure_out.shape[0], num_measures, 2).to(note_out.device)
        for i in range(num_measures):
            cur_input = torch.cat([perform_z_measure_cat[:,i:i+1,:], prev_out], dim=-1)
            cur_tempo_vel, measure_hidden =  self.measure_out_lstm(cur_input, measure_hidden)
            measure_tempo_vel[:,i:i+1,:] = self.measure_out_fc(cur_tempo_vel)
            prev_out = measure_tempo_vel[:,i:i+1,:]


        measure_tempo_vel_broadcasted = span_beat_to_note_num(measure_tempo_vel, measure_numbers)
        note_out = torch.cat([note_out, measure_tempo_vel_broadcasted], dim=-1)

        initial_output = self.initial_result_fc(note_out)
        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.tempo_rnn.hidden_size * 2)).to(note_out.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(note_out.device)


        total_iterated_output = [initial_output]

        out_with_result = torch.cat(
            (note_out, perform_z, initial_beat_hidden, initial_output, initial_margin), 2)
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
            out_beat_cat = torch.cat((out_in_beat, margin_in_beat, res_info), 2)
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
            # print([times[i]-times[i-1] for i in range(1, len(times))])
        return final_out, {'iter_out': total_iterated_output, 'meas_out':measure_tempo_vel}


class IsgnMeasNoteDecoderV2(IsgnDecoder):
    # connect res info to measure-level decoder
    def __init__(self, net_params) -> None:
        super(IsgnMeasNoteDecoderV2, self).__init__(net_params)
        self.perform_style_to_measure = None
        self.measure_perf_fc = None
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

    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        note_out, measure_out = score_embedding
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        num_notes = note_out.shape[1]
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1


        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1,num_measures, -1)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_out, res_info), 2)
        perform_z = perform_z.repeat(num_notes, 1).unsqueeze(0)
        
        measure_hidden = (torch.zeros(2*self.measure_out_lstm.num_layers, 1, measure_out.shape[-1]//2).to(perform_z.device), 
                          torch.zeros(2*self.measure_out_lstm.num_layers, 1, measure_out.shape[-1]//2).to(perform_z.device))
        prev_out = torch.zeros(measure_out.shape[0], 1, 2).to(perform_z.device)
        measure_tempo_vel = torch.zeros(measure_out.shape[0], num_measures, 2).to(note_out.device)
        for i in range(num_measures):
            cur_input = torch.cat([perform_z_measure_cat[:,i:i+1,:], prev_out], dim=-1)
            cur_tempo_vel, measure_hidden =  self.measure_out_lstm(cur_input, measure_hidden)
            measure_tempo_vel[:,i:i+1,:] = self.measure_out_fc(cur_tempo_vel)
            prev_out = measure_tempo_vel[:,i:i+1,:]


        measure_tempo_vel_broadcasted = span_beat_to_note_num(measure_tempo_vel, measure_numbers)
        note_out = torch.cat([note_out, measure_tempo_vel_broadcasted], dim=-1)

        initial_output = self.initial_result_fc(note_out)
        initial_beat_hidden = torch.zeros((note_out.size(0), num_notes, self.tempo_rnn.hidden_size * 2)).to(note_out.device)
        initial_margin = torch.zeros((note_out.size(0), num_notes, self.final_graph_margin_size)).to(note_out.device)


        total_iterated_output = [initial_output]

        out_with_result = torch.cat(
            (note_out, perform_z, initial_beat_hidden, initial_output, initial_margin), 2)
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
            out_beat_cat = torch.cat((out_in_beat, margin_in_beat), 2)
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
            # print([times[i]-times[i-1] for i in range(1, len(times))])
        return final_out, {'iter_out': total_iterated_output, 'meas_out':measure_tempo_vel}

class HanDecoder(nn.Module):
    def __init__(self, net_params) -> None:
        super(HanDecoder, self).__init__()
        self.final_hidden_size = net_params.final.size
        self.final_input = net_params.final.input
        self.num_attention_head = net_params.num_attention_head

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
        self.result_for_tempo_attention = ContextAttention(net_params.output_size - 1, 1)
        self.beat_tempo_fc = nn.Linear(net_params.beat.size, 1)
        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.final_hidden_size, net_params.output_size - 1)


    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        note_emb, beat_emb, measure_emb, _, _ = score_embedding
        assert beat_emb.shape[1] == res_info.shape[1]
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        num_notes = note_emb.shape[1]

        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z = perform_z.view(-1)

        tempo_hidden = self.init_hidden(1,1, note_emb.size(0), self.beat_tempo_fc.in_features, note_emb.device)
        num_beats = beat_emb.size(1)
        result_nodes = torch.zeros(num_beats, self.fc.out_features).to(note_emb.device)

        # num_measures = measure_numbers[start_index + num_notes - 1] - measure_numbers[start_index] + 1
        num_measures = measure_numbers[-1] - measure_numbers[0] + 1
        perform_z_measure_spanned = perform_z.repeat(num_measures, 1).view(1, num_measures, -1)
        if perform_z_measure_spanned.shape[1] != measure_emb.shape[1]:
            print(measure_numbers)
        perform_z_measure_cat = torch.cat((perform_z_measure_spanned, measure_emb), 2)
        measure_perform_style, _ = self.perform_style_to_measure(perform_z_measure_cat)
        measure_perform_style = self.measure_perf_fc(measure_perform_style)
        final_hidden = self.init_hidden(1, 1, note_emb.size(0), self.final_hidden_size, note_emb.device)
        prev_out = torch.zeros(self.fc.out_features + 1).to(note_emb.device)
        prev_tempo = prev_out[QPM_INDEX:QPM_INDEX+1]
        prev_beat = -1
        prev_beat_end = 0
        out_total = torch.zeros(num_notes, self.fc.out_features + 1).to(note_emb.device)
        prev_out_list = []
        for i in range(num_notes):
            current_beat = beat_numbers[i] - beat_numbers[0]
            current_measure = measure_numbers[i] - measure_numbers[0]
            if current_beat > prev_beat:  # beat changed
                if i - prev_beat_end > 0:  # if there are outputs to consider
                    corresp_result = torch.stack(prev_out_list).unsqueeze_(0)
                else:  # there is no previous output
                    corresp_result = torch.zeros((1,1,self.fc.out_features)).to(note_emb.device)
                result_node = self.result_for_tempo_attention(corresp_result)
                prev_out_list = []
                result_nodes[current_beat, :] = result_node

                beat_tempo_cat = torch.cat((beat_emb[0, current_beat, :],
                                            measure_emb[0, current_measure, :], prev_tempo, res_info[0, current_beat, :],
                                            result_nodes[current_beat, :],
                                            measure_perform_style[0, current_measure, :])).view(1, 1, -1)
                beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)

                tmp_tempos = self.beat_tempo_fc(beat_forward)

                prev_beat_end = i
                prev_tempo = tmp_tempos.view(1)
                prev_beat = current_beat

            out_combined = torch.cat(
                (note_emb[0, i, :], beat_emb[0, current_beat, :],
                    measure_emb[0, current_measure, :],
                    prev_out, measure_perform_style[0, current_measure,:])).view(1, 1, -1)

            out, final_hidden = self.output_lstm(out_combined, final_hidden)
            # out = torch.cat((out, out_combined), 2)
            out = out.view(-1)
            out = self.fc(out)

            prev_out_list.append(out)
            out = torch.cat((prev_tempo, out))

            prev_out = out
            out_total[i, :] = out

        out_total = out_total.unsqueeze(0)
            # hidden_total = torch.cat((note_emb, beat_emb, measure), 2)
        return out_total, []
        

    def init_hidden(self, num_layer, num_direction, batch_size, hidden_size, device):
        h0 = torch.zeros(num_layer * num_direction, batch_size, hidden_size).to(device)
        return (h0, h0)




class HanDecoderSingleZ(nn.Module):
    def __init__(self, net_params) -> None:
        super(HanDecoderSingleZ, self).__init__()
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
        self.result_for_tempo_attention = ContextAttention(net_params.output_size - 1, 1)
        self.beat_tempo_fc = nn.Linear(net_params.beat.size, 1)
        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.final_hidden_size, net_params.output_size - 1)


    def forward(self, score_embedding, perf_embedding, res_info, edges, note_locations):
        note_emb, beat_emb, measure_emb, _, _ = score_embedding
        assert beat_emb.shape[1] == res_info.shape[1]
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        num_notes = note_emb.shape[1]

        perform_z = self.style_vector_expandor(perf_embedding)
        perform_z = perform_z.view(-1)

        tempo_hidden = self.init_hidden(1,1, note_emb.size(0), self.beat_tempo_fc.in_features, note_emb.device)
        num_beats = beat_emb.size(1)
        result_nodes = torch.zeros(num_beats, self.fc.out_features).to(note_emb.device)


        final_hidden = self.init_hidden(1, 1, note_emb.size(0), self.final_hidden_size, note_emb.device)
        prev_out = torch.zeros(self.fc.out_features + 1).to(note_emb.device)
        prev_tempo = prev_out[QPM_INDEX:QPM_INDEX+1]
        prev_beat = -1
        prev_beat_end = 0
        out_total = torch.zeros(num_notes, self.fc.out_features + 1).to(note_emb.device)
        prev_out_list = []
        for i in range(num_notes):
            current_beat = beat_numbers[i] - beat_numbers[0]
            current_measure = measure_numbers[i] - measure_numbers[0]
            if current_beat > prev_beat:  # beat changed
                if i - prev_beat_end > 0:  # if there are outputs to consider
                    corresp_result = torch.stack(prev_out_list).unsqueeze_(0)
                else:  # there is no previous output
                    corresp_result = torch.zeros((1,1,self.fc.out_features)).to(note_emb.device)
                result_node = self.result_for_tempo_attention(corresp_result)
                prev_out_list = []
                result_nodes[current_beat, :] = result_node

                beat_tempo_cat = torch.cat((beat_emb[0, current_beat, :],
                                            measure_emb[0, current_measure, :], prev_tempo, res_info[0, current_beat, :],
                                            result_nodes[current_beat, :],
                                            perform_z)).view(1, 1, -1)
                beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)

                tmp_tempos = self.beat_tempo_fc(beat_forward)

                prev_beat_end = i
                prev_tempo = tmp_tempos.view(1)
                prev_beat = current_beat

            out_combined = torch.cat(
                (note_emb[0, i, :], beat_emb[0, current_beat, :],
                    measure_emb[0, current_measure, :],
                    prev_out, perform_z)).view(1, 1, -1)

            out, final_hidden = self.output_lstm(out_combined, final_hidden)
            # out = torch.cat((out, out_combined), 2)
            out = out.view(-1)
            out = self.fc(out)

            prev_out_list.append(out)
            out = torch.cat((prev_tempo, out))

            prev_out = out
            out_total[i, :] = out

        out_total = out_total.unsqueeze(0)
            # hidden_total = torch.cat((note_emb, beat_emb, measure), 2)
        return out_total, []
        

    def init_hidden(self, num_layer, num_direction, batch_size, hidden_size, device):
        h0 = torch.zeros(num_layer * num_direction, batch_size, hidden_size).to(device)
        return (h0, h0)


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