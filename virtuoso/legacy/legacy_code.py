def scale_model_prediction_to_original(prediction, means, stds, loss_type='MSE'):
    for i in range(len(stds)):
        for j in range(len(stds[i])):
            if stds[i][j] < 1e-4:
                stds[i][j] = 1
    prediction = np.squeeze(np.asarray(prediction.cpu()))
    num_notes = len(prediction)
    if loss_type == 'MSE':
        for i in range(11):
            prediction[:, i] *= stds[1][i]
            prediction[:, i] += means[1][i]
        for i in range(11, 15):
            prediction[:, i] *= stds[1][i+4]
            prediction[:, i] += means[1][i+4]
    elif loss_type == 'CE':
        prediction_in_value = np.zeros((num_notes, 16))
        for i in range(num_notes):
            bin_range_start = 0
            for j in range(15):
                feature_bin_size = len(BINS[j]) - 1
                feature_class = np.argmax(
                    prediction[i, bin_range_start:bin_range_start + feature_bin_size])
                feature_value = (BINS[j][feature_class] +
                                 BINS[j][feature_class + 1]) / 2
                prediction_in_value[i, j] = feature_value
                bin_range_start += feature_bin_size
            prediction_in_value[i, 15] = prediction[i, -1]
        prediction = prediction_in_value

    return prediction





class HanVirtuosoNet(VirtuosoNet):
    def __init__(self, net_params):
        super(HanVirtuosoNet, self).__init__()
        self.network_params = net_params
        self.score_encoder = encs.HanEncoder(net_params)
        self.performance_encoder = encp.HanPerfEncoder(net_params)
        # self.performance_decoder = dec.HanDecoderSingleZ(net_params)
        self.performance_decoder = dec.HanDecoderSingleZ(net_params)
        self.residual_info_selector = res.TempoVecSelector()

class IsgnVirtuosoNet(VirtuosoNet):
    def __init__(self, net_params):
        super(IsgnVirtuosoNet, self).__init__()
        self.network_params = net_params
        # self.score_encoder = encs.IsgnResEncoderV2(net_params)
        self.score_encoder = encs.IsgnOldEncoder(net_params)

        self.performance_encoder = encp.IsgnPerfEncoder(net_params)
        # self.performance_decoder = dec.IsgnDecoder(net_params)
        self.performance_decoder = dec.IsgnMeasNoteDecoderV2(net_params)

        # self.residual_info_selector = res.TempoVecSelector() 
        self.residual_info_selector = res.TempoVecMeasSelector()

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
        self.score_encoder = encs.IsgnOldGraphSingleEncoder(net_params)

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
        beat_tempo_vector = note_tempo_infos_to_beat(x, beat_numbers, TEMPO_IDX)

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

        self.score_encoder = encs.HanEncoder(net_params)
        self.perf_encoder = encp.HanPerfEncoder(net_params)
       
        self.perform_style_to_measure = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size, self.encoder_size, num_layers=1, bidirectional=False)

        if self.hierarchy == 'measure':
            self.output_lstm = nn.LSTM(self.measure_hidden_size * 2 + self.encoder_size + self.output_size, self.final_hidden_size, num_layers=1, batch_first=True)
        elif self.hierarchy == 'beat':
            self.output_lstm = nn.LSTM((self.beat_hidden_size + self.measure_hidden_size) * 2 + self.encoder_size + self.output_size, self.final_hidden_size, num_layers=1, batch_first=True)
        else:
            self.beat_tempo_forward = nn.LSTM(
                (self.beat_hidden_size + self.measure_hidden_size) * 2 + 5 + 3 + self.output_size + self.encoder_size, self.beat_hidden_size,
                num_layers=1, batch_first=True, bidirectional=False)
            self.result_for_tempo_attention = ContextAttention(self.output_size - 1, 1)
            self.beat_tempo_fc = nn.Linear(self.beat_hidden_size, 1)

        if self.hierarchy:
            self.fc = nn.Linear(self.final_hidden_size, self.output_size)
        else:
            self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
            if self.is_baseline:
                self.fc = nn.Linear(self.final_hidden_size, self.output_size)
            else:
                self.fc = nn.Linear(self.final_hidden_size, self.output_size - 1)


        self.style_vector_expandor = nn.Sequential(
            nn.Linear(self.encoded_vector_size, self.encoder_size),
            nn.Dropout(self.drop_out),
            nn.ReLU()
        )




    def forward(self, x, y, edges, note_locations, initial_z=False, rand_threshold=0.2, return_z=False):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        num_notes = x.shpae[1]
        note_out, beat_hidden_out, measure_hidden_out = \
            self.score_encoder(x, note_locations)
        beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers, num_notes)
        measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes)

        perform_z, perform_mu, perform_var = self.perf_encoder((note_out, beat_out_spanned, measure_out_spanned), note_locations)


        '''
        # perform_z = self.performance_decoder(perform_z)
        perform_z = self.style_vector_expandor(perform_z)
        perform_z_batched = perform_z.repeat(x.shape[1], 1).view(1,x.shape[1], -1)
        perform_z = perform_z.view(-1)

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
        '''
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
                out_combined = torch.cat((hierarchy_nodes_latent_combined[:,i:i+1,:], prev_out),2)
                out, out_hidden_state = self.output_lstm(out_combined, out_hidden_state)
                out = self.fc(out)
                out_total[:,i,:] = out
                prev_out = out.view(1,1,-1)
            return out_total, perform_mu, perform_var, note_out

        else:
            final_hidden = self.init_hidden(1, 1, x.size(0), self.final_hidden_size)
            qpm_primo = x[:, 0, QPM_PRIMO_IDX]
            tempo_primo = x[0, 0, TEMPO_PRIMO_IDX:]
            prev_out = torch.zeros(self.output_size).to(self.device)
            prev_tempo = prev_out[QPM_INDEX:QPM_INDEX+1]
            prev_beat = -1
            prev_beat_end = 0
            out_total = torch.zeros(num_notes, self.output_size).to(self.device)
            prev_out_list = []
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

                    beat_tempo_cat = torch.cat((beat_hidden_out[0, current_beat, :],
                                                measure_hidden_out[0, current_measure, :], prev_tempo, x[0,i,self.input_size-2:self.input_size-1],
                                                result_nodes[current_beat, :],
                                                measure_perform_style[0, current_measure, :])).view(1, 1, -1)
                    beat_forward, tempo_hidden = self.beat_tempo_forward(beat_tempo_cat, tempo_hidden)

                    tmp_tempos = self.beat_tempo_fc(beat_forward)

                    prev_beat_end = i
                    prev_tempo = tmp_tempos.view(1)
                    prev_beat = current_beat

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

