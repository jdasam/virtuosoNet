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


def load_file_and_encode_style(path, perf_name, composer_name):
    test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(
        path, perf_name, composer_name, MEANS, STDS)
    qpm_primo = test_x[0][4]

    test_x, test_y = handle_data_in_tensor(
        test_x, test_y, hierarchy_test=IN_HIER)
    edges = edges_to_matrix(edges, test_x.shape[0])

    if IN_HIER:
        test_x = test_x.view((1, -1, HIER_MODEL.input_size))
        hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
        perform_z_high = encode_performance_style_vector(
            test_x, hier_y, edges, note_locations, model=HIER_MODEL)
        hier_outputs, _ = run_model_in_steps(
            test_x, hier_y, edges, note_locations, model=HIER_MODEL)
        if HIER_MEAS:
            hierarchy_numbers = [x.measure for x in note_locations]
        elif HIER_BEAT:
            hierarchy_numbers = [x.beat for x in note_locations]
        hier_outputs_spanned = HIER_MODEL.span_beat_to_note_num(
            hier_outputs, hierarchy_numbers, test_x.shape[1], 0)
        input_concat = torch.cat((test_x, hier_outputs_spanned), 2)
        batch_y = test_y[1].view(1, -1, MODEL.output_size)
        perform_z_note = encode_performance_style_vector(
            input_concat, batch_y, edges, note_locations, model=MODEL)
        perform_z = [perform_z_high, perform_z_note]

    else:
        batch_x = test_x.view((1, -1, NUM_INPUT))
        batch_y = test_y.view((1, -1, NUM_OUTPUT))
        perform_z = encode_performance_style_vector(
            batch_x, batch_y, edges, note_locations)
        perform_z = [perform_z]

    return perform_z, qpm_primo


#>>>>>>>>>>>>>> maybe to be removed
def load_all_file_and_encode_style(parsed_data, measure_only=False, emotion_data=False):
    total_z = []
    perf_name_list = []
    num_piece = len(parsed_data[0])
    for i in range(num_piece):
        piece_test_x = parsed_data[0][i]
        piece_test_y = parsed_data[1][i]
        piece_edges = parsed_data[2][i]
        piece_note_locations = parsed_data[3][i]
        piece_perf_name = parsed_data[4][i]
        num_perf = len(piece_test_x)
        if num_perf == 0:
            continue
        piece_z = []
        for j in range(num_perf):
            # test_x, test_y, edges, note_locations, perf_name = perf
            if measure_only:
                test_x, test_y = handle_data_in_tensor(
                    piece_test_x[j], piece_test_y[j], hierarchy_test=IN_HIER)
                edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
                test_x = test_x.view((1, -1, HIER_MODEL.input_size))
                hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
                perform_z_high = encode_performance_style_vector(
                    test_x, hier_y, edges, piece_note_locations[j], model=HIER_MODEL)
            else:
                test_x, test_y = handle_data_in_tensor(
                    piece_test_x[j], piece_test_y[j], hierarchy_test=False)
                edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
                test_x = test_x.view((1, -1, MODEL.input_size))
                test_y = test_y.view(1, -1, MODEL.output_size)
                perform_z_high = encode_performance_style_vector(test_x, test_y, edges, piece_note_locations[j],
                                                                 model=MODEL)
            # perform_z_high = perform_z_high.reshape(-1).cpu().numpy()
            # piece_z.append(perform_z_high)
            # perf_name_list.append(piece_perf_name[j])

            perform_z_high = [z.reshape(-1).cpu().numpy()
                              for z in perform_z_high]
            piece_z += perform_z_high
            perf_name_list += [piece_perf_name[j]] * len(perform_z_high)
        if emotion_data:
            for i, name in enumerate(piece_perf_name):
                if name[-2:] == 'E1':
                    or_idx = i
                    break
            or_z = piece_z.pop(or_idx)
            piece_z = np.asarray(piece_z)
            piece_z -= or_z
            perf_name_list.pop(-(5-or_idx))
        else:
            piece_z = np.asarray(piece_z)
            average_piece_z = np.average(piece_z, axis=0)
            piece_z -= average_piece_z
        total_z.append(piece_z)
    total_z = np.concatenate(total_z, axis=0)
    return total_z, perf_name_list
#<<<<<<<<<<<<<< 


def encode_all_emotionNet_data(path_list, style_keywords):
    perform_z_by_emotion = []
    perform_z_list_by_subject = []
    qpm_list_by_subject = []
    num_style = len(style_keywords)
    if IN_HIER:
        num_model = 2
    else:
        num_model = 1
    for pair in path_list:
        subject_num = pair[2]
        for sub_idx in range(subject_num):
            indiv_perform_z = []
            indiv_qpm = []
            path = cons.emotion_folder_path + pair[0] + '/'
            composer_name = pair[1]
            for key in style_keywords:
                perf_name = key + '_sub' + str(sub_idx+1)
                perform_z_li, qpm_primo = load_file_and_encode_style(
                    path, perf_name, composer_name)
                perform_z_li = [torch.mean(torch.stack(z), 0, True)
                                for z in perform_z_li]
                indiv_perform_z.append(perform_z_li)
                indiv_qpm.append(qpm_primo)
            for i in range(1, num_style):
                for j in range(num_model):
                    indiv_perform_z[i][j] = indiv_perform_z[i][j] - \
                        indiv_perform_z[0][j]
                indiv_qpm[i] = indiv_qpm[i] - indiv_qpm[0]
            perform_z_list_by_subject.append(indiv_perform_z)
            qpm_list_by_subject.append(indiv_qpm)
    for i in range(num_style):
        z_by_models = []
        for j in range(num_model):
            emotion_mean_z = []
            for z_list in perform_z_list_by_subject:
                emotion_mean_z.append(z_list[i][j])
            mean_perform_z = torch.mean(torch.stack(emotion_mean_z), 0, True)
            z_by_models.append(mean_perform_z)
        if i is not 0:
            emotion_qpm = []
            for qpm_change in qpm_list_by_subject:
                emotion_qpm.append(qpm_change[i])
            mean_qpm_change = np.mean(emotion_qpm)
        else:
            mean_qpm_change = 0
        print(style_keywords[i], z_by_models, mean_qpm_change)
        perform_z_by_emotion.append(
            {'z': z_by_models, 'key': style_keywords[i], 'qpm': mean_qpm_change})

    return perform_z_by_emotion
    # with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
    #     pickle.dump(mean_perform_z, f, protocol=2)


def load_stat(args):
    with open(args.dataName + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        if args.trainingLoss == 'CE':
            MEANS, STDS, BINS = u.load()
            new_prime_param = 0
            new_trill_param = 0
            for i in range(NUM_PRIME_PARAM):
                new_prime_param += len(BINS[i]) - 1
            for i in range(NUM_PRIME_PARAM, NUM_PRIME_PARAM + num_trill_param - 1):
                new_trill_param += len(BINS[i]) - 1
            NUM_PRIME_PARAM = new_prime_param
            print('New NUM_PRIME_PARAM: ', NUM_PRIME_PARAM)
            num_trill_param = new_trill_param + 1
            NUM_OUTPUT = NUM_PRIME_PARAM + num_trill_param
            NUM_TEMPO_PARAM = len(BINS[0]) - 1
        else:
            MEANS, STDS = u.load()
            BINS = None
    return MEANS, STDS, BINS


def read_xml_to_array(path_name, means, stds, start_tempo, composer_name, vel_standard):
    # TODO: update to adapt pyScoreParser
    xml_object, xml_notes = xml_matching.read_xml_to_notes(path_name)
    beats = xml_object.get_beat_positions()
    measure_positions = xml_object.get_measure_positions()
    features = xml_matching.extract_score_features(
        xml_notes, measure_positions, beats, qpm_primo=start_tempo, vel_standard=vel_standard)
    features = make_index_continuous(features, score=True)
    composer_vec = composer_name_to_vec(composer_name)
    edges = score_graph.make_edge(xml_notes)

    for i in range(len(stds[0])):
        if stds[0][i] < 1e-4 or isinstance(stds[0][i], complex):
            stds[0][i] = 1

    test_x = []
    note_locations = []
    for feat in features:
        temp_x = [(feat.midi_pitch - means[0][0]) / stds[0][0], (feat.duration - means[0][1]) / stds[0][1],
                  (feat.beat_importance -
                   means[0][2])/stds[0][2], (feat.measure_length-means[0][3])/stds[0][3],
                  (feat.qpm_primo - means[0][4]) /
                  stds[0][4], (feat.following_rest - means[0][5]) / stds[0][5],
                  (feat.distance_from_abs_dynamic - means[0][6]) / stds[0][6],
                  (feat.distance_from_recent_tempo - means[0][7]) / stds[0][7],
                  feat.beat_position, feat.xml_position, feat.grace_order,
                  feat.preceded_by_grace_note, feat.followed_by_fermata_rest] \
            + feat.pitch + feat.tempo + feat.dynamic + feat.time_sig_vec + \
            feat.slur_beam_vec + composer_vec + feat.notation + feat.tempo_primo
        # temp_x.append(feat.is_beat)
        test_x.append(temp_x)
        note_locations.append(feat.note_location)

    return test_x, xml_notes, xml_object, edges, note_locations


class PerformDataset():
    def __init__(self, data_path, split, graph=False, samples=None):
        return
    def __len__(self):
        return NumberOfPieces

    def files(self):
        return NotImplementedError

class YamahaDataset(PerformDataset):
    def __init__(self, data_path, split, graph=False, samples=None):
        return
    def __getitem__(self, index):
        return input_features, output_features, score_graph

    def __len__(self):
        return NumberOfSegments
    
    def files(self):
        # load yamaha set data utilize pyScoreParser.PieceData
        return NotImplementedError



def load_file_and_generate_performance(model, path_name, args, trill_model, hier_model=None, return_features=False):
    composer=args.composer
    z=args.latent
    start_tempo=args.startTempo
    path_name = Path(path_name)
    if path_name.suffix not in ('xml', 'musicxml'):
        path_name = path_name / 'xml.xml'
        if not path_name.exists():
            path_name = path_name.parent / 'musicxml_cleaned.musicxml'

    score_features = model.score_feature_keys
    score_data = ScoreData(path_name)
    feature_extractor = ScoreExtractor()
    test_x = feature_extractor.extract_score_features(score_data)
    test_x, xml_notes, xml_doc, edges, note_locations = xml_matching.read_xml_to_array(path_name, means, stds,
                                                                                       start_tempo, composer,
                                                                                       vel_pair)
    batch_x = torch.Tensor(test_x)
    num_notes = len(test_x)
    input_y = torch.zeros(1, num_notes, NUM_OUTPUT).to(DEVICE)

    if type(z) is dict:
        initial_z = z['z']
        qpm_change = z['qpm']
        z = z['key']
        batch_x[:, QPM_PRIMO_IDX] = batch_x[:, QPM_PRIMO_IDX] + qpm_change
    else:
        initial_z = 'zero'

    if IN_HIER:
        batch_x = batch_x.to(DEVICE).view(1, -1, hier_model.input_size)
        graph = edges_to_matrix(edges, batch_x.shape[1])
        MODEL.is_teacher_force = False
        if type(initial_z) is list:
            hier_z = initial_z[0]
            final_z = initial_z[1]
        else:
            # hier_z = [z] * HIER_MODEL_PARAM.encoder.size
            hier_z = 'zero'
            final_z = initial_z
        hier_input_y = torch.zeros(1, num_notes, hier_model.output_size)
        hier_output, _ = run_model_in_steps(
            batch_x, hier_input_y, graph, note_locations, initial_z=hier_z, model=hier_model)
        if 'measure' in args.hierCode:
            hierarchy_numbers = [x.measure for x in note_locations]
        else:
            hierarchy_numbers = [x.section for x in note_locations]
        hier_output_spanned = hier_model.span_beat_to_note_num(
            hier_output, hierarchy_numbers, len(test_x), 0)
        combined_x = torch.cat((batch_x, hier_output_spanned), 2)
        prediction, _ = run_model_in_steps(
            combined_x, input_y, graph, note_locations, initial_z=final_z, model=model)
    else:
        if type(initial_z) is list:
            initial_z = initial_z[0]
        batch_x = batch_x.to(DEVICE).view(1, -1, NUM_INPUT)
        graph = edges_to_matrix(edges, batch_x.shape[1])
        prediction, _ = run_model_in_steps(
            batch_x, input_y, graph, note_locations, initial_z=initial_z, model=model)

    trill_batch_x = torch.cat((batch_x, prediction), 2)
    trill_prediction, _ = run_model_in_steps(trill_batch_x, torch.zeros(
        1, num_notes, cons.num_trill_param), graph, note_locations, model=trill_model)

    prediction = torch.cat((prediction, trill_prediction), 2)
    prediction = scale_model_prediction_to_original(prediction, means, stds)

    output_features = xml_matching.model_prediction_to_feature(prediction)
    output_features = xml_matching.add_note_location_to_features(
        output_features, note_locations)
    if return_features:
        return output_features

    output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                           predicted=True)
    output_midi, midi_pedals = xml_matching.xml_notes_to_midi(output_xml)
    piece_name = path_name.split('/')
    save_name = 'test_result/' + \
        piece_name[-2] + '_by_' + args.modelCode + '_z' + str(z)

    perf_worm.plot_performance_worm(output_features, save_name + '.png')
    xml_matching.save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_name + '.mid',
                                               bool_pedal=args.boolPedal, disklavier=args.disklavier)




def test(args,
         model,
         TRILL_model,
         device,
         param):
    # TODO: seperate validation / test / inference.
    if os.path.isfile('prime_' + args.modelCode + args.resume):
        print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
        # model_codes = ['prime', 'trill']
        filename = 'prime_' + args.modelCode + args.resume
        print('device is ', args.device)
        th.cuda.set_device(args.device)
        if th.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        checkpoint = th.load(filename, map_location=map_location)
        # args.start_epoch = checkpoint['epoch']
        # best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['state_dict'])
        # model.num_graph_iteration = 10
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        # NUM_UPDATED = checkpoint['training_step']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # trill_filename = args.trillCode + args.resume
        trill_filename = args.trillCode + '_best.pth.tar'
        checkpoint = th.load(trill_filename, map_location=map_location)
        TRILL_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(trill_filename, checkpoint['epoch']))

        if args.in_hier:
            HIER_model_PARAM = param.load_parameters(args.hierCode + '_param')
            HIER_model = model.HAN_Integrated(HIER_model_PARAM, device, True).to(device)
            filename = 'prime_' + args.hierCode + args.resume
            checkpoint = th.load(filename, map_location=device)
            HIER_model.load_state_dict(checkpoint['state_dict'])
            print("=> high-level model loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    model.is_teacher_force = False
    

def inference_old(model, args):
    # Suggestion: move inference-like mode to inference.py
    if args.sessMode == 'test':
        random.seed(0)
        inference.load_file_and_generate_performance(args.testPath, args)
    elif args.sessMode=='testAll':
        path_list = const.emotion_data_path
        emotion_list = const.emotion_key_list
        perform_z_by_list = dataset.encode_all_emotionNet_data(path_list, emotion_list)
        test_list = const.test_piece_list
        for piece in test_list:
            path = './test_pieces/' + piece[0] + '/'
            composer = piece[1]
            if len(piece) == 3:
                start_tempo = piece[2]
            else:
                start_tempo = 0
            for perform_z_pair in perform_z_by_list:
                inference.load_file_and_generate_performance(path, composer, z=perform_z_pair, start_tempo=start_tempo)
            inference.load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)
    elif args.sessMode == 'testAllzero':
        test_list = const.test_piece_list
        for piece in test_list:
            path = './test_pieces/' + piece[0] + '/'
            composer = piece[1]
            if len(piece) == 3:
                start_tempo = piece[2]
            else:
                start_tempo = 0
            random.seed(0)
            inference.load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)

    elif args.sessMode == 'encode':
        perform_z, qpm_primo = dataset.load_file_and_encode_style(args.testPath, args.perfName, args.composer)
        print(perform_z)
        with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
            pickle.dump(perform_z, f, protocol=2)

    elif args.sessMode =='evaluate':
        test_data_name = args.dataName + "_test.dat"
        if not os.path.isfile(test_data_name):
            test_data_name = '/mnt/ssd1/jdasam_data/' + test_data_name
        with open(test_data_name, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            # p = u.load()
            # complete_xy = pickle.load(f)
            complete_xy = u.load()

        tempo_loss_total = []
        vel_loss_total = []
        deviation_loss_total = []
        trill_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        kld_total = []

        prev_perf_x = complete_xy[0][0]
        prev_perfs_worm_data = []
        prev_reconstructed_worm_data = []
        prev_zero_predicted_worm_data = []
        piece_wise_loss = []
        human_correlation_total = []
        human_correlation_results = xml_matching.CorrelationResult()
        model_correlation_total = []
        model_correlation_results = xml_matching.CorrelationResult()
        zero_sample_correlation_total = []
        zero_sample_correlation_results= xml_matching.CorrelationResult()



        for xy_tuple in complete_xy:
            current_perf_index = complete_xy.index(xy_tuple)
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]
            edges = xy_tuple[5]
            graphs = graph.edges_to_matrix(edges, len(test_x))
            if args.loss == 'CE':
                test_y = categorize_value_to_vector(test_y, bins)

            if xml_matching.check_feature_pair_is_from_same_piece(prev_perf_x, test_x):
                piece_changed = False
                # current_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(test_y, note_locations=note_locations, momentum=0.2)
                # for prev_worm in prev_perfs_worm_data:
                #     tempo_r, _ = xml_matching.cal_correlation(current_perf_worm_data[0], prev_worm[0])
                #     dynamic_r, _ = xml_matching.cal_correlation(current_perf_worm_data[1], prev_worm[1])
                #     human_correlation_results.append_result(tempo_r, dynamic_r)
                # prev_perfs_worm_data.append(current_perf_worm_data)
            else:
                piece_changed = True

            if piece_changed or current_perf_index == len(complete_xy)-1:
                prev_perf_x = test_x
                if piece_wise_loss:
                    piece_wise_loss_mean = np.mean(np.asarray(piece_wise_loss), axis=0)
                    tempo_loss_total.append(piece_wise_loss_mean[0])
                    vel_loss_total.append(piece_wise_loss_mean[1])
                    deviation_loss_total.append(piece_wise_loss_mean[2])
                    articul_loss_total.append(piece_wise_loss_mean[3])
                    pedal_loss_total.append(piece_wise_loss_mean[4])
                    trill_loss_total.append(piece_wise_loss_mean[5])
                    kld_total.append(piece_wise_loss_mean[6])
                piece_wise_loss = []

                # human_correlation_total.append(human_correlation_results)
                # human_correlation_results = xml_matching.CorrelationResult()
                #
                # for predict in prev_reconstructed_worm_data:
                #     for human in prev_perfs_worm_data:
                #         tempo_r, _ = xml_matching.cal_correlation(predict[0], human[0])
                #         dynamic_r, _ = xml_matching.cal_correlation(predict[1], human[1])
                #         model_correlation_results.append_result(tempo_r, dynamic_r)
                #
                # model_correlation_total.append(model_correlation_results)
                # model_correlation_results = xml_matching.CorrelationResult()
                #
                # for zero in prev_zero_predicted_worm_data:
                #     for human in prev_perfs_worm_data:
                #         tempo_r, _ = xml_matching.cal_correlation(zero[0], human[0])
                #         dynamic_r, _ = xml_matching.cal_correlation(zero[1], human[1])
                #         zero_sample_correlation_results.append_result(tempo_r, dynamic_r)
                #
                # zero_sample_correlation_total.append(zero_sample_correlation_results)
                # zero_sample_correlation_results = xml_matching.CorrelationResult()
                #
                # prev_reconstructed_worm_data = []
                # prev_zero_predicted_worm_data = []
                # prev_perfs_worm_data = []
                #
                # print('Human Correlation: ', human_correlation_total[-1])
                # print('Reconst Correlation: ', model_correlation_total[-1])
                # print('Zero Sampled Correlation: ', zero_sample_correlation_total[-1])

            batch_x, batch_y = handle_data_in_tensor(test_x, test_y, hierarchy_test=IN_HIER)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(device)
            pedal_status = th.Tensor(pedal_status).view(1, -1, 1).to(device)

            if IN_HIER:
                batch_x = batch_x.view((1, -1, HIER_model.input_size))
                hier_y = batch_y[0].view(1, -1, HIER_model.output_size)
                hier_outputs, _ = run_model_in_steps(batch_x, hier_y, graphs, note_locations, model=HIER_model)
                if HIER_MEAS:
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif HIER_BEAT:
                    hierarchy_numbers = [x.beat for x in note_locations]
                hier_outputs_spanned = HIER_model.span_beat_to_note_num(hier_outputs, hierarchy_numbers, batch_x.shape[1], 0)
                input_concat = th.cat((batch_x, hier_outputs_spanned),2)
                batch_y = batch_y[1].view(1,-1, model.output_size)
                outputs, perform_z = run_model_in_steps(input_concat, batch_y, graphs, note_locations, model=model)

                # make another prediction with random sampled z
                zero_hier_outputs, _ = run_model_in_steps(batch_x, hier_y, graphs, note_locations, model=HIER_model,
                                                        initial_z='zero')
                zero_hier_spanned = HIER_model.span_beat_to_note_num(zero_hier_outputs, hierarchy_numbers, batch_x.shape[1], 0)
                zero_input_concat = th.cat((batch_x, zero_hier_spanned),2)
                zero_prediction, _ = run_model_in_steps(zero_input_concat, batch_y, graphs, note_locations, model=model)

            else:
                batch_x = batch_x.view((1, -1, NUM_INPUT))
                batch_y = batch_y.view((1, -1, NUM_OUTPUT))
                outputs, perform_z = run_model_in_steps(batch_x, batch_y, graphs, note_locations)

                # make another prediction with random sampled z
                zero_prediction, _ = run_model_in_steps(batch_x, batch_y, graphs, note_locations, model=model,
                                                     initial_z='zero')

            output_as_feature = outputs.view(-1, NUM_OUTPUT).cpu().numpy()
            predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(output_as_feature, note_locations,
                                                                                momentum=0.2)
            zero_prediction_as_feature = zero_prediction.view(-1, NUM_OUTPUT).cpu().numpy()
            zero_predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(zero_prediction_as_feature, note_locations,
                                                                                     momentum=0.2)

            prev_reconstructed_worm_data.append(predicted_perf_worm_data)
            prev_zero_predicted_worm_data.append(zero_predicted_perf_worm_data)

            # for prev_worm in prev_perfs_worm_data:
            #     tempo_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[0], prev_worm[0])
            #     dynamic_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[1], prev_worm[1])
            #     model_correlation_results.append_result(tempo_r, dynamic_r)
            # print('Model Correlation: ', model_correlation_results)

            # valid_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], batch_y[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], align_matched)
            if model.is_baseline:
                tempo_loss = criterion(outputs[:, :, 0], batch_y[:, :, 0], align_matched)
            else:
                tempo_loss = cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
            if args.loss == 'CE':
                vel_loss = criterion(outputs[:, :, const.NUM_TEMPO_PARAM:const.NUM_TEMPO_PARAM + len(bins[1])],
                                     batch_y[:, :, const.NUM_TEMPO_PARAM:const.NUM_TEMPO_PARAM + len(bins[1])], align_matched)
                deviation_loss = criterion(
                    outputs[:, :, const.NUM_TEMPO_PARAM + len(bins[1]):const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2])],
                    batch_y[:, :, const.NUM_TEMPO_PARAM + len(bins[1]):const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2])])
                pedal_loss = criterion(
                    outputs[:, :, const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2]):-const.num_trill_param],
                    batch_y[:, :, const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2]):-const.num_trill_param])
                trill_loss = criterion(outputs[:, :, -const.num_trill_param:], batch_y[:, :, -const.num_trill_param:])
            else:
                vel_loss = criterion(outputs[:, :, const.VEL_PARAM_IDX], batch_y[:, :, const.VEL_PARAM_IDX], align_matched)
                deviation_loss = criterion(outputs[:, :, const.DEV_PARAM_IDX], batch_y[:, :, const.DEV_PARAM_IDX],
                                           align_matched)
                articul_loss = criterion(outputs[:, :, const.PEDAL_PARAM_IDX], batch_y[:, :, const.PEDAL_PARAM_IDX],
                                         pedal_status)
                pedal_loss = criterion(outputs[:, :, const.PEDAL_PARAM_IDX + 1:], batch_y[:, :, const.PEDAL_PARAM_IDX + 1:],
                                       align_matched)
                trill_loss = th.zeros(1)

            piece_kld = []
            for z in perform_z:
                perform_mu, perform_var = z
                kld = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                piece_kld.append(kld)
            piece_kld = th.mean(th.stack(piece_kld))

            piece_wise_loss.append((tempo_loss.item(), vel_loss.item(), deviation_loss.item(), articul_loss.item(), pedal_loss.item(), trill_loss.item(), piece_kld.item()))



        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_articul_loss = np.mean(articul_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)
        mean_kld_loss = np.mean(kld_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss / 2 + mean_pedal_loss * 8) / 10.5

        print("Test Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss, mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss, mean_kld_loss))
        # num_piece = len(model_correlation_total)
        # for i in range(num_piece):
        #     if len(human_correlation_total) > 0:
        #         print('Human Correlation: ', human_correlation_total[i])
        #         print('Model Correlation: ', model_correlation_total[i])

    elif args.sessMode == 'correlation':
        with open('selected_corr_30.dat', "rb") as f:
            u = pickle._Unpickler(f)
            selected_corr = u.load()
        model_cor = []
        for piece_corr in selected_corr:
            if piece_corr is None or piece_corr==[]:
                continue
            path = piece_corr[0].path_name
            composer_name = copy.copy(path).split('/')[1]
            output_features = load_file_and_generate_performance(path, composer_name, 'zero', return_features=True)
            for slice_corr in piece_corr:
                slc_idx = slice_corr.slice_index
                sliced_features = output_features[slc_idx[0]:slc_idx[1]]
                tempos, dynamics = perf_worm.cal_tempo_and_velocity_by_beat(sliced_features)
                model_correlation_results = xml_matching.CorrelationResult()
                model_correlation_results.path_name = slice_corr.path_name
                model_correlation_results.slice_index = slice_corr.slice_index
                human_tempos = slice_corr.tempo_features
                human_dynamics = slice_corr.dynamic_features
                for i in range(slice_corr.num_performance):
                    tempo_r, _ = xml_matching.cal_correlation(tempos, human_tempos[i])
                    dynamic_r, _ = xml_matching.cal_correlation(dynamics, human_dynamics[i])
                    model_correlation_results._append_result(tempo_r, dynamic_r)
                print(model_correlation_results)
                model_correlation_results.tempo_features = copy.copy(slice_corr.tempo_features)
                model_correlation_results.dynamic_features = copy.copy(slice_corr.dynamic_features)
                model_correlation_results.tempo_features.append(tempos)
                model_correlation_results.dynamic_features.append(dynamics)

                save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_', 10) + '_note{}-{}_by_{}.png'.format(slc_idx[0], slc_idx[1], args.modelCode)
                perf_worm.plot_human_model_features_compare(model_correlation_results.tempo_features, save_name)
                model_cor.append(model_correlation_results)

        with open(args.modelCode + "_cor.dat", "wb") as f:
            pickle.dump(model_cor, f, protocol=2)