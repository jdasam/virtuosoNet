import torch as th
import shutil


def save_checkpoint(state, is_best, filename='isgn', model_name='prime'):
    save_name = model_name + '_' + filename + '_checkpoint.pth.tar'
    th.save(state, save_name)
    if is_best:
        best_name = model_name + '_' + filename + '_best.pth.tar'
        shutil.copyfile(save_name, best_name)


def encode_performance_style_vector(input, input_y, edges, note_locations, model=MODEL):
    with th.no_grad():
        model_eval = model.eval()
        if edges is not None:
            edges = edges.to(DEVICE)
        encoded_z = model_eval(input, input_y, edges,
                               note_locations=note_locations, start_index=0, return_z=True)
    return encoded_z


def run_model_in_steps(input, input_y, edges, note_locations, initial_z=False, model=MODEL):
    num_notes = input.shape[1]
    with torch.no_grad():  # no need to track history in validation
        model_eval = model.eval()
        total_output = []
        total_z = []
        measure_numbers = [x.measure for x in note_locations]
        slice_indexes = dp.make_slicing_indexes_by_measure(
            num_notes, measure_numbers, steps=VALID_STEPS, overlap=False)
        if edges is not None:
            edges = edges.to(DEVICE)

        for slice_idx in slice_indexes:
            batch_start, batch_end = slice_idx
            if edges is not None:
                batch_graph = edges[:, batch_start:batch_end,
                                    batch_start:batch_end]
            else:
                batch_graph = None
            batch_input = input[:, batch_start:batch_end, :].view(
                1, -1, model.input_size)
            batch_input_y = input_y[:, batch_start:batch_end, :].view(
                1, -1, model.output_size)
            temp_outputs, perf_mu, perf_var, _ = model_eval(batch_input, batch_input_y, batch_graph,
                                                            note_locations=note_locations, start_index=batch_start, initial_z=initial_z)
            total_z.append((perf_mu, perf_var))
            total_output.append(temp_outputs)

        outputs = torch.cat(total_output, 1)
        return outputs, total_z


def batch_time_step_run(data, model, batch_size=batch_size):
    warnings.warn('moved to utils.py', DeprecationWarning)
    batch_start, batch_end = training_data['slice_idx']
    batch_x, batch_y = handle_data_in_tensor(
        data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

    batch_x = batch_x.view((batch_size, -1, NUM_INPUT))
    batch_y = batch_y.view((batch_size, -1, NUM_OUTPUT))

    align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end]).view(
        (batch_size, -1, 1)).to(DEVICE)
    pedal_status = torch.Tensor(data['pedal_status'][batch_start:batch_end]).view(
        (batch_size, -1, 1)).to(DEVICE)

    if training_data['graphs'] is not None:
        edges = training_data['graphs']
        if edges.shape[1] == batch_end - batch_start:
            edges = edges.to(DEVICE)
        else:
            edges = edges[:, batch_start:batch_end,
                          batch_start:batch_end].to(DEVICE)
    else:
        edges = training_data['graphs']

    prime_batch_x = batch_x
    if HIERARCHY:
        prime_batch_y = batch_y
    else:
        prime_batch_y = batch_y[:, :, 0:NUM_PRIME_PARAM]

    model_train = model.train()
    outputs, perform_mu, perform_var, total_out_list \
        = model_train(prime_batch_x, prime_batch_y, edges, note_locations, batch_start)

    if HIERARCHY:
        if HIER_MEAS:
            hierarchy_numbers = [x.measure for x in note_locations]
        elif HIER_BEAT:
            hierarchy_numbers = [x.beat for x in note_locations]
        tempo_in_hierarchy = MODEL.note_tempo_infos_to_beat(
            batch_y, hierarchy_numbers, batch_start, 0)
        dynamics_in_hierarchy = MODEL.note_tempo_infos_to_beat(
            batch_y, hierarchy_numbers, batch_start, 1)
        tempo_loss = criterion(outputs[:, :, 0:1], tempo_in_hierarchy)
        vel_loss = criterion(outputs[:, :, 1:2], dynamics_in_hierarchy)
        if args.deltaLoss and outputs.shape[1] > 1:
            vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
            vel_true_delta = dynamics_in_hierarchy[:,
                                                   1:, :] - dynamics_in_hierarchy[:, :-1, :]

            vel_loss += criterion(vel_out_delta, vel_true_delta) * DELTA_WEIGHT
            vel_loss /= 1 + DELTA_WEIGHT

        total_loss = tempo_loss + vel_loss
    elif TRILL:
        trill_bool = batch_x[:, :,
                             is_trill_index_concated:is_trill_index_concated + 1]
        if torch.sum(trill_bool) > 0:
            total_loss = criterion(outputs, batch_y, trill_bool)
        else:
            return torch.zeros(1), torch.zeros(1), torch.zeros(1),  torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

    else:
        if 'isgn' in args.modelCode and args.intermediateLoss:
            total_loss = torch.zeros(1).to(DEVICE)
            for out in total_out_list:
                if model.is_baseline:
                    tempo_loss = criterion(out[:, :, 0:1],
                                           prime_batch_y[:, :, 0:1], align_matched)
                else:
                    tempo_loss = cal_tempo_loss_in_beat(
                        out, prime_batch_y, note_locations, batch_start)
                vel_loss = criterion(out[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX],
                                     prime_batch_y[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX], align_matched)
                dev_loss = criterion(out[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX],
                                     prime_batch_y[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX], align_matched)
                articul_loss = criterion(out[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX+1],
                                         prime_batch_y[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX+1], pedal_status)
                pedal_loss = criterion(out[:, :, PEDAL_PARAM_IDX+1:], prime_batch_y[:, :, PEDAL_PARAM_IDX+1:],
                                       align_matched)

                total_loss += (tempo_loss + vel_loss + dev_loss +
                               articul_loss + pedal_loss * 7) / 11
            total_loss /= len(total_out_list)
        else:
            if model.is_baseline:
                tempo_loss = criterion(outputs[:, :, 0:1],
                                       prime_batch_y[:, :, 0:1], align_matched)
            else:
                tempo_loss = cal_tempo_loss_in_beat(
                    outputs, prime_batch_y, note_locations, batch_start)
            vel_loss = criterion(outputs[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX],
                                 prime_batch_y[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX], align_matched)
            dev_loss = criterion(outputs[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX],
                                 prime_batch_y[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX], align_matched)
            articul_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX + 1],
                                     prime_batch_y[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX + 1], pedal_status)
            pedal_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX + 1:], prime_batch_y[:, :, PEDAL_PARAM_IDX + 1:],
                                   align_matched)
            total_loss = (tempo_loss + vel_loss + dev_loss +
                          articul_loss + pedal_loss * 7) / 11

    if isinstance(perform_mu, bool):
        perform_kld = torch.zeros(1)
    else:
        perform_kld = -0.5 * \
            torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
        total_loss += perform_kld * kld_weight
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if HIERARCHY:
        return tempo_loss, vel_loss, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), perform_kld
    elif TRILL:
        return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), total_loss, torch.zeros(1)
    else:
        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), perform_kld

    # loss = criterion(outputs, batch_y)
    # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])
