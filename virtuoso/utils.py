import th as th
import shutil
import model_constants as const
from . import data_process as dp


def save_checkpoint(state, is_best, filename='isgn', model_name='prime'):
    save_name = model_name + '_' + filename + '_checkpoint.pth.tar'
    th.save(state, save_name)
    if is_best:
        best_name = model_name + '_' + filename + '_best.pth.tar'
        shutil.copyfile(save_name, best_name)


def encode_performance_style_vector(input, input_y, edges, note_locations, device, model):
    with th.no_grad():
        model_eval = model.eval()
        if edges is not None:
            edges = edges.to(device)
        encoded_z = model_eval(input, input_y, edges,
                               note_locations=note_locations, start_index=0, return_z=True)
    return encoded_z


def run_model_in_steps(input, input_y, args, edges, note_locations, model, device, initial_z=False):
    num_notes = input.shape[1]
    with th.no_grad():  # no need to track history in validation
        model_eval = model.eval()
        total_output = []
        total_z = []
        measure_numbers = [x.measure for x in note_locations]
        slice_indexes = dp.make_slicing_indexes_by_measure(
            num_notes, measure_numbers, steps=args.valid_steps, overlap=False)
        if edges is not None:
            edges = edges.to(device)

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

        outputs = th.cat(total_output, 1)
        return outputs, total_z


def batch_time_step_run(data, model, batch_size=batch_size):
    batch_start, batch_end = training_data['slice_idx']
    batch_x, batch_y = handle_data_in_tensor(
        data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

    batch_x = batch_x.view((batch_size, -1, NUM_INPUT))
    batch_y = batch_y.view((batch_size, -1, NUM_OUTPUT))

    align_matched = th.Tensor(data['align_matched'][batch_start:batch_end]).view(
        (batch_size, -1, 1)).to(DEVICE)
    pedal_status = th.Tensor(data['pedal_status'][batch_start:batch_end]).view(
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
        if th.sum(trill_bool) > 0:
            total_loss = criterion(outputs, batch_y, trill_bool)
        else:
            return th.zeros(1), th.zeros(1), th.zeros(1),  th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1)

    else:
        if 'isgn' in args.modelCode and args.intermediateLoss:
            total_loss = th.zeros(1).to(DEVICE)
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
        perform_kld = th.zeros(1)
    else:
        perform_kld = -0.5 * \
            th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
        total_loss += perform_kld * kld_weight
    optimizer.zero_grad()
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if HIERARCHY:
        return tempo_loss, vel_loss, th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), perform_kld
    elif TRILL:
        return th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), total_loss, th.zeros(1)
    else:
        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, th.zeros(1), perform_kld

    # loss = criterion(outputs, batch_y)
    # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])


def categorize_value_to_vector(y, bins):
    vec_length = sum([len(x) for x in bins])
    num_notes = len(y)
    y_categorized = []
    num_categorized_params = len(bins)
    for i in range(num_notes):
        note = y[i]
        total_vec = []
        for j in range(num_categorized_params):
            temp_vec = [0] * (len(bins[j]) - 1)
            temp_vec[int(note[j])] = 1
            total_vec += temp_vec
        total_vec.append(note[-1])  # add up trill
        y_categorized.append(total_vec)

    return y_categorized


def handle_data_in_tensor(x, y, args, DEVICE, hierarchy_test=False):
    x = th.Tensor(x)
    y = th.Tensor(y)
    if args.hier_meas:
        hierarchy_output = y[:, const.MEAS_TEMPO_IDX:const.MEAS_TEMPO_IDX+2]
    elif y:
        hierarchy_output = y[:, const.BEAT_TEMPO_IDX:const.BEAT_TEMPO_IDX+2]

    if hierarchy_test:
        y = y[:, :const.NUM_PRIME_PARAM]
        return x.to(DEVICE), (hierarchy_output.to(DEVICE), y.to(DEVICE))

    if args.hierarchy:
        y = hierarchy_output
    elif args.in_hier:
        x = th.cat((x, hierarchy_output), 1)
        y = y[:, :const.NUM_PRIME_PARAM]
    elif args.trill:
        x = th.cat((x, y[:, :const.NUM_PRIME_PARAM]), 1)
        y = y[:, -const.NUM_TRILL_PARAM:]
    else:
        y = y[:, :const.NUM_PRIME_PARAM]

    return x.to(DEVICE), y.to(DEVICE)


def cal_tempo_loss_in_beat(pred_x, true_x, note_locations, start_index, qpm_idx, criterion, args, device):
    previous_beat = -1

    num_notes = pred_x.shape[1]
    start_beat = note_locations[start_index].beat
    num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1

    pred_beat_tempo = th.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(device)
    true_beat_tempo = th.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(device)
    for i in range(num_notes):
        current_beat = note_locations[i+start_index].beat
        if current_beat > previous_beat:
            previous_beat = current_beat
            if 'baseline' in args.modelCode:
                for j in range(i, num_notes):
                    if note_locations[j+start_index].beat > current_beat:
                        break
                if not i == j:
                    pred_beat_tempo[current_beat - start_beat] = th.mean(pred_x[0, i:j, qpm_idx])
                    true_beat_tempo[current_beat - start_beat] = th.mean(true_x[0, i:j, qpm_idx])
            else:
                pred_beat_tempo[current_beat-start_beat] = pred_x[0,i,qpm_idx:qpm_idx + const.NUM_TEMPO_PARAM]
                true_beat_tempo[current_beat-start_beat] = true_x[0,i,qpm_idx:qpm_idx + const.NUM_TEMPO_PARAM]

    tempo_loss = criterion(pred_beat_tempo, true_beat_tempo)
    if args.deltaLoss and pred_beat_tempo.shape[0] > 1:
        prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
        true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
        delta_loss = criterion(prediction_delta, true_delta)

        tempo_loss = (tempo_loss + delta_loss * args.delta_weight) / (1 + args.delta_weight)

    return tempo_loss