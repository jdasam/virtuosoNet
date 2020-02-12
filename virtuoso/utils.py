import torch as th
import shutil
from . import model_constants as const
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


def batch_train_run(data, model, args, optimizer):
    batch_start, batch_end = data['slice_idx']
    batch_x, batch_y = handle_data_in_tensor(
        data['x'][batch_start:batch_end], data['y'][batch_start:batch_end], args, device=args.device)

    batch_x = batch_x.view((args.batch_size, -1, model.input_size))
    batch_y = batch_y.view((args.batch_size, -1, model.output_size))

    align_matched = th.Tensor(data['align_matched'][batch_start:batch_end]).view(
        (args.time_steps, -1, 1)).to(args.device)
    pedal_status = th.Tensor(data['pedal_status'][batch_start:batch_end]).view(
        (args.time_steps, -1, 1)).to(args.device)
    note_locations = data['note_locations'][batch_start:batch_end]

    if data['graphs'] is not None:
        edges = data['graphs']
        if edges.shape[1] == batch_end - batch_start:
            edges = edges.to(args.device)
        else:
            edges = edges[:, batch_start:batch_end,
                          batch_start:batch_end].to(args.device)
    else:
        edges = data['graphs']

    prime_batch_x = batch_x
    if model.is_hierarchy:
        prime_batch_y = batch_y
    else:
        prime_batch_y = batch_y[:, :, 0:const.NUM_PRIME_PARAM]

    model_train = model.train()
    outputs, perform_mu, perform_var, total_out_list \
        = model_train(prime_batch_x, prime_batch_y, edges, note_locations, batch_start)

    if model.config.hierarchy in ['measure', 'beat']:
        if model.is_hierarchy == 'measure':
            hierarchy_numbers = [x.measure for x in note_locations]
        elif model.is_hierarchy == 'beat':
            hierarchy_numbers = [x.beat for x in note_locations]
        tempo_in_hierarchy = model.note_tempo_infos_to_beat(
            batch_y, hierarchy_numbers, batch_start, 0)
        dynamics_in_hierarchy = model.note_tempo_infos_to_beat(
            batch_y, hierarchy_numbers, batch_start, 1)
        tempo_loss = criterion(outputs[:, :, 0:1], tempo_in_hierarchy, model.config)
        vel_loss = criterion(outputs[:, :, 1:2], dynamics_in_hierarchy, model.config)
        if args.deltaLoss and outputs.shape[1] > 1:
            vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
            vel_true_delta = dynamics_in_hierarchy[:,
                                                   1:, :] - dynamics_in_hierarchy[:, :-1, :]

            vel_loss += criterion(vel_out_delta, vel_true_delta, model.config) * args.delta_weight
            vel_loss /= 1 + args.delta_weight
        total_loss = tempo_loss + vel_loss
    elif model.config.is_trill:
        trill_bool = batch_x[:, :,
                             const.is_trill_index_concated:const.is_trill_index_concated + 1]
        if th.sum(trill_bool) > 0:
            total_loss = criterion(outputs, batch_y, trill_bool)
        else:
            return th.zeros(1), th.zeros(1), th.zeros(1),  th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1)

    else:
        if 'isgn' in args.modelCode and args.intermediateLoss:
            total_loss = th.zeros(1).to(args.device)
            for out in total_out_list:
                _, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss = \
                    cal_loss_by_output_type(out, prime_batch_y, align_matched, pedal_status, args,
                                            model.config, note_locations, batch_start)

                total_loss += (tempo_loss + vel_loss + dev_loss +
                               articul_loss + pedal_loss * 7) / 11
            total_loss /= len(total_out_list)
        else:
            total_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss =\
                cal_loss_by_output_type(outputs, prime_batch_y, align_matched, pedal_status, args,
                                        model.config, note_locations, batch_start)

    if isinstance(perform_mu, bool):
        perform_kld = th.zeros(1)
    else:
        perform_kld = -0.5 * \
            th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
        total_loss += perform_kld * data['kld_weight']
    optimizer.zero_grad()
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    if model.config.hierarchy in ['measure', 'beat']:
        return tempo_loss, vel_loss, th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), perform_kld
    elif model.config.is_trill:
        return th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), total_loss, th.zeros(1)
    else:
        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, th.zeros(1), perform_kld

    # loss = criterion(outputs, batch_y)
    # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])


def cal_loss_by_output_type(output, target_y, align_matched, pedal_weight, args, model_config, note_locations, batch_start):
    if model_config.is_baseline:
        tempo_loss = criterion(output[:, :, 0:1],
                               target_y[:, :, 0:1], model_config, align_matched)
    else:
        tempo_loss = cal_tempo_loss_in_beat(
            output, target_y, note_locations, batch_start, args, model_config)
    vel_loss = criterion(output[:, :, const.VEL_PARAM_IDX:const.DEV_PARAM_IDX],
                         target_y[:, :, const.VEL_PARAM_IDX:const.DEV_PARAM_IDX],model_config,  align_matched)
    dev_loss = criterion(output[:, :, const.DEV_PARAM_IDX:const.PEDAL_PARAM_IDX],
                         target_y[:, :, const.DEV_PARAM_IDX:const.PEDAL_PARAM_IDX],model_config,  align_matched)
    articul_loss = criterion(output[:, :, const.PEDAL_PARAM_IDX:const.PEDAL_PARAM_IDX + 1],
                             target_y[:, :, const.PEDAL_PARAM_IDX:const.PEDAL_PARAM_IDX + 1], model_config,  pedal_weight)
    pedal_loss = criterion(output[:, :, const.PEDAL_PARAM_IDX + 1:], target_y[:, :, const.PEDAL_PARAM_IDX + 1:], model_config,
                           align_matched)
    total_loss = (tempo_loss + vel_loss + dev_loss +
                  articul_loss + pedal_loss * 7) / 11

    return total_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss


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


def handle_data_in_tensor(x, y, model_config, device, hierarchy_test=False):
    x = th.Tensor(x)
    y = th.Tensor(y)
    if model_config.hierarchy == 'measure':
        hierarchy_output = y[:, const.MEAS_TEMPO_IDX:const.MEAS_TEMPO_IDX+2]
    elif model_config.hierarchy == 'beat':
        hierarchy_output = y[:, const.BEAT_TEMPO_IDX:const.BEAT_TEMPO_IDX+2]

    if hierarchy_test:
        y = y[:, :const.NUM_PRIME_PARAM]
        return x.to(device), (hierarchy_output.to(device), y.to(device))

    if model_config.hierarchy in ['measure', 'beat']:
        y = hierarchy_output
    elif model_config.hierarchy == 'note':
        x = th.cat((x, hierarchy_output), 1)
        y = y[:, :const.NUM_PRIME_PARAM]
    elif model_config.is_trill:
        x = th.cat((x, y[:, :const.NUM_PRIME_PARAM]), 1)
        y = y[:, -const.NUM_TRILL_PARAM:]
    else:
        y = y[:, :const.NUM_PRIME_PARAM]

    return x.to(device), y.to(device)


def cal_tempo_loss_in_beat(pred_x, true_x, note_locations, start_index, args, model_config):
    previous_beat = -1

    num_notes = pred_x.shape[1]
    start_beat = note_locations[start_index].beat
    num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1

    pred_beat_tempo = th.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(args.device)
    true_beat_tempo = th.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(args.device)
    for i in range(num_notes):
        current_beat = note_locations[i+start_index].beat
        if current_beat > previous_beat:
            previous_beat = current_beat
            if 'baseline' in args.modelCode:
                for j in range(i, num_notes):
                    if note_locations[j+start_index].beat > current_beat:
                        break
                if not i == j:
                            pred_beat_tempo[current_beat - start_beat] = th.mean(pred_x[0, i:j, const.QPM_INDEX])
                    true_beat_tempo[current_beat - start_beat] = th.mean(true_x[0, i:j, const.QPM_INDEX])
            else:
                pred_beat_tempo[current_beat-start_beat] = pred_x[0,i,const.QPM_INDEX:const.QPM_INDEX + const.NUM_TEMPO_PARAM]
                true_beat_tempo[current_beat-start_beat] = true_x[0,i,const.QPM_INDEX:const.QPM_INDEX + const.NUM_TEMPO_PARAM]

    tempo_loss = criterion(pred_beat_tempo, true_beat_tempo, model_config)
    if args.deltaLoss and pred_beat_tempo.shape[0] > 1:
        prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
        true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
        delta_loss = criterion(prediction_delta, true_delta, model_config)

        tempo_loss = (tempo_loss + delta_loss * args.delta_weight) / (1 + args.delta_weight)

    return tempo_loss


def criterion(pred, target, model_config, aligned_status=1):
    if isinstance(aligned_status, int):
        data_size = pred.shape[-2] * pred.shape[-1]
    else:
        data_size = th.sum(aligned_status).item() * pred.shape[-1]
        if data_size == 0:
            data_size = 1
    if target.shape != pred.shape:
        print('Error: The shape of the target and prediction for the loss calculation is different')
        print(target.shape, pred.shape)
        # return th.zeros(1).to(DEVICE)
        return th.zeros(1)

    if model_config.loss_type == 'MSE':
        return th.sum(((target - pred) ** 2) * aligned_status) / data_size
    elif model_config.loss_type == 'CE':
        return -1 * th.sum((target * th.log(pred) + (1 - target) * th.log(1 - pred)) * aligned_status) / data_size
    else:
        print('Undefined loss type:', model_config.loss_type)



# if LOSS_TYPE == 'MSE':
#     def criterion(pred, target, aligned_status=1):
#         if isinstance(aligned_status, int):
#             data_size = pred.shape[-2] * pred.shape[-1]
#         else:
#             data_size = th.sum(aligned_status).item() * pred.shape[-1]
#             if data_size == 0:
#                 data_size = 1
#         if target.shape != pred.shape:
#             print('Error: The shape of the target and prediction for the loss calculation is different')
#             print(target.shape, pred.shape)
#             return th.zeros(1).to(DEVICE)
#         return th.sum(((target - pred) ** 2) * aligned_status) / data_size
# elif LOSS_TYPE == 'CE':
#     # criterion = nn.CrossEntropyLoss()
#     def criterion(pred, target, aligned_status=1):
#         if isinstance(aligned_status, int):
#             data_size = pred.shape[-2] * pred.shape[-1]
#         else:
#             data_size = th.sum(aligned_status).item() * pred.shape[-1]
#             if data_size ==0:
#                 data_size = 1
#                 print('data size for loss calculation is zero')
#         return -1 * th.sum((target * th.log(pred) + (1-target) * th.log(1-pred)) * aligned_status) / data_size
