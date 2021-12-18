from numpy.lib.arraysetops import isin
import torch
import shutil
from .model_constants import TEMPO_IDX
from . import data_process as dp
from omegaconf import OmegaConf
import yaml
import _pickle as pickle
from pathlib import Path


def read_model_setting(yml_path):
    with open(yml_path, 'r') as f:
        yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
    config = OmegaConf.create(yaml_obj)
    return config

def load_dat(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_weight(model, checkpoint_path):
    if not isinstance(checkpoint_path, str):
        checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path,  map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.stats = checkpoint['stats']
    model.model_code = checkpoint['model_code']
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(checkpoint_path, checkpoint['epoch']))
    return model

# def note_tempo_infos_to_beat(y, beat_numbers, index=0):
#     beat_tempos = []
#     num_notes = y.size(1)
#     prev_beat = -1
#     for i in range(num_notes):
#         cur_beat = beat_numbers[i]
#         if cur_beat > prev_beat:
#             beat_tempos.append(y[0,i,index])
#             prev_beat = cur_beat
#     num_beats = len(beat_tempos)
#     beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
#     return beat_tempos

def make_criterion_func(loss_type, device):
    if loss_type == 'MSE':
        def criterion(pred, target, aligned_status=1):
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = torch.sum(aligned_status).item() * pred.shape[-1]
                if data_size == 0:
                    data_size = 1
            if target.shape != pred.shape:
                print('Error: The shape of the target and prediction for the loss calculation is different')
                print(target.shape, pred.shape)
                return torch.zeros(1).to(device)
            return torch.sum(((target - pred) ** 2) * aligned_status) / data_size
    elif loss_type == 'CE':
        # criterion = nn.CrossEntropyLoss()
        def criterion(pred, target, aligned_status=1):
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = torch.sum(aligned_status).item() * pred.shape[-1]
                if data_size ==0:
                    data_size = 1
                    print('data size for loss calculation is zero')
            return -1 * torch.sum((target * torch.log(pred) + (1-target) * torch.log(1-pred)) * aligned_status) / data_size

    return criterion


def save_checkpoint(dir, state, is_best):
    if isinstance(dir, str):
        dir = Path(dir)
    save_name = dir / 'checkpoint_last.pt'
    torch.save(state, save_name)
    if is_best:
        best_name = dir / 'checkpoint_best.pt'
        shutil.copyfile(save_name, best_name)


def encode_performance_style_vector(input, input_y, edges, note_locations, device, model):
    with torch.no_grad():
        model_eval = model.eval()
        if edges is not None:
            edges = edges.to(device)
        encoded_z = model_eval(input, input_y, edges,
                               note_locations=note_locations, start_index=0, return_z=True)
    return encoded_z


def run_model_in_steps(input, input_y, args, edges, note_locations, model, device, initial_z=False):
    num_notes = input.shape[1]
    with torch.no_grad():  # no need to track history in validation
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

        outputs = torch.cat(total_output, 1)
        return outputs, total_z


def batch_time_step_run(data, model, loss_calculator, optimizer, kld_weight):
    batch_x, batch_y, note_locations, align_matched, pedal_status, edges = data

    # prime_batch_x = batch_x
    # if HIERARCHY:
    #     prime_batch_y = batch_y
    # else:
    #     prime_batch_y = batch_y[:, :, 0:NUM_PRIME_PARAM]

    model_train = model.train()
    outputs, perform_mu, perform_var, total_out_list \
        = model_train(batch_x, batch_y, edges, note_locations)

    total_loss = loss_calculator(outputs, batch_y, total_out_list, note_locations, align_matched, pedal_status)
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
    return total_loss

    # if HIERARCHY:
    #     return tempo_loss, vel_loss, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), perform_kld
    # elif TRILL:
    #     return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), total_loss, torch.zeros(1)
    # else:
    #     return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, torch.zeros(1), perform_kld

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


def note_feature_to_beat_mean(feature, beat_numbers, use_mean=True):
    '''
    Input: feature = Tensor (Batch X Num Notes X Feature dimension)
           beat_numbers = LongTensor of beat index for each notes in feature
           use_mean = use mean to get the representative. Otherwise, sample first one as the representative
    Output: Tensor (Batch X Num Beats X Feature dimension)
    '''
    boundaries = [0] + (torch.where(beat_numbers[1:] - beat_numbers[:-1] == 1)[0] + 1).cpu().tolist() + [len(beat_numbers)]
    if use_mean:
        beat_features = torch.stack([torch.mean(feature[:,boundaries[i-1]:boundaries[i],:], dim=1)
                                for i in range(1, len(boundaries))]).permute(1,0,2)
    else:
        beat_features = torch.stack([feature[:,boundaries[i],:]
                                for i in range(0, len(boundaries)-1)]).permute(1,0,2)

    return beat_features

def note_tempo_infos_to_beat(y, beat_numbers, index=0):
    beat_tempos = []
    num_notes = y.size(1)
    prev_beat = -1
    for i in range(num_notes):
        cur_beat = beat_numbers[i]
        if cur_beat > prev_beat:
            if index is None:
                beat_tempos.append(y[0,i,:])
            if index == TEMPO_IDX:
                beat_tempos.append(y[0,i,TEMPO_IDX:TEMPO_IDX+5])
            else:
                beat_tempos.append(y[0,i,index])
            prev_beat = cur_beat
    num_beats = len(beat_tempos)
    beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
    return beat_tempos

def batch_to_device(batch, device):
    if len(batch) == 6:
        batch_x, batch_y, note_locations, align_matched, pedal_status, edges = batch
    elif len(batch) == 8:
        batch_x, batch_y, beat_y, meas_y, note_locations, align_matched, pedal_status, edges = batch
    else:
        print(f'Unrecognizable batch length: {len(batch)}')
        return
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    align_matched = align_matched.to(device)
    pedal_status = pedal_status.to(device)
    if edges is not None:
        edges = edges.to(device)

    if len(batch) == 6:
        return batch_x, batch_y, note_locations, align_matched, pedal_status, edges
    elif len(batch) == 8:
        return batch_x, batch_y, beat_y.to(device), meas_y.to(device), note_locations, align_matched, pedal_status, edges
    