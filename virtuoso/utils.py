import torch as th
import shutil
from . import model_constants as const
from . import data_process as dp
from omegaconf import OmegaConf
import yaml
import _pickle as pickle


def read_model_setting(yml_path):
    with open(yml_path, 'r') as f:
        yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
    config = OmegaConf.create(yaml_obj)
    return config

def load_dat(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_weight(model, checkpoint_path):
    checkpoint = th.load(checkpoint_path,  map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(checkpoint_path, checkpoint['epoch']))
    return model

class LossCalculator:
    def __init__(self, criterion, args, logger):
        self.criterion = criterion
        self.logger = logger

        self.delta = args.delta_loss
        self.delta_weight = args.delta_weight
        self.is_hier = args.is_hier
        self.hier_meas = args.hier_meas
        self.hier_beat = args.hier_beat
        self.is_trill = args.is_trill
        self.intermediate_loss = args.intermediate_loss
        self.tempo_loss_in_note = args.tempo_loss_in_note

        self.tempo_idx = 0
        self.vel_idx = 1
        self.dev_idx = 2
        self.pedal_idx = 3

    def cal_loss_by_term(self, out, target, note_locations, align_matched, pedal_status):
        # if self.tempo_loss_in_note:
        #     tempo_loss = self.criterion(out[:, :, 0:1],
        #                         target[:, :, 0:1], align_matched)
        # else:
        tempo_loss = self.cal_tempo_loss_in_beat(out, target, note_locations['beat'])
        vel_loss = self.criterion(out[:, :, self.vel_idx:self.dev_idx],
                            target[:, :, self.vel_idx:self.dev_idx], align_matched)
        dev_loss = self.criterion(out[:, :, self.dev_idx:self.pedal_idx],
                            target[:, :, self.dev_idx:self.pedal_idx], align_matched)
        articul_loss = self.criterion(out[:, :, self.pedal_idx:self.pedal_idx+1],
                                target[:, :, self.pedal_idx:self.pedal_idx+1], pedal_status)
        pedal_loss = self.criterion(out[:, :, self.pedal_idx+1:], target[:, :, self.pedal_idx+1:],
                            align_matched)
        total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11
        loss_dict = {'tempo': tempo_loss.item(), 'vel': vel_loss.item(), 'dev': dev_loss.item(), 'articul': articul_loss.item(), 'pedal': pedal_loss.item()}
        return total_loss, loss_dict
         

    def cal_tempo_loss_in_beat(self, pred_x, target, beat_indices):
        previous_beat = -1

        num_notes = pred_x.shape[1]
        start_beat = beat_indices[0]
        num_beats = beat_indices[num_notes-1] - start_beat + 1

        pred_beat_tempo = th.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(pred_x.device)
        true_beat_tempo = th.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(target.device)
        for i in range(num_notes):
            current_beat = beat_indices[i]
            if current_beat > previous_beat:
                previous_beat = current_beat
                if self.tempo_loss_in_note:
                    for j in range(i, num_notes):
                        if beat_indices[j] > current_beat:
                            break
                    if not i == j:
                        pred_beat_tempo[current_beat - start_beat] = th.mean(pred_x[0, i:j, self.tempo_idx])
                        true_beat_tempo[current_beat - start_beat] = th.mean(target[0, i:j, self.tempo_idx])
                else:
                    pred_beat_tempo[current_beat-start_beat] = pred_x[0,i,self.tempo_idx:self.tempo_idx + const.NUM_TEMPO_PARAM]
                    true_beat_tempo[current_beat-start_beat] = target[0,i,self.tempo_idx:self.tempo_idx + const.NUM_TEMPO_PARAM]

        tempo_loss = self.criterion(pred_beat_tempo, true_beat_tempo)
        if self.delta and pred_beat_tempo.shape[0] > 1:
            prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
            true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
            delta_loss = self.criterion(prediction_delta, true_delta)

            tempo_loss = (tempo_loss + delta_loss * self.delta_weight) / (1 + self.delta_weight)

        return tempo_loss

    def __call__(self, output, target, total_out_list, note_locations, align_matched, pedal_status):
        if self.is_hier:
            if self.hier_meas:
                hierarchy_numbers = note_locations['measure']
            elif self.hier_beat:
                hierarchy_numbers = note_locations['beat']
            tempo_in_hierarchy = note_tempo_infos_to_beat(target, hierarchy_numbers, 0)
            dynamics_in_hierarchy = note_tempo_infos_to_beat(target, hierarchy_numbers, 1)
            tempo_loss = self.criterion(output[:, :, 0:1], tempo_in_hierarchy)
            vel_loss = self.criterion(output[:, :, 1:2], dynamics_in_hierarchy)
            if self.delta and output.shape[1] > 1:
                vel_out_delta = output[:, 1:, 1:2] - output[:, :-1, 1:2]
                vel_true_delta = dynamics_in_hierarchy[:,1:, :] - dynamics_in_hierarchy[:, :-1, :]

                vel_loss += self.criterion(vel_out_delta, vel_true_delta) * self.delta_weight
                vel_loss /= 1 + self.delta_weight
            total_loss = tempo_loss + vel_loss
            loss_dict = {'tempo': tempo_loss.item(), 'vel': vel_loss.item(), 'dev': th.zeros(1).item(), 'articul': th.zeros(1).item(), 'pedal': th.zeros(1).item()}
        # elif self.is_trill:
        #     trill_bool = batch_x[:, :,
        #                         is_trill_index_concated:is_trill_index_concated + 1]
        #     if th.sum(trill_bool) > 0:
        #         total_loss = criterion(outputs, batch_y, trill_bool)
        #     else:
        #         return th.zeros(1), th.zeros(1), th.zeros(1),  th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1)
        else:
            if self.intermediate_loss:
                total_loss = th.zeros(1).to(output.device)
                # for out in total_out_list:
                for out in total_out_list:
                    total_l, loss_dict = self.cal_loss_by_term(out, target, note_locations, align_matched, pedal_status)
                    total_loss += total_l
                total_loss /= len(total_out_list)
            else:
                total_loss, loss_dict = self.cal_loss_by_term(output, target, note_locations, align_matched, pedal_status)
        return total_loss, loss_dict


def get_mean_of_loss_dict(loss_dict_list):
    output = {}
    for key in loss_dict_list[0].keys():
        output[key] = sum([x[key] for x in loss_dict_list]) / len(loss_dict_list)
    return output

def note_tempo_infos_to_beat(y, beat_numbers, index=0):
    beat_tempos = []
    num_notes = y.size(1)
    prev_beat = -1
    for i in range(num_notes):
        cur_beat = beat_numbers[i]
        if cur_beat > prev_beat:
            beat_tempos.append(y[0,i,index])
            prev_beat = cur_beat
    num_beats = len(beat_tempos)
    beat_tempos = th.stack(beat_tempos).view(1,num_beats,-1)
    return beat_tempos

def make_criterion_func(loss_type, device):
    if loss_type == 'MSE':
        def criterion(pred, target, aligned_status=1):
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = th.sum(aligned_status).item() * pred.shape[-1]
                if data_size == 0:
                    data_size = 1
            if target.shape != pred.shape:
                print('Error: The shape of the target and prediction for the loss calculation is different')
                print(target.shape, pred.shape)
                return th.zeros(1).to(device)
            return th.sum(((target - pred) ** 2) * aligned_status) / data_size
    elif loss_type == 'CE':
        # criterion = nn.CrossEntropyLoss()
        def criterion(pred, target, aligned_status=1):
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = th.sum(aligned_status).item() * pred.shape[-1]
                if data_size ==0:
                    data_size = 1
                    print('data size for loss calculation is zero')
            return -1 * th.sum((target * th.log(pred) + (1-target) * th.log(1-pred)) * aligned_status) / data_size

    return criterion


def save_checkpoint(state, is_best):
    save_name = 'checkpoint_last.pt'
    th.save(state, save_name)
    if is_best:
        best_name = 'checkpoint_best.pt'
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
        perform_kld = th.zeros(1)
    else:
        perform_kld = -0.5 * \
            th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
        total_loss += perform_kld * kld_weight
    optimizer.zero_grad()
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    return total_loss

    # if HIERARCHY:
    #     return tempo_loss, vel_loss, th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), perform_kld
    # elif TRILL:
    #     return th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), total_loss, th.zeros(1)
    # else:
    #     return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, th.zeros(1), perform_kld

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

