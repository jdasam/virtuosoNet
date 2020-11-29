import torch
from .utils import note_tempo_infos_to_beat
from . import model_constants as const

class LossCalculator:
    def __init__(self, criterion, args, logger):
        self.criterion = criterion
        self.logger = logger

        self.delta = args.delta_loss
        self.delta_weight = args.delta_weight
        self.is_hier = args.is_hier
        self.hier_meas = args.hier_meas
        self.hier_beat = args.hier_beat
        self.meas_note = args.meas_note
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

        pred_beat_tempo = torch.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(pred_x.device)
        true_beat_tempo = torch.zeros([num_beats, const.NUM_TEMPO_PARAM]).to(target.device)
        for i in range(num_notes):
            current_beat = beat_indices[i]
            if current_beat > previous_beat:
                previous_beat = current_beat
                if self.tempo_loss_in_note:
                    for j in range(i, num_notes):
                        if beat_indices[j] > current_beat:
                            break
                    if not i == j:
                        pred_beat_tempo[current_beat - start_beat] = torch.mean(pred_x[0, i:j, self.tempo_idx])
                        true_beat_tempo[current_beat - start_beat] = torch.mean(target[0, i:j, self.tempo_idx])
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
            loss_dict = {'tempo': tempo_loss.item(), 'vel': vel_loss.item(), 'dev': torch.zeros(1).item(), 'articul': torch.zeros(1).item(), 'pedal': torch.zeros(1).item()}
        elif self.meas_note:
            iterative_out_list = total_out_list['iter_out']
            meas_out = total_out_list['meas_out']
            note_target = target['note']
            measure_target = target['measure']
            measure_numbers = note_locations['measure']
            if self.intermediate_loss:
                total_loss = torch.zeros(1).to(output.device)
                for out in iterative_out_list:
                    total_l, loss_dict = self.cal_loss_by_term(out, note_target, note_locations, align_matched, pedal_status)
                    total_loss += total_l
                total_loss /= len(iterative_out_list)
            else:
                total_loss, loss_dict = self.cal_loss_by_term(output, note_target, note_locations, align_matched, pedal_status)
            tempo_in_hierarchy = note_tempo_infos_to_beat(measure_target, measure_numbers, 0)
            dynamics_in_hierarchy = note_tempo_infos_to_beat(measure_target, measure_numbers, 1)
            meas_tempo_loss = self.criterion(meas_out[:, :, 0:1], tempo_in_hierarchy)
            meas_vel_loss = self.criterion(meas_out[:, :, 1:2], dynamics_in_hierarchy)
            loss_dict['meas_tempo'] = meas_tempo_loss
            loss_dict['meas_vel'] = meas_vel_loss
            total_loss += (meas_tempo_loss + meas_vel_loss) / 2
            total_loss /= 2
        # elif self.is_trill:
        #     trill_bool = batch_x[:, :,
        #                         is_trill_index_concated:is_trill_index_concated + 1]
        #     if torch.sum(trill_bool) > 0:
        #         total_loss = criterion(outputs, batch_y, trill_bool)
        #     else:
        #         return torch.zeros(1), torch.zeros(1), torch.zeros(1),  torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
        else:
            if self.intermediate_loss:
                total_loss = torch.zeros(1).to(output.device)
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