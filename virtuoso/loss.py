import torch
from .utils import note_tempo_infos_to_beat, note_feature_to_beat_mean, note_location_numbers_to_padding_bool
from .model_utils import span_beat_to_note_num
from . import model_constants as const
from .pyScoreParser.feature_utils import make_index_continuous

class LossCalculator:
  def __init__(self, criterion, args):
    self.criterion = criterion

    self.delta = args.delta_loss
    self.delta_weight = args.delta_weight
    self.vel_balance = args.vel_balance_loss
    self.is_hier = args.is_hier
    self.hier_meas = args.hier_meas
    self.hier_beat = args.hier_beat
    self.meas_note = args.meas_note
    self.meas_loss_weight = args.meas_loss_weight
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
    tempo_loss, tempo_delta_loss = self.cal_tempo_loss_in_beat(out, target, note_locations['beat'])
    

    vel_loss = self.criterion(out[:, :, self.vel_idx:self.dev_idx],
                        target[:, :, self.vel_idx:self.dev_idx], align_matched)
    dev_loss = self.criterion(out[:, :, self.dev_idx:self.pedal_idx],
                        target[:, :, self.dev_idx:self.pedal_idx], align_matched)
    articul_loss = self.criterion(out[:, :, self.pedal_idx:self.pedal_idx+1],
                            target[:, :, self.pedal_idx:self.pedal_idx+1], pedal_status)
    pedal_loss = self.criterion(out[:, :, self.pedal_idx+1:], target[:, :, self.pedal_idx+1:],
                        align_matched)

    if self.vel_balance:
        vel_balance_loss = cal_velocity_balance_loss(out[:, :, self.vel_idx:self.dev_idx], 
                                        target[:, :, self.vel_idx:self.dev_idx], 
                                        note_locations, align_matched, self.criterion)
    else:
        vel_balance_loss = torch.zeros_like(vel_loss)
                                
    total_loss = (tempo_loss + tempo_delta_loss + vel_loss + vel_balance_loss + dev_loss + articul_loss + pedal_loss * 7) / (11+self.delta_weight)
    loss_dict = {'tempo': tempo_loss.item(), 
                  'vel': vel_loss.item(), 
                  'dev': dev_loss.item(), 
                  'articul': articul_loss.item(), 
                  'pedal': pedal_loss.item(),
                  'tempo_delta': tempo_delta_loss.item(),
                  'vel_balance': vel_balance_loss.item(),
                  }
    return total_loss, loss_dict
        
  def cal_delta_loss(self, pred, target, valid_beat_state=1):
    prediction_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    valid_del_beat_state = valid_beat_state[:,:-1]
    delta_loss = self.criterion(prediction_delta, target_delta, valid_del_beat_state)
    return delta_loss
  
  def add_delta_loss_with_weight(self, loss, delta_loss):
    return (loss + delta_loss * self.delta_weight)  # / (1 + self.delta_weight)

  def cal_tempo_loss_in_beat(self, pred_x, target, beat_indices):
    use_mean = False
    if self.tempo_loss_in_note:
        use_mean = True
    pred_beat_tempo = note_feature_to_beat_mean(pred_x[:,:,self.tempo_idx:self.tempo_idx+const.NUM_TEMPO_PARAM], beat_indices, use_mean)
    true_beat_tempo = note_feature_to_beat_mean(target[:,:,self.tempo_idx:self.tempo_idx+const.NUM_TEMPO_PARAM], beat_indices, use_mean)
    valid_beat_state = note_location_numbers_to_padding_bool(beat_indices).to(pred_x.device)
    tempo_loss = self.criterion(pred_beat_tempo, true_beat_tempo, valid_beat_state)
    if self.delta and pred_beat_tempo.shape[1] > 1:
        delta_loss = self.cal_delta_loss(pred_beat_tempo, true_beat_tempo, valid_beat_state)
        delta_loss *= self.delta_weight
    else:
        delta_loss = torch.zeros_like(tempo_loss)
    return tempo_loss, delta_loss

  def cal_measure_loss(self, meas_out, target, note_locations, loss_dict, hier_type='measure'):
    measure_target = target[hier_type]
    measure_numbers = note_locations[hier_type]

    valid_beat_state = note_location_numbers_to_padding_bool(measure_numbers).to(meas_out.device)
    tempo_in_hierarchy = note_tempo_infos_to_beat(measure_target, measure_numbers, self.tempo_idx)
    meas_tempo_loss = self.criterion(meas_out[:, :, 0:1], tempo_in_hierarchy, valid_beat_state)
    loss_dict[f'{hier_type[:4]}_tempo'] = meas_tempo_loss.item()

    dynamics_in_hierarchy = note_tempo_infos_to_beat(measure_target, measure_numbers, self.vel_idx)
    meas_vel_loss = self.criterion(meas_out[:, :, 1:2], dynamics_in_hierarchy, valid_beat_state)
    loss_dict[f'{hier_type[:4]}_vel'] = meas_vel_loss.item()

    if self.delta and meas_out.shape[1] > 1:
      tempo_delta_loss = self.cal_delta_loss(meas_out[:, :, 0:1], tempo_in_hierarchy, valid_beat_state)
      tempo_delta_loss *= self.delta_weight
      vel_delta_loss = self.cal_delta_loss(meas_out[:, :, 1:2], dynamics_in_hierarchy, valid_beat_state)
      vel_delta_loss *= self.delta_weight
      loss_dict[f'{hier_type[:4]}_tempo_delta'] = tempo_delta_loss.item()
      loss_dict[f'{hier_type[:4]}_vel_delta'] = vel_delta_loss.item()
    else:
      tempo_delta_loss = torch.zeros_like(meas_tempo_loss)
      vel_delta_loss = torch.zeros_like(meas_vel_loss)
      loss_dict[f'{hier_type[:4]}_tempo_delta'] = 0
      loss_dict[f'{hier_type[:4]}_vel_delta'] = 0
    
    return meas_tempo_loss, meas_vel_loss, tempo_delta_loss, vel_delta_loss

  def get_is_hier_loss(self, output, target, note_locations):
    '''
    Loss for model that only predicts measure-level or beat-level output
    '''
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
    return total_loss, loss_dict

  def get_meas_note_loss(self, output, target, total_out_list, note_locations, align_matched, pedal_status):
    note_target = target['note']
    if self.intermediate_loss:
      iterative_out_list = total_out_list['iter_out']
      total_loss = torch.zeros(1).to(output.device)
      for out in iterative_out_list:
        total_l, loss_dict = self.cal_loss_by_term(out, note_target, note_locations, align_matched, pedal_status)
        total_loss += total_l
      total_loss /= len(iterative_out_list)
    else:
      total_loss, loss_dict = self.cal_loss_by_term(output, note_target, note_locations, align_matched, pedal_status)
    meas_tempo_loss, meas_vel_loss, tempo_delta_loss, vel_delta_loss = self.cal_measure_loss(total_out_list['meas_out'], target, note_locations, loss_dict)
    if 'beat_out' in total_out_list:
      _, beat_vel_loss, _, beat_vel_delta_loss = self.cal_measure_loss(total_out_list['beat_out'], target, note_locations, loss_dict, hier_type='beat')
    else:
      beat_vel_loss = torch.zeros_like(meas_vel_loss)
      beat_vel_delta_loss = torch.zeros_like(meas_vel_loss)
    
    total_loss += (meas_tempo_loss + meas_vel_loss + tempo_delta_loss + vel_delta_loss + beat_vel_loss + beat_vel_delta_loss  ) / 2 * self.meas_loss_weight
    return total_loss, loss_dict

  def __call__(self, output, target, total_out_list, note_locations, align_matched, pedal_status):
    if self.is_hier:
      total_loss, loss_dict = self.get_is_hier_loss(output, target, note_locations)
    elif self.meas_note:
      total_loss, loss_dict = self.get_meas_note_loss(output, target, total_out_list, note_locations, align_matched, pedal_status)
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

def cal_multiple_perf_style_loss(perf_mu, perf_var, margin=2):
    num_perf = int(perf_mu.shape[0])
    perf_mu_loss = torch.mean(torch.abs(torch.sum(perf_mu, dim=0)))
    perf_var_loss = torch.mean(torch.abs(torch.sum(perf_var, dim=0)))
    distance = [torch.dist(perf_mu[i], perf_mu[j]) for i in range(num_perf) for j in range(i+1,num_perf)]
    mean_distance = sum(distance) / len(distance)

    total_loss = perf_mu_loss + perf_var_loss + torch.max(torch.zeros_like(mean_distance), (margin - mean_distance))
    loss_dict = {
        'mu': perf_mu_loss,
        'var': perf_var_loss,
        'dist': mean_distance
    }
    return total_loss, loss_dict



def cal_velocity_balance_loss(pred_vel, target_vel, note_locations, align_matched, criterion):
  # filter matched notes only 
  valid_notes_indices = (align_matched==1)[..., 0]
  beat_numbers = torch.nn.utils.rnn.pad_sequence([ note_locations['beat'][i, valid_notes_indices[i]] for i in range(len(pred_vel))], True)
  valid_pred = torch.nn.utils.rnn.pad_sequence([pred_vel[i, valid_notes_indices[i]] for i in range(len(pred_vel))], True)
  valid_target = torch.nn.utils.rnn.pad_sequence([target_vel[i, valid_notes_indices[i]] for i in range(len(pred_vel))], True)

  # Make valid_beat_numbers to continuous
  continuous_beat_numbers = torch.stack([torch.unique_consecutive(beat_numbers[i], return_inverse=True)[1] for i in range(len(pred_vel))])
  is_padded = (valid_pred[...,0]==0) * (valid_target[...,0]==0)
  continuous_beat_numbers[is_padded]= 0 

  beat_pred = note_feature_to_beat_mean(valid_pred, continuous_beat_numbers)
  beat_target = note_feature_to_beat_mean(valid_target, continuous_beat_numbers)
  beat_pred_in_note = span_beat_to_note_num(beat_pred, continuous_beat_numbers)
  beat_target_in_note = span_beat_to_note_num(beat_target, continuous_beat_numbers)
  
  loss = criterion(valid_pred - beat_pred_in_note, valid_target - beat_target_in_note, ~is_padded.unsqueeze(-1)) 
  return loss 