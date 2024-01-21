import copy
import torch
import numpy as np
from .pyScoreParser.xml_utils import xml_notes_to_midi
from .pyScoreParser.feature_to_performance import apply_tempo_perform_features
from .utils import note_feature_to_beat_mean, make_criterion_func, handle_args
from .loss import LossCalculator
from .parser import get_parser
from .dataset import ScorePerformDataset


class MidiConverter:
  def __init__(self, stats_info):
    '''
    stats_info (dict): contains key name, (mean, stds), key_to_dim
    '''
    self.stats = stats_info['stats']
    self.output_key_to_dim = stats_info['key_to_dim']['output']
  
  def _scale_model_prediction_to_original(self, prediction):
    '''
    prediction (torch.Tensor): output of VirtuosoNet. 1 x Num_notes X Num_outuput
    '''
    pred = np.squeeze(prediction.cpu().detach().clone().numpy())
    for key, value in self.output_key_to_dim.items():
        pred[:,value[0]] = pred[:,value[0]] * self.stats[key]['stds'] +  self.stats[key]['mean']
    return pred
  
  def _model_prediction_to_feature(self, prediction):
    '''
    prediction (np.array): unnormalized features
    
    out (dict)
    '''
    output_features = {}
    for key, value in self.output_key_to_dim.items():
        output_features[key] = prediction[:,value[0]]
    return output_features

  def save_qpm_for_each_note(self, xml_notes, tempos):
    tempo_idx = 0
    qpm = tempos[0].qpm 
    for note in xml_notes:
      while note.note_duration.xml_position >= tempos[tempo_idx].end_xml and tempos[tempo_idx].end_xml != 0:
        tempo_idx += 1
      qpm = tempos[tempo_idx].qpm
      note.qpm = qpm
    return xml_notes


  def _elongate_pedal(self, midi_notes, midi_pedals, threshold=64):
    clone_notes = copy.deepcopy(midi_notes)
    sorted_by_end_indices = sorted(range(len(clone_notes)), key=lambda x: clone_notes[x].end)
    clone_notes.sort(key=lambda x:x.end)
    elongated_notes = []
    current_note_idx = 0
    for i, pedal in enumerate(midi_pedals[:-1]):
      next_pedal = midi_pedals[i+1]
      if pedal.value > threshold:
        while next_pedal.time > clone_notes[current_note_idx].end > pedal.time:
          elongated_notes.append(clone_notes[current_note_idx])
          current_note_idx += 1
      else:
        for note in elongated_notes:
          note.end = pedal.time
          elongated_notes = []
        while next_pedal.time > clone_notes[current_note_idx].end > pedal.time:
          current_note_idx += 1
    for note in elongated_notes:
      note.end = midi_pedals[-1].time
    clone_notes.sort(key=lambda x:x.start)
    return clone_notes
  
  def _elongate_xml_notes_by_pedal(self, xml_notes, midi_pedals, threshold=64):
    clone_notes = copy.deepcopy(xml_notes)
    sorted_by_end_indices = sorted(range(len(clone_notes)), key=lambda x: clone_notes[x].note_duration.time_position + clone_notes[x].note_duration.seconds)
    clone_notes.sort(key=lambda x:x.note_duration.time_position + x.note_duration.seconds)
    elongated_notes = []
    current_note_idx = 0
    for i, pedal in enumerate(midi_pedals[:-1]):
      next_pedal = midi_pedals[i+1]
      if pedal.value > threshold:
        # print(clone_notes[current_note_idx].note_duration.time_position + clone_notes[current_note_idx].note_duration.seconds, pedal.time, next_pedal.time)
        while next_pedal.time > clone_notes[current_note_idx].note_duration.time_position + clone_notes[current_note_idx].note_duration.seconds > pedal.time:
          elongated_notes.append(clone_notes[current_note_idx])
          current_note_idx += 1
      else:
        for note in elongated_notes:
          # print(note.note_duration.seconds, pedal.time - note.note_duration.time_position)
          note.note_duration.seconds = pedal.time - note.note_duration.time_position
          elongated_notes = []
        while next_pedal.time > clone_notes[current_note_idx].note_duration.time_position + clone_notes[current_note_idx].note_duration.seconds > pedal.time:
          current_note_idx += 1
    for note in elongated_notes:
      note.note_duration.seconds = midi_pedals[-1].time - note.note_duration.time_position
    clone_notes.sort(key=lambda x:(x.note_duration.xml_position, -x.pitch[1]))
    return clone_notes

  def __call__(self, score_data, prediction, elongate_pedal=True):
    '''
    Arguments:
      score_data (pyScoreParser.ScoreData): Score Data
      perform_features (torch.Tensor): performance features

    Return 
      output_midi (list): List of midi notes
      midi_pedals (list): List of midi pedals
    '''
    
    unnorm_pred = self._scale_model_prediction_to_original(prediction)
    perf_features = self._model_prediction_to_feature(unnorm_pred)
    xml_notes, tempos = apply_tempo_perform_features(score_data, perf_features, start_time=0.5, predicted=True, return_tempo=True, sort_notes=False)
    output_midi, midi_pedals = xml_notes_to_midi(xml_notes, multi_instruments=False, ignore_overlapped=True)

    tempo_xml_notes = [note for note in xml_notes if not note.is_rest and  not note.is_overlapped]
    tempo_xml_notes = self.save_qpm_for_each_note(tempo_xml_notes, tempos)

    if elongate_pedal:
      output_midi = self._elongate_pedal(output_midi, midi_pedals)
      tempo_xml_notes = self._elongate_xml_notes_by_pedal(tempo_xml_notes, midi_pedals)
    return output_midi, midi_pedals, tempo_xml_notes 



class PerfEvaluator:
  '''
    PerfEvaluator that evaluates the performance
  '''
  def __init__(self, data_collection, data_stats):
    '''
    Arguments:
      data_collection (list): A list of performance items. Each item is a one single performance, perf_dict. The values of features are already normalized by data_stats.
      data_stats (): Statistics of features that are applied to teh data_collection

    Initialized:
      self.ref_performances (list of dict): {'piece_path': path_to_musicxml (str), 'ids': corresponding performance index in self.data }
    '''
    self.data = data_collection
    self.stats = data_stats
    self.ref_performances = self.get_grouped_ids_from_dataset(data_collection)
    self.midi_converter = MidiConverter(data_stats)
    args = self._make_default_args()
    criterion = make_criterion_func(args.loss_type)
    self.loss_calculator = LossCalculator(criterion)
    # loss_calculator = LossCalculator(criterion, args)
  

  def _make_default_args(self):
    parser = get_parser()
    args = parser.parse_args(
        args=["--delta_loss=true",
              "--vel_balance_loss=true",
              ]
    )
    # args, net_params, configs = handle_args(args)
    return args

  def get_grouped_ids_from_dataset(self, data_collection):
    '''
    This function returns the grouped index of performance that are playing the same piece, from a given list of entire performances

    Args:
      data_collection (list): self.data of ScorePerformDataset. Each item is a one single performance.

    Output:
      grouped_idx (list): A list of dictionary. Each item has 'piece_path' and 'ids'.

    '''
    valid_data_list = [{'path': item['score_path'], 'idx':i} for i, item in enumerate(data_collection)] 
    valid_data_list.sort(key=lambda x:x['path'])

    grouped_idx = []
    prev_piece_path = valid_data_list[0]['path']
    temp_groups = {'piece_path': prev_piece_path, 'ids':[]}
    for item in valid_data_list:
      if item['path'] == prev_piece_path:
        temp_groups['ids'].append(item['idx'])
      else:
        prev_piece_path = item['path']
        if len(temp_groups) > 1:
          grouped_idx.append(temp_groups)
        temp_groups = {'piece_path':item['path'], 'ids':[item['idx']]}
    return grouped_idx


  def _find_corresp_ref(self, perf_xml_path):
    '''
    Find performance dictionary for given perf_xml_path from the self.ref_performances

    Argument:
      perf_xml_path (str): xml_path
    '''
    for perf in self.ref_performances:
      if perf['piece_path'] == perf_xml_path:
        return perf
    else:
      raise Exception(f"Cannot find piece with path of {perf_xml_path}")

  def _tempo_to_original_scale(self, tempo):
    '''
    Scale the given tempo into original scale

    Arguments:
      tempo (np.array)
    '''
    return tempo * self.stats['beat_tempo']['stds'] + self.stats['beat_tempo']['mean']

  def _get_beat_tempo(self, perf_dict):
    '''
    Get tempo in beat-level of from perf_dict

    Argument:
      perf_dict (dict): Data for single performance which are packaged for training VirtuosoNet.
                        Has input(torch.Tensor), output(torch.Tensor), note_location(dict) as keys
    
    Output:
      
    
    '''

    note_locations = {'beat': torch.LongTensor([perf_dict['note_location']['beat']])}
    tempo = torch.Tensor(perf_dict['output'][:,0:1]).unsqueeze(0)
    beat_tempo = note_feature_to_beat_mean(tempo, note_locations['beat'], use_mean=False)
    return self._tempo_to_original_scale(beat_tempo)

  def _get_total_beat_tempo(self, corresp_ref_perf):
    '''
    Arguments:
      corresp_ref_perf (dict): 
    '''
    total_beat_tempo = [self._get_beat_tempo(self.data[idx]) for idx in corresp_ref_perf['ids']]
    total_beat_tempo = torch.cat(total_beat_tempo)
    beat_tempo_diff = total_beat_tempo.diff(dim=1)
    return total_beat_tempo, beat_tempo_diff
  
  def _cal_faster_point_overlap(self, out_beat_tempo, ref_beat_tempo_diff, threshold=0.1, agree_ratio=0.6):
    '''
    Calculate overlap between given tempo curve and other reference performances, in terms of accelearting moment.

    Arguments:
      out_beat_tempo: 
    '''
    num_faster = (ref_beat_tempo_diff[..., 0] > threshold).sum(dim=0)
    generated_faster_spot = out_beat_tempo[..., 0].diff(dim=1) > threshold
    common_faseter_spot = num_faster > ref_beat_tempo_diff.shape[0] * agree_ratio
    overlap = (generated_faster_spot * common_faseter_spot).sum() / common_faseter_spot.sum()
    return overlap
  
  def _cal_slower_point_overlap(self, out_beat_tempo, ref_beat_tempo_diff, threshold=0.1, agree_ratio=0.6):
    num_faster = (ref_beat_tempo_diff[..., 0] < -threshold).sum(dim=0)
    generated_faster_spot = out_beat_tempo[..., 0].diff(dim=1) < -threshold
    common_faseter_spot = num_faster > ref_beat_tempo_diff.shape[0] * agree_ratio
    overlap = (generated_faster_spot * common_faseter_spot).sum() / common_faseter_spot.sum()
    return overlap
  
  
  def _compare_two(self, comp_perf, perf_xml_path):
    '''
    This function compares a performance with other performances of the same piece, one-by-one.


    comp_perf (torch.Tensor): sequence of note output features 1 X T X C

    Output:
    
    '''
    total_out = []
    corresp_ref_perf = self._find_corresp_ref(perf_xml_path)
    for ref_id in corresp_ref_perf['ids']:
      output = comp_perf
      ref = self.data[ref_id]
      target = torch.Tensor(ref['output']).unsqueeze(0)
      note_locations = {'beat': torch.LongTensor([ref['note_location']['beat']])}
      align_matched = torch.LongTensor(ref['align_matched'])
      align_matched = align_matched.unsqueeze(0).unsqueeze(-1)
      pedal_status = torch.Tensor(ref['articulation_loss_weight'])
      pedal_status = pedal_status.unsqueeze(0).unsqueeze(-1)
      out, loss_dict = self.loss_calculator.cal_loss_by_term(output, target, note_locations, align_matched, pedal_status)
      total_out.append(loss_dict)
    return total_out
  
  def __call__(self, output, perf_xml_path):
    corresp_ref_perf = self._find_corresp_ref(perf_xml_path)
    note_locations = {'beat': torch.LongTensor([self.data[corresp_ref_perf['ids'][0]]['note_location']['beat']])}
    out_beat_tempo = note_feature_to_beat_mean(output, note_locations['beat'], use_mean=False)
    _, ref_beat_tempo_diff = self._get_total_beat_tempo(corresp_ref_perf)
    
    faster_point_overlap = self._cal_faster_point_overlap(out_beat_tempo, ref_beat_tempo_diff)
    slower_point_overlap = self._cal_slower_point_overlap(out_beat_tempo, ref_beat_tempo_diff)
    
    return faster_point_overlap, slower_point_overlap
    

class PedalCompare:
  def __init__(self):
    pass

  
  def _make_pedal(self, score, output):
    '''
    score (pyScoreParser.ScoreData)
    output 
    '''

    xml_notes, tempos = apply_tempo_perform_features(score, output, start_time=0.5, predicted=True, return_tempo=True)
    _, midi_pedals = xml_notes_to_midi(xml_notes)
    return midi_pedals

  def compare_two_pedals(self, out_a, out_b, score):
    '''
    

    score (pyScoreParser.ScoreData)
    '''


    xml_notes, tempos = apply_tempo_perform_features(score, output_features, start_time=0.5, predicted=True, return_tempo=True)
    output_midi, midi_pedals = xml_notes_to_midi(xml_notes)

    return


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args(
      args=["--yml_path=ymls/han_measnote.yml",
            "--data_path=datasets/main_pkl/",
            "--emotion_data_path=datasets/emotion_pkl/",
            "--delta_loss=true",
            "--vel_balance_loss=true",
            "--device=cpu"]
  )
  args, net_params, configs = handle_args(args)
  device = 'cpu'

  criterion = make_criterion_func(args.loss_type)
  loss_calculator = LossCalculator(criterion, args)
  valid_set = ScorePerformDataset(args.data_path, 
                                type="valid", 
                                len_slice=args.len_valid_slice, 
                                len_graph_slice=args.len_graph_slice, 
                                graph_keys=args.graph_keys, 
                               )
  from virtuoso.inference import InferenceModel

  ckpt_path = '/home/teo/userdata/virtuosonet_checkpoints/yml_path=ymls/han_measnote.yml meas_note=True delta_weight=5.0 delta_loss=True vel_balance_loss=True intermediate_loss=False_220201-225301/checkpoint_last.pt'
  inferencer = InferenceModel(ckpt_path, 'cpu', args.output_path)
  converter = MidiConverter(inferencer.model.stats)