import torch
from .pyScoreParser.xml_utils import xml_notes_to_midi
from .pyScoreParser.feature_to_performance import apply_tempo_perform_features

class PerfEvaluator:
  def __init__(self):
    
    self.ref_performances = self._get_references

  def _get_references(self, data_collection):
    '''
    data_collection (list)
    '''
    valid_data_list = [{'path': item['score_path'], 'idx':i} for i, item in enumerate(data_collection)] 
    valid_data_list.sort(key=lambda x:x['path'])

    grouped_idx = []
    prev_piece_path = valid_data_list[0]['path']
    temp_groups = []
    for item in valid_data_list:
      if item['path'] == prev_piece_path:
        temp_groups.append(item['idx'])
      else:
        prev_piece_path = item['path']
        if len(temp_groups) > 1:
          grouped_idx.append(temp_groups)
        temp_groups = []
    
    return grouped_idx

  def _compare_two(self, comp_perf):
    '''
    comp_perf (torch.Tensor): sequence of note output features 1 X T X C
    
    

    '''
    for ref in self.ref_performances:
      output = comp_perf
      target = torch.Tensor(ref['output_data']).unsqueeze(0)
      note_locations = {'beat': torch.LongTensor([ref['note_location']['beat']])}
      align_matched = torch.LongTensor(ref['align_matched'])
      align_matched = align_matched.unsqueeze(0).unsqueeze(-1)
      pedal_status = torch.Tensor(valid_set.data[target_id]['articulation_loss_weight'])
      pedal_status = pedal_status.unsqueeze(0).unsqueeze(-1)
  
    out, loss_dict = loss_calculator.cal_loss_by_term(output, target, note_locations, align_matched, pedal_status)


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