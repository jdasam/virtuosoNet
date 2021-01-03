from pickle import load
import numpy as np
import argparse
import torch
import copy
from math import log
import csv
import pretty_midi

from .constants import *
from .pyScoreParser.data_class import ScoreData
from .pyScoreParser.feature_extraction import ScoreExtractor
from .pyScoreParser.data_for_training import convert_feature_to_VirtuosoNet_format
from .pyScoreParser.feature_to_performance import apply_tempo_perform_features
from .pyScoreParser.xml_utils import xml_notes_to_midi
from .pyScoreParser.performanceWorm import plot_performance_worm
from .pyScoreParser.midi_utils.midi_utils import save_midi_notes_as_piano_midi
from .pyScoreParser.utils import binary_index, get_item_by_xml_position
from pathlib import Path
from .utils import load_weight
from . import graph


def inference(args, model, device):
    model = load_weight(model, args.checkpoint)
    model.eval()
    # load score
    score, input, edges, note_locations = get_input_from_xml(args.xml_path, args.composer, args.qpm_primo, model.stats['input_keys'], model.stats['graph_keys'], model.stats['stats'], device)
    with torch.no_grad():
        outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z='zero')

    save_path = args.output_path / f"{args.xml_path.parent.stem}_{args.xml_path.stem}_by_{args.model_code}.mid"
    save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'], note_locations, args.multi_instruments, args.boolPedal, args.disklavier)

def generate_midi_from_xml(model, xml_path, composer, save_path, device, initial_z='zero', bool_pedal=False, disklavier=False):
    score, input, edges, note_locations = get_input_from_xml(xml_path, composer, None, model.stats['input_keys'], model.stats['graph_keys'], model.stats['stats'], device)
    with torch.no_grad():
        outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z=initial_z)

    save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'], note_locations, bool_pedal=bool_pedal, disklavier=disklavier)


def save_model_output_as_midi(model_outputs, save_path, score, output_keys, stats, note_locations, multi_instruments=False, bool_pedal=False, disklavier=False):
    outputs = scale_model_prediction_to_original(model_outputs, output_keys, stats)
    output_features = model_prediction_to_feature(outputs, output_keys)

    xml_notes, tempos = apply_tempo_perform_features(score, output_features, start_time=0.5, predicted=True, return_tempo=True)
    if not save_path.parent.exists():
        save_path.parent.mkdir()
    output_midi, midi_pedals = xml_notes_to_midi(xml_notes, multi_instruments)

    plot_performance_worm(output_features, note_locations['beat'], save_path.with_suffix('.png'))
    eighth_positions = score.xml_obj.get_interval_positions(interval_in_16th=2)
    def cal_time_position_with_tempo(xml_position, tempos, divisions):
        corresp_tempo = get_item_by_xml_position(tempos, dict(xml_position=xml_position))
        previous_sec = corresp_tempo.time_position
        passed_duration = xml_position - corresp_tempo.xml_position
        passed_second = passed_duration / divisions / corresp_tempo.qpm * 60

        return previous_sec + passed_second

    eighth_times = []
    for position in eighth_positions:
        last_note = get_item_by_xml_position(xml_notes, dict(xml_position=position))
        divisions = last_note.state_fixed.divisions
        eighth_times.append(cal_time_position_with_tempo(position, tempos, divisions))
    with open(f'{save_path}_beat.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([f'{el:.3f}' for el in eighth_times])

    # add midi click channel
    
    click_notes = [pretty_midi.Note(velocity=64, pitch=64, start=el, end=el+0.01) for el in eighth_times]

    save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_path,
                                  bool_pedal=bool_pedal, disklavier=disklavier, tempo_clock=click_notes)


def get_input_from_xml(xml_path, composer, qpm_primo, input_keys, graph_keys, stats, device='cuda'):
    score = ScoreData(xml_path, None, composer, read_xml_only=True)
    feature_extractor = ScoreExtractor(input_keys)
    input_features = feature_extractor.extract_score_features(score)
    if qpm_primo is not None:
        input_features['qpm_primo'] = log(qpm_primo, 10)
    if 'note_location' not in input_features:
        input_features['note_location'] = feature_extractor.get_note_location(score)
    input, _, _, _ = convert_feature_to_VirtuosoNet_format(input_features, stats, input_keys=input_keys, output_keys=[], meas_keys=[], beat_keys=[])
    input = torch.Tensor(input).unsqueeze(0).to(device)
    if graph_keys and len(graph_keys) > 0:
        edges = graph.edges_to_matrix(score.notes_graph, score.num_notes, graph_keys).to(device)
    else:
        edges = None
    note_locations = {
            'beat': torch.Tensor(input_features['note_location']['beat']).type(torch.int32),
            'measure': torch.Tensor(input_features['note_location']['measure']).type(torch.int32),
            'section': torch.Tensor(input_features['note_location']['section']).type(torch.int32),
            'voice': torch.Tensor(input_features['note_location']['voice']).type(torch.int32),
    }
    return score, input, edges, note_locations

def scale_model_prediction_to_original(prediction, output_keys, stats):
    prediction = np.squeeze(prediction.cpu().numpy())
    idx = 0
    for key in output_keys:
        prediction[:,idx]  *= stats[key]['stds']
        prediction[:,idx]  += stats[key]['mean']
        idx += 1 
    return prediction


def model_prediction_to_feature(prediction, output_keys):
    output_features = {}
    idx = 0
    for key in output_keys:
        output_features[key] = prediction[:,idx]
        idx += 1
    
    return output_features




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--sessMode", type=str,
                        default='train', help="train or test or testAll")
    parser.add_argument("-path", "--test_path", type=str,
                        default="./test_pieces/bps_5_1/", help="folder path of test mat")
    parser.add_argument("-tempo", "--startTempo", type=int,
                        default=0, help="start tempo. zero to use xml first tempo")

    model = load_model
    load_file_and_generate_performance(args.test_path, args)