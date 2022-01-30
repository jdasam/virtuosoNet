from pickle import load
import numpy as np
import argparse
import torch
import copy
from math import log
import csv
import pretty_midi
from sklearn.cluster import KMeans

from .constants import *
from .pyScoreParser.data_class import ScoreData
from .pyScoreParser.feature_extraction import ScoreExtractor
from .pyScoreParser.data_for_training import FeatureConverter, convert_feature_to_VirtuosoNet_format
from .pyScoreParser.feature_to_performance import apply_tempo_perform_features
from .pyScoreParser.xml_utils import xml_notes_to_midi
from .pyScoreParser.performanceWorm import plot_performance_worm
from .pyScoreParser.midi_utils.midi_utils import save_midi_notes_as_piano_midi
from .pyScoreParser.utils import binary_index, get_item_by_xml_position
from .pyScoreParser.xml_midi_matching import read_corresp, match_score_pair2perform, make_xml_midi_pair, make_available_xml_midi_positions
from .pyScoreParser.midi_utils import midi_utils
from .pyScoreParser import feature_utils
from .pyScoreParser.utils import get_item_by_xml_position
from pathlib import Path
from .utils import load_weight
from . import graph
from . import style_analysis as sty
from .emotion import get_style_from_emotion_data
from .dataset import EmotionDataset, FeatureCollate, split_graph_to_batch
from torch.utils.data import DataLoader


def inference(args, model, device):
    model = load_weight(model, args.checkpoint)
    model.eval()
    # load score
    score, input, edges, note_locations = get_input_from_xml(args.xml_path, args.composer, args.qpm_primo, model.stats['input_keys'], model.stats['graph_keys'], model.stats['stats'], device)
    with torch.no_grad():
        outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z='zero')
        if args.save_cluster:
            attention_weights = model.score_encoder.get_attention_weights(input, edges, note_locations)
        else:
            attention_weights = None
        # outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z='rand')
    Path(args.output_path).mkdir(exist_ok=True)
    save_path = args.output_path / f"{args.xml_path.parent.stem}_{args.xml_path.stem}_by_{args.model_code}.mid"
    save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'], note_locations, 
                              args.velocity_multiplier, args.multi_instruments, args.tempo_clock,  args.boolPedal, args.disklavier, 
                              clock_interval_in_16th=args.clock_interval_in_16th, save_csv=args.save_csv, save_cluster=args.save_cluster,
                              attention_weights=attention_weights, mod_midi_path=args.mod_midi_path)


def inference_with_emotion(args, model, device):
    model = load_weight(model, args.checkpoint)
    model.eval()
    # encode_emotion
    emotion_set = EmotionDataset(args.emotion_data_path, type="entire", len_slice=2000, len_graph_slice=2000, graph_keys= model.stats['graph_keys'])
    emotion_loader = DataLoader(emotion_set, 5, shuffle=False, num_workers=0, pin_memory=False, collate_fn=FeatureCollate())

    total_perform_z = get_style_from_emotion_data(model, emotion_loader, device)
    abs_mean_by_emotion, norm_mean_by_emotion = sty.get_emotion_representative_vectors(total_perform_z)


    # load score
    score, input, edges, note_locations = get_input_from_xml(args.xml_path, args.composer, args.qpm_primo, model.stats['input_keys'], model.stats['graph_keys'], model.stats['stats'], device)
    Path(args.output_path).mkdir(exist_ok=True)
    with torch.no_grad():
        for i, emotion_z in enumerate(abs_mean_by_emotion):
          outputs, _, _, _ = model(input, None, edges, note_locations, initial_z=emotion_z)
          save_path = args.output_path / f"{args.xml_path.parent.stem}_{args.xml_path.stem}_absE{i+1}_by_{args.model_code}.mid"
          save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'], note_locations, 
                                    args.velocity_multiplier, args.multi_instruments, args.tempo_clock,  args.boolPedal, args.disklavier,)
        for i, emotion_z in enumerate(norm_mean_by_emotion):
          outputs, _, _, _ = model(input, None, edges, note_locations, initial_z=emotion_z)
          save_path = args.output_path / f"{args.xml_path.parent.stem}_{args.xml_path.stem}_normE{i+1}_by_{args.model_code}.mid"
          save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'], note_locations, 
                                    args.velocity_multiplier, args.multi_instruments, args.tempo_clock,  args.boolPedal, args.disklavier,)




def generate_midi_from_xml(model, xml_path, composer, save_path, device, initial_z='zero', multi_instruments=False, tempo_clock=False, bool_pedal=False, disklavier=False):
    score, input, edges, note_locations = get_input_from_xml(xml_path, composer, None, model.stats['input_keys'], model.stats['graph_keys'], model.stats['stats'], device)
    with torch.no_grad():
        outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z=initial_z)
    save_model_output_as_midi(outputs, save_path, score, model.stats['output_keys'], model.stats['stats'], note_locations, 
                              multi_instruments=multi_instruments, tempo_clock=tempo_clock, bool_pedal=bool_pedal, disklavier=disklavier)



def save_model_output_as_midi(model_outputs, save_path, score, output_keys, stats, note_locations, 
                                velocity_multiplier=1, multi_instruments=False, tempo_clock=False, bool_pedal=False, disklavier=False, clock_interval_in_16th=4, 
                                save_csv=False, save_cluster=False, mod_midi_path=None, attention_weights=None):
    outputs = scale_model_prediction_to_original(model_outputs, output_keys, stats)
    output_features = model_prediction_to_feature(outputs, output_keys)
    if velocity_multiplier != 1:
        mean_vel = np.mean(output_features['velocity'])
        output_features['velocity'] = (output_features['velocity'] - mean_vel) * velocity_multiplier + mean_vel

    xml_notes, tempos = apply_tempo_perform_features(score, output_features, start_time=0.5, predicted=True, return_tempo=True)
    # if save_cluster:
    #     cluster = cluster_note_embeddings(model_outputs)
    #     for note, cluster_id in zip(xml_notes, cluster):
    #         note.cluster = cluster_id
    if attention_weights is not None:
        for i, note in enumerate(xml_notes):
            note.attention_weights = attention_weights['note'][i]
    if not save_path.parent.exists():
        save_path.parent.mkdir()
    output_midi, midi_pedals = xml_notes_to_midi(xml_notes, multi_instruments, ignore_overlapped=(mod_midi_path is None))
    if mod_midi_path is not None:
        score_pairs = [{'xml':xml, 'midi':midi} for xml, midi in zip (xml_notes,output_midi)]
        output_midi = load_and_apply_modified_perf_midi(mod_midi_path, score_pairs, xml_notes, output_features, score.beat_positions)
        xml_notes, tempos = apply_tempo_perform_features(score, output_features, start_time=0.5, predicted=True, return_tempo=True)
        output_midi, midi_pedals = xml_notes_to_midi(xml_notes, multi_instruments, ignore_overlapped=(mod_midi_path is None))
        output_midi.sort(key=lambda x:x.start)
    if attention_weights is not None:
        plot_performance_worm(output_features, note_locations['beat'], save_path.with_suffix('.png'), save_csv=save_csv, attention_weights=attention_weights['beat'].tolist())
    else:
        plot_performance_worm(output_features, note_locations['beat'][0], save_path.with_suffix('.png'), save_csv=save_csv)
    if tempo_clock:
        nth_position = score.xml_obj.get_interval_positions(interval_in_16th=clock_interval_in_16th)
        def cal_time_position_with_tempo(xml_position, tempos, divisions):
            corresp_tempo = get_item_by_xml_position(tempos, dict(xml_position=xml_position))
            previous_sec = corresp_tempo.time_position
            passed_duration = xml_position - corresp_tempo.xml_position
            passed_second = passed_duration / divisions / corresp_tempo.qpm * 60
            return previous_sec + passed_second

        nth_times = []
        for position in nth_position:
            last_note = get_item_by_xml_position(xml_notes, dict(xml_position=position))
            divisions = last_note.state_fixed.divisions
            nth_times.append(cal_time_position_with_tempo(position, tempos, divisions))
        with open(f'{save_path}_beat.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([f'{el:.3f}' for el in nth_times])
        # add midi clock channel
        clock_notes = [pretty_midi.Note(velocity=64, pitch=64, start=el, end=el+0.01) for el in nth_times]
    else:
        clock_notes = None
    if save_csv:
        if multi_instruments:
            midi_notes = [note_info_to_tuple(note) for inst in output_midi for note in inst] 
        else:
            midi_notes = [note_info_to_tuple(note) for note in output_midi] 
        with open(f'{save_path}_midi_notes.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(["xml_idx", "start", "end", "pitch", "velocity", "channel", "att_0","att_1","att_2","att_3","att_4","att_5","att_6","att_7" ])
            [writer.writerow(el) for el in midi_notes]
            

    save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_path,
                                  bool_pedal=bool_pedal, disklavier=disklavier, tempo_clock=clock_notes)

def note_info_to_tuple(note):
    info_list = [note.xml_idx, note.start, note.end, note.pitch, note.velocity, note.channel]
    if hasattr(note, "attention_weights"):
        attention = note.attention_weights.tolist()
    else:
        attention = [0] * 8
    info_list += attention
    return info_list

def get_input_from_xml(xml_path, composer, qpm_primo, input_keys, graph_keys, stats, device='cuda', len_graph_slice=400, graph_slice_margin=100,):
    score = ScoreData(xml_path, None, composer, read_xml_only=True)
    feature_extractor = ScoreExtractor(input_keys)
    input_features = feature_extractor.extract_score_features(score)
    if qpm_primo is not None:
        if 'section_tempo' in input_features:
            initial_qpm_primo = input_features['section_tempo'][0]
            input_features['section_tempo'] = [   x + log(qpm_primo, 10) - initial_qpm_primo for x in input_features['section_tempo']]
        input_features['qpm_primo'] = log(qpm_primo, 10)
    if 'note_location' not in input_features:
        input_features['note_location'] = feature_extractor.get_note_location(score)
    feature_converter = FeatureConverter(stats, input_keys=input_keys, output_keys=[], beat_keys=[], meas_keys=[])
    features = feature_converter(input_features)
    # input, _, _, _ = convert_feature_to_VirtuosoNet_format(input_features, stats, input_keys=input_keys, output_keys=[], meas_keys=[], beat_keys=[])
    input = torch.Tensor(features['input']).unsqueeze(0).to(device)
    if graph_keys and len(graph_keys) > 0:
        edges = graph.edges_to_matrix(score.notes_graph, score.num_notes, graph_keys)
        edges = split_graph_to_batch(edges, len_graph_slice ,graph_slice_margin).unsqueeze(0).to(device)
    else:
        edges = None
    note_locations = {
            'beat': torch.LongTensor(input_features['note_location']['beat']).unsqueeze(0),
            'measure': torch.LongTensor(input_features['note_location']['measure']).unsqueeze(0),
            'section': torch.LongTensor(input_features['note_location']['section']).unsqueeze(0),
            'voice': torch.LongTensor(input_features['note_location']['voice']).unsqueeze(0),
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

def cluster_note_embeddings(note_embeddings, n=8):
    np_embeddings = note_embeddings.squeeze().cpu().numpy()
    kmeans = KMeans(n_clusters=n, random_state=0).fit(np_embeddings)
    indices = kmeans.predict(np_embeddings)
    return indices.squeeze().tolist()

def load_and_apply_modified_perf_midi(midi_path, score_pairs, xml_notes, output_features, beat_positions):
    midi = midi_utils.to_midi_zero(midi_path)
    midi_notes = [note for instrument in midi.instruments for note in instrument.notes]
    midi_notes.sort(key=lambda x:x.start)
    corresp_path = midi_path.split('.')[0] + '_corresp.txt'
    corresp = read_corresp(corresp_path)
    match_between_xml_perf = match_score_pair2perform(score_pairs, midi_notes, corresp)
    perform_pairs = make_xml_midi_pair(xml_notes, midi_notes, match_between_xml_perf)
    for i, pair in enumerate(perform_pairs):
        if pair == []:
            score_pairs[i] == []
            continue
        pair['xml'].note_duration.time_position = pair['midi'].start
        pair['xml'].note_duration.seconds = pair['midi'].end - pair['midi'].start
        pair['xml'].velocity = pair['midi'].velocity
        score_pairs[i]['midi'].start = pair['midi'].start
        score_pairs[i]['midi'].end = pair['midi'].end
        score_pairs[i]['midi'].velocity = pair['midi'].velocity
    valid_position_pairs, _ = make_available_xml_midi_positions(perform_pairs)
    tempos = feature_utils.cal_tempo_by_positions(beat_positions, valid_position_pairs)
    output_features['beat_tempo'] =  [log(get_item_by_xml_position(tempos, note).qpm, 10) for note in xml_notes]
    output_features['velocity'] = [note.velocity for note in xml_notes]

    return [x['midi'] for x in score_pairs if x!=[]]

def regulate_tempo_by_measure_number(outputs, xml_notes, start_measure, end_measure):
    note_measure_numbers = [x.measure_number for x in xml_notes]
    start_note_index = note_measure_numbers.index(start_measure)
    end_note_index = note_measure_numbers.index(end_measure)

    mean_tempo = torch.mean(outputs[:, start_note_index:end_note_index, 0] )
    outputs[:, start_note_index:end_note_index, 0] = mean_tempo
    # outputs[:, start_note_index:end_note_index, 0] = mean_tempo + (outputs[:, start_note_index:end_note_index, 0] - mean_tempo) / 3

    return outputs



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-mode", "--sessMode", type=str,
#                         default='train', help="train or test or testAll")
#     parser.add_argument("-path", "--test_path", type=str,
#                         default="./test_pieces/bps_5_1/", help="folder path of test mat")
#     parser.add_argument("-tempo", "--startTempo", type=int,
#                         default=0, help="start tempo. zero to use xml first tempo")

#     model = load_model
#     load_file_and_generate_performance(args.test_path, args)