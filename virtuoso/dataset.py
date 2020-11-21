from virtuoso.model_utils import make_higher_node
from virtuoso.pyScoreParser.data_class import DataSet
import numpy as np
import torch
import pickle
import random
import math

# from .pyScoreParser import xml_matching
from pathlib import Path
from .utils import load_dat
from .data_process import make_slicing_indexes_by_measure, make_slice_with_same_measure_number, key_augmentation
from . import graph

class ScorePerformDataset:
    def __init__(self, path, type, len_slice, graph_keys, hier_type=[], device='cuda'):
        # type = one of ['train', 'valid', 'test']
        path = Path(path)
        self.path = path / type
        stat_data = load_dat(path/"stat.dat")
        self.stats = stat_data
        self.device = device

        self.data_paths = list(self.path.glob("*.dat"))
        self.data = [load_dat(x) for x in self.data_paths]
        self.len_slice = len_slice
        self.len_graph_slice = 400
        self.graph_margin = 100
        if len(graph_keys)>0:
            self.is_graph = True
            self.graph_keys = graph_keys
            self.stats['graph_keys'] = graph_keys
        else:
            self.is_graph = False
            self.stats['graph_keys'] = []
        hier_keys = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas']
        for key in hier_keys:
            if key in hier_type:
                setattr(self, key, True)
            else:
                setattr(self, key, False)

        self.update_slice_info()

    def update_slice_info(self):
        self.slice_info = []
        for i, data in enumerate(self.data):
            data_size = len(data['input_data'])
            measure_numbers = data['note_location']['measure']
            if self.is_hier and self.hier_meas:
                slice_indices = make_slice_with_same_measure_number(data_size, measure_numbers, measure_steps=self.len_slice)
            else:
                slice_indices = make_slicing_indexes_by_measure(data_size, measure_numbers, steps=self.len_slice)
            for idx in slice_indices:
                self.slice_info.append((i, idx))
    
    def __getitem__(self, index):
        idx, sl_idx = self.slice_info[index]
        data = self.data[idx]
        batch_start, batch_end = sl_idx

        aug_key = random.randrange(-5, 7)
        batch_x = torch.Tensor(key_augmentation(data['input_data'][batch_start:batch_end], aug_key, self.stats['stats']["midi_pitch"]["stds"]))
        if self.in_hier:
            if self.hier_meas:
                batch_x = torch.cat((batch_x, torch.Tensor(data['meas_level_data'][batch_start:batch_end])), dim=-1)
        if self.is_hier:
            if self.hier_meas:
                batch_y = torch.Tensor(data['meas_level_data'][batch_start:batch_end])
        else:
            batch_y = torch.Tensor(data['output_data'][batch_start:batch_end])
        note_locations = {
            'beat': torch.Tensor(data['note_location']['beat'][batch_start:batch_end]).type(torch.int32),
            'measure': torch.Tensor(data['note_location']['measure'][batch_start:batch_end]).type(torch.int32),
            'section': torch.Tensor(data['note_location']['section'][batch_start:batch_end]).type(torch.int32),
            'voice': torch.Tensor(data['note_location']['voice'][batch_start:batch_end]).type(torch.int32),
        }

        align_matched = torch.Tensor(data['align_matched'][batch_start:batch_end])
        articulation_loss_weight = torch.Tensor(data['articulation_loss_weight'][batch_start:batch_end])
        if self.is_graph:
            graphs = graph.edges_to_matrix_short(data['graph'], sl_idx, self.graph_keys)
            if self.len_graph_slice != self.len_slice:
                graphs = split_graph_to_batch(graphs, self.len_graph_slice, self.graph_margin)
        else:
            graphs = None
        return batch_x, batch_y, note_locations, align_matched, articulation_loss_weight, graphs

    def __len__(self):
        return len(self.slice_info)

def split_graph_to_batch(graphs, len_slice, len_margin):
    if graphs.shape[1] < len_slice:
        return graphs
    num_types = graphs.shape[0]
    num_batch = 1 + math.ceil( (graphs.shape[1] - len_slice) / (len_slice - len_margin*2) )
    input_split = torch.zeros((num_batch * num_types, len_slice, len_slice)).to(graphs.device)
    hop_size = len_slice - len_margin * 2
    for i in range(num_batch-1):
        input_split[i*num_types:(i+1)*num_types] = graphs[:, hop_size*i:hop_size*i+len_slice, hop_size*i:hop_size*i+len_slice]
    input_split[-num_types:] = graphs[:,-len_slice:, -len_slice:]
    return input_split

class FeatureCollate:
    # def __init__(self, device='cuda'):
    #     self.device= device
    def __call__(self, batch):
        if len(batch) == 1:
            batch_x, batch_y, note_locations, align_matched, pedal_status, edges = batch[0]
            return (batch_x.unsqueeze(0), 
                    batch_y.unsqueeze(0), 
                    note_locations, 
                    align_matched.view(1,-1,1), 
                    pedal_status.view(1,-1,1), 
                    edges
            )
        else:
            return batch


def load_file_and_encode_style(path, perf_name, composer_name):
    test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(
        path, perf_name, composer_name, MEANS, STDS)
    qpm_primo = test_x[0][4]

    test_x, test_y = handle_data_in_tensor(
        test_x, test_y, hierarchy_test=IN_HIER)
    edges = edges_to_matrix(edges, test_x.shape[0])

    if IN_HIER:
        test_x = test_x.view((1, -1, HIER_MODEL.input_size))
        hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
        perform_z_high = encode_performance_style_vector(
            test_x, hier_y, edges, note_locations, model=HIER_MODEL)
        hier_outputs, _ = run_model_in_steps(
            test_x, hier_y, edges, note_locations, model=HIER_MODEL)
        if HIER_MEAS:
            hierarchy_numbers = [x.measure for x in note_locations]
        elif HIER_BEAT:
            hierarchy_numbers = [x.beat for x in note_locations]
        hier_outputs_spanned = HIER_MODEL.span_beat_to_note_num(
            hier_outputs, hierarchy_numbers, test_x.shape[1], 0)
        input_concat = torch.cat((test_x, hier_outputs_spanned), 2)
        batch_y = test_y[1].view(1, -1, MODEL.output_size)
        perform_z_note = encode_performance_style_vector(
            input_concat, batch_y, edges, note_locations, model=MODEL)
        perform_z = [perform_z_high, perform_z_note]

    else:
        batch_x = test_x.view((1, -1, NUM_INPUT))
        batch_y = test_y.view((1, -1, NUM_OUTPUT))
        perform_z = encode_performance_style_vector(
            batch_x, batch_y, edges, note_locations)
        perform_z = [perform_z]

    return perform_z, qpm_primo


#>>>>>>>>>>>>>> maybe to be removed
def load_all_file_and_encode_style(parsed_data, measure_only=False, emotion_data=False):
    total_z = []
    perf_name_list = []
    num_piece = len(parsed_data[0])
    for i in range(num_piece):
        piece_test_x = parsed_data[0][i]
        piece_test_y = parsed_data[1][i]
        piece_edges = parsed_data[2][i]
        piece_note_locations = parsed_data[3][i]
        piece_perf_name = parsed_data[4][i]
        num_perf = len(piece_test_x)
        if num_perf == 0:
            continue
        piece_z = []
        for j in range(num_perf):
            # test_x, test_y, edges, note_locations, perf_name = perf
            if measure_only:
                test_x, test_y = handle_data_in_tensor(
                    piece_test_x[j], piece_test_y[j], hierarchy_test=IN_HIER)
                edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
                test_x = test_x.view((1, -1, HIER_MODEL.input_size))
                hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
                perform_z_high = encode_performance_style_vector(
                    test_x, hier_y, edges, piece_note_locations[j], model=HIER_MODEL)
            else:
                test_x, test_y = handle_data_in_tensor(
                    piece_test_x[j], piece_test_y[j], hierarchy_test=False)
                edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
                test_x = test_x.view((1, -1, MODEL.input_size))
                test_y = test_y.view(1, -1, MODEL.output_size)
                perform_z_high = encode_performance_style_vector(test_x, test_y, edges, piece_note_locations[j],
                                                                 model=MODEL)
            # perform_z_high = perform_z_high.reshape(-1).cpu().numpy()
            # piece_z.append(perform_z_high)
            # perf_name_list.append(piece_perf_name[j])

            perform_z_high = [z.reshape(-1).cpu().numpy()
                              for z in perform_z_high]
            piece_z += perform_z_high
            perf_name_list += [piece_perf_name[j]] * len(perform_z_high)
        if emotion_data:
            for i, name in enumerate(piece_perf_name):
                if name[-2:] == 'E1':
                    or_idx = i
                    break
            or_z = piece_z.pop(or_idx)
            piece_z = np.asarray(piece_z)
            piece_z -= or_z
            perf_name_list.pop(-(5-or_idx))
        else:
            piece_z = np.asarray(piece_z)
            average_piece_z = np.average(piece_z, axis=0)
            piece_z -= average_piece_z
        total_z.append(piece_z)
    total_z = np.concatenate(total_z, axis=0)
    return total_z, perf_name_list
#<<<<<<<<<<<<<< 


def encode_all_emotionNet_data(path_list, style_keywords):
    perform_z_by_emotion = []
    perform_z_list_by_subject = []
    qpm_list_by_subject = []
    num_style = len(style_keywords)
    if IN_HIER:
        num_model = 2
    else:
        num_model = 1
    for pair in path_list:
        subject_num = pair[2]
        for sub_idx in range(subject_num):
            indiv_perform_z = []
            indiv_qpm = []
            path = cons.emotion_folder_path + pair[0] + '/'
            composer_name = pair[1]
            for key in style_keywords:
                perf_name = key + '_sub' + str(sub_idx+1)
                perform_z_li, qpm_primo = load_file_and_encode_style(
                    path, perf_name, composer_name)
                perform_z_li = [torch.mean(torch.stack(z), 0, True)
                                for z in perform_z_li]
                indiv_perform_z.append(perform_z_li)
                indiv_qpm.append(qpm_primo)
            for i in range(1, num_style):
                for j in range(num_model):
                    indiv_perform_z[i][j] = indiv_perform_z[i][j] - \
                        indiv_perform_z[0][j]
                indiv_qpm[i] = indiv_qpm[i] - indiv_qpm[0]
            perform_z_list_by_subject.append(indiv_perform_z)
            qpm_list_by_subject.append(indiv_qpm)
    for i in range(num_style):
        z_by_models = []
        for j in range(num_model):
            emotion_mean_z = []
            for z_list in perform_z_list_by_subject:
                emotion_mean_z.append(z_list[i][j])
            mean_perform_z = torch.mean(torch.stack(emotion_mean_z), 0, True)
            z_by_models.append(mean_perform_z)
        if i is not 0:
            emotion_qpm = []
            for qpm_change in qpm_list_by_subject:
                emotion_qpm.append(qpm_change[i])
            mean_qpm_change = np.mean(emotion_qpm)
        else:
            mean_qpm_change = 0
        print(style_keywords[i], z_by_models, mean_qpm_change)
        perform_z_by_emotion.append(
            {'z': z_by_models, 'key': style_keywords[i], 'qpm': mean_qpm_change})

    return perform_z_by_emotion
    # with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
    #     pickle.dump(mean_perform_z, f, protocol=2)


def load_stat(args):
    with open(args.dataName + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        if args.trainingLoss == 'CE':
            MEANS, STDS, BINS = u.load()
            new_prime_param = 0
            new_trill_param = 0
            for i in range(NUM_PRIME_PARAM):
                new_prime_param += len(BINS[i]) - 1
            for i in range(NUM_PRIME_PARAM, NUM_PRIME_PARAM + num_trill_param - 1):
                new_trill_param += len(BINS[i]) - 1
            NUM_PRIME_PARAM = new_prime_param
            print('New NUM_PRIME_PARAM: ', NUM_PRIME_PARAM)
            num_trill_param = new_trill_param + 1
            NUM_OUTPUT = NUM_PRIME_PARAM + num_trill_param
            NUM_TEMPO_PARAM = len(BINS[0]) - 1
        else:
            MEANS, STDS = u.load()
            BINS = None
    return MEANS, STDS, BINS


def read_xml_to_array(path_name, means, stds, start_tempo, composer_name, vel_standard):
    # TODO: update to adapt pyScoreParser
    xml_object, xml_notes = xml_matching.read_xml_to_notes(path_name)
    beats = xml_object.get_beat_positions()
    measure_positions = xml_object.get_measure_positions()
    features = xml_matching.extract_score_features(
        xml_notes, measure_positions, beats, qpm_primo=start_tempo, vel_standard=vel_standard)
    features = make_index_continuous(features, score=True)
    composer_vec = composer_name_to_vec(composer_name)
    edges = score_graph.make_edge(xml_notes)

    for i in range(len(stds[0])):
        if stds[0][i] < 1e-4 or isinstance(stds[0][i], complex):
            stds[0][i] = 1

    test_x = []
    note_locations = []
    for feat in features:
        temp_x = [(feat.midi_pitch - means[0][0]) / stds[0][0], (feat.duration - means[0][1]) / stds[0][1],
                  (feat.beat_importance -
                   means[0][2])/stds[0][2], (feat.measure_length-means[0][3])/stds[0][3],
                  (feat.qpm_primo - means[0][4]) /
                  stds[0][4], (feat.following_rest - means[0][5]) / stds[0][5],
                  (feat.distance_from_abs_dynamic - means[0][6]) / stds[0][6],
                  (feat.distance_from_recent_tempo - means[0][7]) / stds[0][7],
                  feat.beat_position, feat.xml_position, feat.grace_order,
                  feat.preceded_by_grace_note, feat.followed_by_fermata_rest] \
            + feat.pitch + feat.tempo + feat.dynamic + feat.time_sig_vec + \
            feat.slur_beam_vec + composer_vec + feat.notation + feat.tempo_primo
        # temp_x.append(feat.is_beat)
        test_x.append(temp_x)
        note_locations.append(feat.note_location)

    return test_x, xml_notes, xml_object, edges, note_locations


class PerformDataset():
    def __init__(self, data_path, split, graph=False, samples=None):
        return
    def __len__(self):
        return NumberOfPieces

    def files(self):
        return NotImplementedError

class YamahaDataset(PerformDataset):
    def __init__(self, data_path, split, graph=False, samples=None):
        return
    def __getitem__(self, index):
        return input_features, output_features, score_graph

    def __len__(self):
        return NumberOfSegments
    
    def files(self):
        # load yamaha set data utilize pyScoreParser.PieceData
        return NotImplementedError