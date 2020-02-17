import numpy as np
import torch as th
import pickle
import _pickle as cPickle
from abc import abstractmethod
from collections import defaultdict, namedtuple
from tqdm import tqdm
from bisect import bisect_left, bisect_right
from pathlib import Path

from pyScoreParser import xml_matching


ChunkInfo = namedtuple('ChunkInfo', ['file_index', 'local_index', 'midi_index'])

'''
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
'''


class PerformDataset():
    def __init__(self, path, split, graph=False, samples=None, stride=None):
        self.path = path
        self.split = split
        self.graph = graph
        self.samples = samples
        if stride is None:
            stride = samples
        self.stride = stride

        # iterate over whole dataset to get length
        self.files = self.feature_files()
        self.num_chunks = 0
        self.entry = [] 

        for file_idx in tqdm(range(len(self.files))):
            if samples is None:
                self.entry.append(ChunkInfo(file_idx, 0, 0))
                continue
            feature_file = self.files[file_idx]
            feature_lists = self.load_file(feature_file)
            length = feature_lists['input_data'].shape[0]

            if length < samples:
                continue
            num_segs = (length - samples) // stride + 1
            for seg_idx in range(num_segs):
                self.entry.append(ChunkInfo(file_idx, seg_idx, seg_idx*stride))


    def __getitem__(self, index):
        file_idx, seg_idx, note_offset = self.entry[index]
        feature = self.load_file(self.files[file_idx])
        if self.samples is None:
            return feature
        else:
            feature['input_data'] = \
                feature['input_data'][note_offset:note_offset+self.samples, :]
            feature['output_data'] = feature['output_data'][note_offset:note_offset+self.samples, :]
            feature['note_location'] = feature['note_location'][note_offset:note_offset+self.samples]
            feature['align_matched'] = feature['align_matched'][note_offset:note_offset+self.samples]
            feature['articulation_loss_weight'] = feature['articulation_loss_weight'][note_offset:note_offset+self.samples]
            notes_in_graph = [el[0] for el in feature['graph']]
            idx_left = bisect_left(notes_in_graph, note_offset)
            idx_right = bisect_right(notes_in_graph, note_offset + self.samples)
            feature['graph'] = feature['graph'][idx_left: idx_right]
            return feature
        
    @classmethod
    @abstractmethod
    def feature_files(self):
        ''' return feature .dat file paths'''
        raise NotImplementedError

    def __len__(self):
        return len(self.entry)

    @staticmethod
    def load_file(data_path):
        with open(data_path, "rb") as f:
            u = cPickle.Unpickler(f)
            features = u.load()
        return features

class YamahaDataset(PerformDataset):
    def __init__(self, path, split, graph=False, samples=None, stride=None):
        super().__init__(path, split, graph, samples, stride)
    
    def feature_files(self):
        return sorted(Path(self.path).glob(f'{self.split}/*.dat'))