import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import argparse
import math
import numpy as np
import asyncio
import shutil
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import performanceWorm
import copy
import random
import xml_matching
import nnModel
import model_parameters as param
import model_constants as cons

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test or testAll")
# parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./test_pieces/mozart545-1/", help="folder path of test mat")
# parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
parser.add_argument("-data", "--dataName", type=str, default="very_short", help="dat file name")
parser.add_argument("--resume", type=str, default="_best.pth.tar", help="best model path")
parser.add_argument("-tempo", "--startTempo", type=int, default=0, help="start tempo. zero to use xml first tempo")
parser.add_argument("-trill", "--trainTrill", default=False, type=lambda x: (str(x).lower() == 'true'), help="train trill")
parser.add_argument("--beatTempo", type=bool, default=True, help="cal tempo from beat level")
parser.add_argument("-voice", "--voiceEdge", default=True, type=lambda x: (str(x).lower() == 'true'), help="network in voice level")
parser.add_argument("-vel", "--velocity", type=str, default='50,65', help="mean velocity of piano and forte")
parser.add_argument("-dev", "--device", type=int, default=0, help="cuda device number")
parser.add_argument("-code", "--modelCode", type=str, default='isgn_test', help="code name for saving the model")
parser.add_argument("-tCode", "--trillCode", type=str, default='default', help="code name for loading trill model")
parser.add_argument("-comp", "--composer", type=str, default='Chopin', help="composer name of the input piece")
parser.add_argument("--latent", type=float, default=0, help='initial_z value')
parser.add_argument("-bp", "--boolPedal", default=False, type=lambda x: (str(x).lower() == 'true'), help='make pedal value zero under threshold')
parser.add_argument("-loss", "--trainingLoss", type=str, default='MSE', help='type of training loss')
parser.add_argument("-reTrain", "--resumeTraining", default=False, type=lambda x: (str(x).lower() == 'true'), help='resume training after loading model')
parser.add_argument("-perf", "--perfName", default='Anger_sub1', type=str, help='resume training after loading model')


args = parser.parse_args()
LOSS_TYPE = args.trainingLoss

### parameters
learning_rate = 0.0003
TIME_STEPS = 450
VALID_STEPS = 3000
NUM_UPDATED = 0
print('Learning Rate and Time Steps are ', learning_rate, TIME_STEPS)
num_epochs = 100
num_key_augmentation = 1


SCORE_INPUT = 77 #score information only
DROP_OUT = 0.5
TOTAL_OUTPUT = 16

NUM_PRIME_PARAM = 11
NUM_TEMPO_PARAM = 1
VEL_PARAM_IDX = 1
DEV_PARAM_IDX = 2
PEDAL_PARAM_IDX = 3
num_second_param = 0
num_trill_param = 5
num_voice_feed_param = 0 # velocity, onset deviation
num_tempo_info = 0
num_dynamic_info = 0 # distance from marking, dynamics vector 4, mean_piano, forte marking and velocity = 4
is_trill_index_score = -11
is_trill_index_concated = -11 - (NUM_PRIME_PARAM + num_second_param)


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
        TOTAL_OUTPUT = NUM_PRIME_PARAM + num_trill_param
        NUM_TEMPO_PARAM = len(BINS[0]) - 1
    else:
        MEANS, STDS = u.load()

QPM_INDEX = 0
# VOICE_IDX = 11
TEMPO_IDX = 25
PITCH_IDX = 12
QPM_PRIMO_IDX = 4
TEMPO_PRIMO_IDX = -2
GRAPH_KEYS = ['onset', 'forward', 'melisma', 'rest', 'slur']
if args.voiceEdge:
    GRAPH_KEYS.append('voice')
N_EDGE_TYPE = len(GRAPH_KEYS) * 2
# mean_vel_start_index = 7
# vel_vec_start_index = 33

batch_size = 1

torch.cuda.set_device(args.device)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not args.trainTrill:
    if args.sessMode == 'train':
        NET_PARAM = param.initialize_model_parameters_by_code(args.modelCode)
        NET_PARAM.num_edge_types = N_EDGE_TYPE
        param.save_parameters(NET_PARAM, args.modelCode + '_param')
    else:
        NET_PARAM = param.load_parameters(args.modelCode + '_param')
        TrillNET_Param = param.load_parameters(args.trillCode + '_trill_param')
        if not hasattr(NET_PARAM, 'num_edge_types'):
            NET_PARAM.num_edge_types = 10
        if not hasattr(TrillNET_Param, 'num_edge_types'):
            TrillNET_Param.num_edge_types = 10

        trill_model = nnModel.TrillGraph(TrillNET_Param, is_trill_index_concated, LOSS_TYPE, DEVICE).to(DEVICE)

    if 'ggnn_non_ar' in args.modelCode:
        MODEL = nnModel.GGNN_HAN(NET_PARAM, DEVICE, LOSS_TYPE, NUM_TEMPO_PARAM).to(DEVICE)
    elif 'ggnn_ar' in args.modelCode:
        MODEL = nnModel.GGNN_Recursive(NET_PARAM, DEVICE).to(DEVICE)
    elif 'ggnn_simple_ar' in args.modelCode:
        MODEL = nnModel.GGNN_Recursive(NET_PARAM, DEVICE).to(DEVICE)
    elif 'sequential_ggnn' in args.modelCode:
        MODEL = nnModel.Sequential_GGNN(NET_PARAM, DEVICE).to(DEVICE)
    elif 'sggnn_alt' in args.modelCode:
        MODEL = nnModel.SGGNN_Alt(NET_PARAM, DEVICE).to(DEVICE)
    elif 'sggnn_note' in args.modelCode:
        MODEL = nnModel.SGGNN_Note(NET_PARAM, DEVICE).to(DEVICE)
    elif 'isgn' in args.modelCode:
        MODEL = nnModel.ISGN(NET_PARAM, DEVICE).to(DEVICE)
    elif 'han' in args.modelCode:
        if 'ar' in args.modelCode:
            step_by_step = True
        else:
            step_by_step = False
        MODEL = nnModel.HAN_Integrated(NET_PARAM, DEVICE, step_by_step).to(DEVICE)
    else:
        print('Error: Unclassified model code')
        # Model = nnModel.HAN_VAE(NET_PARAM, DEVICE, False).to(DEVICE)

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=learning_rate)
    # second_optimizer = torch.optim.Adam(second_model.parameters(), lr=learning_rate)
else:
    TrillNET_Param = param.initialize_model_parameters_by_code(args.modelCode)
    TrillNET_Param.input_size = SCORE_INPUT + TOTAL_OUTPUT - num_trill_param
    TrillNET_Param.output_size = num_trill_param
    TrillNET_Param.note.size = 16
    TrillNET_Param.note.layer = 1
    TrillNET_Param.num_edge_types = N_EDGE_TYPE
    param.save_parameters(TrillNET_Param, args.modelCode + '_trill_param')

    trill_model = nnModel.TrillGraph(TrillNET_Param, is_trill_index_concated, LOSS_TYPE, DEVICE).to(DEVICE)

    trill_optimizer = torch.optim.Adam(trill_model.parameters(), lr=learning_rate)



### Model

# class PerformanceEncoder(GGNN_HAN):
#     def __init__(self, network_parameters):
#         super(perfor)


# model = BiRNN(input_size, hidden_size, num_layers, num_output).to(device)
# second_model = ExtraHAN(NET_PARAM).to(device)


if LOSS_TYPE == 'MSE':
    def criterion(pred, target, aligned_status=1):
        if isinstance(aligned_status, int):
            data_size = pred.shape[-2] * pred.shape[-1]
        else:
            data_size = torch.sum(aligned_status).item() * pred.shape[-1]
            if data_size ==0:
                data_size = 1
        return torch.sum( ((target - pred) ** 2) * aligned_status) / data_size
elif LOSS_TYPE == 'CE':
    # criterion = nn.CrossEntropyLoss()
    def criterion(pred, target, aligned_status=1):
        if isinstance(aligned_status, int):
            data_size = pred.shape[-2] * pred.shape[-1]
        else:
            data_size = torch.sum(aligned_status).item() * pred.shape[-1]
            if data_size ==0:
                data_size = 1
                print('data size for loss calculation is zero')
        return -1 * torch.sum((target * torch.log(pred)  + (1-target) * torch.log(1-pred)) * aligned_status) / data_size


def save_checkpoint(state, is_best, filename=args.modelCode, model_name='prime'):
    save_name = model_name + '_' + filename + '_checkpoint.pth.tar'
    torch.save(state, save_name)
    if is_best:
        best_name = model_name + '_' + filename + '_best.pth.tar'
        shutil.copyfile(save_name, best_name)


def key_augmentation(data_x, key_change):
    # key_change = 0
    if key_change == 0:
        return data_x
    data_x_aug = copy.deepcopy(data_x)
    pitch_start_index = PITCH_IDX
    # while key_change == 0:
    #     key_change = random.randrange(-5, 7)
    for data in data_x_aug:
        octave = data[pitch_start_index]
        pitch_class_vec = data[pitch_start_index+1:pitch_start_index+13]
        pitch_class = pitch_class_vec.index(1)
        new_pitch = pitch_class + key_change
        if new_pitch < 0:
            octave -= 0.25
        elif new_pitch > 12:
            octave += 0.25
        new_pitch = new_pitch % 12

        new_pitch_vec = [0] * 13
        new_pitch_vec[0] = octave
        new_pitch_vec[new_pitch+1] = 1

        data[pitch_start_index: pitch_start_index+13] = new_pitch_vec
        data[0] = data[0] + key_change

    return data_x_aug


def edges_to_matrix(edges, num_notes):
    num_keywords = len(GRAPH_KEYS)
    matrix = np.zeros((N_EDGE_TYPE, num_notes, num_notes))

    for edg in edges:
        if edg[2] not in GRAPH_KEYS:
            continue
        edge_type = GRAPH_KEYS.index(edg[2])
        matrix[edge_type, edg[0], edg[1]] = 1
        if edge_type != 0:
            matrix[edge_type+num_keywords, edg[1], edg[0]] = 1
        else:
            matrix[edge_type, edg[1], edg[0]] = 1

    matrix[num_keywords, :,:] = np.identity(num_notes)
    matrix = torch.Tensor(matrix)
    return matrix


def edges_to_sparse_tensor(edges):
    num_keywords = len(GRAPH_KEYS)
    edge_list = []
    edge_type_list = []

    for edg in edges:
        edge_type = GRAPH_KEYS.index(edg[2])
        edge_list.append(edg[0:2])
        edge_list.append([edg[1], edg[0]])
        edge_type_list.append(edge_type)
        if edge_type != 0:
            edge_type_list.append(edge_type+num_keywords)
        else:
            edge_type_list.append(edge_type)

        edge_list = torch.LongTensor(edge_list)
    edge_type_list = torch.FloatTensor(edge_type_list)

    matrix = torch.sparse.FloatTensor(edge_list.t(), edge_type_list)

    return matrix


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


def load_file_and_generate_performance(filename, composer=args.composer, z=args.latent, save_name='performed_by_nn'):
    path_name = filename
    composer_name = composer
    vel_pair = (int(args.velocity.split(',')[0]), int(args.velocity.split(',')[1]))
    test_x, xml_notes, xml_doc, edges, note_locations = xml_matching.read_xml_to_array(path_name, MEANS, STDS,
                                                                                       args.startTempo, composer_name,
                                                                                       vel_pair)
    batch_x = torch.Tensor(test_x)
    for i in range(len(STDS)):
        for j in range(len(STDS[i])):
            if STDS[i][j] < 1e-4:
                STDS[i][j] = 1

    if args.startTempo == 0:
        start_tempo = xml_notes[0].state_fixed.qpm / 60 * xml_notes[0].state_fixed.divisions
        start_tempo = math.log(start_tempo, 10)
        # start_tempo_norm = (start_tempo - means[1][0]) / stds[1][0]
    else:
        start_tempo = math.log(args.startTempo, 10)
    input_y = torch.zeros(1, 1, TOTAL_OUTPUT)

    #
    if LOSS_TYPE == 'MSE':
        start_tempo_norm = (start_tempo - MEANS[1][0]) / STDS[1][0]
        input_y[0, 0, 0] = start_tempo_norm
        for i in range(1, TOTAL_OUTPUT - 1):
            input_y[0, 0, i] -= MEANS[1][i]
            input_y[0, 0, i] /= STDS[1][i]
        input_y = input_y.to(DEVICE)
        tempo_stats = [MEANS[1][0], STDS[1][0]]
    else:
        tempo_stats = [0, 0]

    if type(z) is dict:
        initial_z = z['z']
        qpm_change = z['qpm']
        z = z['key']
        batch_x[:,QPM_PRIMO_IDX] = batch_x[:,QPM_PRIMO_IDX] + qpm_change
    else:
        initial_z = [z] * NET_PARAM.encoder.size

    batch_x = batch_x.to(DEVICE).view(1, -1, SCORE_INPUT)
    graph = edges_to_matrix(edges, batch_x.shape[1])
    MODEL.is_teacher_force = False
    prediction = perform_xml(batch_x, input_y, graph, note_locations, tempo_stats, initial_z=initial_z)

    prediction = np.squeeze(np.asarray(prediction))
    num_notes = len(prediction)
    if LOSS_TYPE == 'MSE':
        for i in range(15):
            prediction[:, i] *= STDS[1][i]
            prediction[:, i] += MEANS[1][i]
    elif LOSS_TYPE == 'CE':
        prediction_in_value = np.zeros((num_notes, 16))
        for i in range(num_notes):
            bin_range_start = 0
            for j in range(15):
                feature_bin_size = len(BINS[j]) - 1
                feature_class = np.argmax(prediction[i, bin_range_start:bin_range_start + feature_bin_size])
                feature_value = (BINS[j][feature_class] + BINS[j][feature_class + 1]) / 2
                prediction_in_value[i, j] = feature_value
                bin_range_start += feature_bin_size
            prediction_in_value[i, 15] = prediction[i, -1]
        prediction = prediction_in_value
    output_features = []
    num_notes = len(xml_notes)
    for i in range(num_notes):
        pred = prediction[i]
        # feat = {'IOI_ratio': pred[0], 'articulation': pred[1], 'loudness': pred[2], 'xml_deviation': 0,
        feat = xml_matching.MusicFeature()
        feat.qpm = pred[0]
        feat.velocity = pred[1]
        feat.xml_deviation = pred[2]
        feat.articulation = pred[3]
        # feat.xml_deviation = 0
        feat.pedal_refresh_time = pred[4]
        feat.pedal_cut_time = pred[5]
        feat.pedal_at_start = pred[6]
        feat.pedal_at_end = pred[7]
        feat.soft_pedal = pred[8]
        feat.pedal_refresh = pred[9]
        feat.pedal_cut = pred[10]

        feat.beat_index = note_locations[i].beat
        feat.measure_index = note_locations[i].measure

        feat.trill_param = pred[11:16]
        feat.trill_param[0] = feat.trill_param[0]
        feat.trill_param[1] = (feat.trill_param[1])
        feat.trill_param[2] = (feat.trill_param[2])
        feat.trill_param[3] = (feat.trill_param[3])
        feat.trill_param[4] = round(feat.trill_param[4])

        # if test_x[i][is_trill_index_score] == 1:
        #     print(feat.trill_param)
        output_features.append(feat)

    output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                           predicted=True)
    output_midi = xml_matching.xml_notes_to_midi(output_xml)
    piece_name = path_name.split('/')
    save_name = 'test_result/' + piece_name[-2] + '_by_' + args.modelCode + '_z' + str(z)

    performanceWorm.plot_performance_worm(output_features, save_name + '.png')
    xml_matching.save_midi_notes_as_piano_midi(output_midi, save_name + '.mid',
                                               bool_pedal=args.boolPedal, disklavier=True)


def load_file_and_encode_style(path, perf_name, composer_name):
    test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(path, perf_name, composer_name, MEANS, STDS)
    qpm_primo = test_x[0][4]
    test_x = torch.Tensor(test_x).to(DEVICE).view(1, -1, SCORE_INPUT)
    test_y = torch.Tensor(test_y).to(DEVICE).view(1, -1, TOTAL_OUTPUT)
    edges = edges_to_matrix(edges, test_x.shape[1])
    perform_z = encode_performance_style_vector(test_x, test_y, edges, note_locations)
    return perform_z, qpm_primo


def encode_performance_style_vector(input, input_y, edges, note_locations):
    with torch.no_grad():
        model_eval = MODEL.eval()
        prime_input_y = input_y[:, :, 0:NUM_PRIME_PARAM].view(1, -1, NUM_PRIME_PARAM)
        batch_graph = edges.to(DEVICE)
        encoded_z = model_eval(input, prime_input_y, batch_graph,
                               note_locations=note_locations, start_index=0, return_z=True)
    return encoded_z


def encode_all_emotionNet_data(path_list, style_keywords):
    perform_z_by_emotion = []
    perform_z_list_by_subject = []
    qpm_list_by_subject = []
    num_style = len(style_keywords)
    for pair in path_list:
        subject_num = pair[2]
        for sub_idx in range(subject_num):
            indiv_perform_z = []
            indiv_qpm = []
            path = cons.emotion_folder_path + pair[0] + '/'
            composer_name = pair[1]
            for key in style_keywords:
                perf_name = key + '_sub' + str(sub_idx+1)
                perform_z, qpm_primo = load_file_and_encode_style(path, perf_name, composer_name)
                indiv_perform_z.append(perform_z)
                indiv_qpm.append(qpm_primo)
            for i in range(1,num_style):
                indiv_perform_z[i] = indiv_perform_z[i] - indiv_perform_z[0]
                indiv_qpm[i] = indiv_qpm[i] - indiv_qpm[0]
            perform_z_list_by_subject.append(indiv_perform_z)
            qpm_list_by_subject.append(indiv_qpm)
    for i in range(num_style):
        emotion_mean_z = []
        for z_list in perform_z_list_by_subject:
            emotion_mean_z.append(z_list[i])
        mean_perform_z = torch.mean(torch.stack(emotion_mean_z), 0, True)
        if i is not 0:
            emotion_qpm = []
            for qpm_change in qpm_list_by_subject:
                emotion_qpm.append(qpm_change[i])
            mean_qpm_change = np.mean(emotion_qpm)
        else:
            mean_qpm_change = 0
        print(style_keywords[i], mean_perform_z, mean_qpm_change)
        perform_z_by_emotion.append({'z': mean_perform_z, 'key': style_keywords[i], 'qpm': mean_qpm_change})

    return perform_z_by_emotion
        # with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
        #     pickle.dump(mean_perform_z, f, protocol=2)


def perform_xml(input, input_y, edges, note_locations, tempo_stats, valid_y = None, initial_z=False):
    num_notes = input.shape[1]
    total_valid_batch = int(math.ceil(num_notes / VALID_STEPS))
    with torch.no_grad():  # no need to track history in validation
        if not args.trainTrill:
            model_eval = MODEL.eval()
            if args.sessMode == 'test' or args.trainTrill:
                trill_model_eval = trill_model.eval()

            total_output = []
            if num_notes < VALID_STEPS:
                if input_y.shape[1] > 1:
                    prime_input_y = input_y[:, :, 0:NUM_PRIME_PARAM].view(1, -1, NUM_PRIME_PARAM)
                else:
                    prime_input_y = input_y[:, :, 0:NUM_PRIME_PARAM].view(1, 1, NUM_PRIME_PARAM)
                batch_graph = edges.to(DEVICE)
                prime_outputs, _, _, note_hidden_out = model_eval(input, prime_input_y, batch_graph,
                                                                  note_locations=note_locations, start_index=0,
                                                                  initial_z=initial_z)
                # second_inputs = torch.cat((input,prime_outputs), 2)
                # second_input_y = input_y[:,:,num_prime_param:num_prime_param+num_second_param].view(1,-1,num_second_param)
                # model_eval = second_model.eval()
                # second_outputs = model_eval(second_inputs, second_input_y, note_locations, 0, step_by_step=True)
                if args.sessMode == 'test' and torch.sum(input[:, :, is_trill_index_score]) > 0:
                    trill_inputs = torch.cat((input, prime_outputs), 2)
                    trill_outputs = trill_model_eval(trill_inputs, batch_graph)
                else:
                    trill_outputs = torch.zeros(1, num_notes, num_trill_param).to(DEVICE)

                outputs = torch.cat((prime_outputs, trill_outputs), 2)
            else:
                for i in range(total_valid_batch):
                    batch_start = i * VALID_STEPS
                    if i == total_valid_batch-1:
                        batch_end = num_notes
                    else:
                        batch_end = (i+1) * VALID_STEPS
                    if input_y.shape[1] > 1:
                        prime_input_y = input_y[:,batch_start:batch_end, 0:NUM_PRIME_PARAM].view(1, -1, NUM_PRIME_PARAM)
                    else:
                        prime_input_y = input_y[:, :, 0:NUM_PRIME_PARAM].view(1, 1, NUM_PRIME_PARAM)
                    batch_input = input[:,batch_start:batch_end,:]
                    batch_graph = edges[:,batch_start:batch_end, batch_start:batch_end].to(DEVICE)
                    prime_outputs, _, _, note_hidden_out = model_eval(batch_input, prime_input_y, batch_graph, note_locations=note_locations, start_index=0, initial_z=initial_z)
                    # second_inputs = torch.cat((input,prime_outputs), 2)
                    # second_input_y = input_y[:,:,num_prime_param:num_prime_param+num_second_param].view(1,-1,num_second_param)
                    # model_eval = second_model.eval()
                    # second_outputs = model_eval(second_inputs, second_input_y, note_locations, 0, step_by_step=True)
                    if args.sessMode == 'test' and torch.sum(input[:, :, is_trill_index_score]) > 0:
                        trill_inputs = torch.cat((batch_input, prime_outputs), 2)
                        trill_outputs = trill_model_eval(trill_inputs, batch_graph)
                    else:
                        trill_outputs = torch.zeros(1, batch_end-batch_start, num_trill_param).to(DEVICE)

                    temp_outputs = torch.cat((prime_outputs, trill_outputs),2)
                    total_output.append(temp_outputs)
                outputs = torch.cat(total_output, 1)
            return outputs
        else:
            trill_model_eval = trill_model.eval()
            if num_notes < VALID_STEPS:
                batch_graph = edges.to(DEVICE)
                if torch.sum(input[:, :, is_trill_index_score]) > 0:
                    trill_inputs = torch.cat((input, input_y[:,:,:-num_trill_param]), 2)
                    trill_outputs = trill_model_eval(trill_inputs, batch_graph)
                else:
                    trill_outputs = torch.zeros(1, num_notes, num_trill_param).to(DEVICE)

                outputs = torch.cat((input_y[:,:,:-num_trill_param], trill_outputs), 2)
            else:
                total_output = []
                for i in range(total_valid_batch):
                    batch_start = i * VALID_STEPS
                    if i == total_valid_batch - 1:
                        batch_end = num_notes
                    else:
                        batch_end = (i + 1) * VALID_STEPS
                    batch_input = input[:, batch_start:batch_end, :]
                    batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end].to(DEVICE)
                    batch_prime_output = input_y[:, batch_start:batch_end, :-num_trill_param]
                    # second_inputs = torch.cat((input,prime_outputs), 2)
                    # second_input_y = input_y[:,:,num_prime_param:num_prime_param+num_second_param].view(1,-1,num_second_param)
                    # model_eval = second_model.eval()
                    # second_outputs = model_eval(second_inputs, second_input_y, note_locations, 0, step_by_step=True)
                    if args.sessMode == 'test' and torch.sum(input[:, :, is_trill_index_score]) > 0:
                        trill_inputs = torch.cat((batch_input, batch_prime_output), 2)
                        trill_outputs = trill_model_eval(trill_inputs, batch_graph)
                    else:
                        trill_outputs = torch.zeros(1, batch_end - batch_start, num_trill_param).to(DEVICE)

                    temp_outputs = torch.cat((batch_prime_output, trill_outputs), 2)
                    total_output.append(temp_outputs)
                outputs = torch.cat(total_output, 1)
            return outputs


def batch_time_step_run(x, y, edges, note_locations, align_matched, slice_index, model, batch_size=batch_size, kld_weight=1):
    batch_start, batch_end = slice_index
    num_notes = batch_end - batch_start
    batch_x = torch.Tensor(x[batch_start:batch_end])
    batch_y = torch.Tensor(y[batch_start:batch_end])
    align_matched = torch.Tensor(align_matched[batch_start:batch_end])
    batch_x = batch_x.view((batch_size, -1, SCORE_INPUT)).to(DEVICE)
    batch_y = batch_y.view((batch_size, -1, TOTAL_OUTPUT)).to(DEVICE)
    align_matched = align_matched.view((batch_size, -1, 1)).to(DEVICE)
    batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end].to(DEVICE)

    if not args.trainTrill:
        prime_batch_x = batch_x
        prime_batch_y = batch_y[:, :, 0:NUM_PRIME_PARAM]


        model_train = model.train()
        prime_outputs, perform_mu, perform_var, total_out_list \
            = model_train(prime_batch_x, prime_batch_y, batch_graph, note_locations, batch_start)

        # prime_outputs *= align_matched
        # prime_batch_y *= align_matched

        if 'isgn' in args.modelCode:
            prime_loss = torch.zeros(1).to(DEVICE)
            for out in total_out_list:
                tempo_loss = cal_tempo_loss_in_beat(out, prime_batch_y, note_locations, batch_start)
                vel_loss = criterion(out[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX],
                                     prime_batch_y[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX], align_matched)
                dev_loss = criterion(out[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX],
                                     prime_batch_y[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX], align_matched)
                pedal_loss = criterion(out[:, :, PEDAL_PARAM_IDX:], prime_batch_y[:, :, PEDAL_PARAM_IDX:],
                                       align_matched)

                prime_loss += (tempo_loss + vel_loss + dev_loss / 2 + pedal_loss * 8) / 10.5
            prime_loss /= len(total_out_list)
        else:
            tempo_loss = cal_tempo_loss_in_beat(prime_outputs, prime_batch_y, note_locations, batch_start)
            vel_loss = criterion(prime_outputs[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX],
                                 prime_batch_y[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX], align_matched)
            dev_loss = criterion(prime_outputs[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX],
                                 prime_batch_y[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX], align_matched)
            pedal_loss = criterion(prime_outputs[:, :, PEDAL_PARAM_IDX:], prime_batch_y[:, :, PEDAL_PARAM_IDX:], align_matched)

            prime_loss = (tempo_loss + vel_loss + dev_loss / 2 + pedal_loss * 8) / 10.5

        if isinstance(perform_mu, bool):
            perform_kld = torch.zeros(1)
        else:
            perform_kld = -0.5 * torch.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
            prime_loss += perform_kld * kld_weight
        optimizer.zero_grad()
        prime_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        return tempo_loss, vel_loss, dev_loss, pedal_loss, torch.zeros(1), perform_kld

    else:
        trill_bool = batch_x[:,:,is_trill_index_score:is_trill_index_score+1]
        if torch.sum(trill_bool) > 0:
            trill_batch_x = torch.cat((batch_x, batch_y[:,:, 0:NUM_PRIME_PARAM + num_second_param]), 2)
            trill_batch_y = batch_y[:,:,-num_trill_param:]
            model_train = model.train()
            trill_output = model_train(trill_batch_x, batch_graph)
            trill_loss = criterion(trill_output, trill_batch_y, trill_bool)
            trill_optimizer.zero_grad()
            trill_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            trill_optimizer.step()
        else:
            trill_loss = torch.zeros(1)

        return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), trill_loss, torch.zeros(1)

    # loss = criterion(outputs, batch_y)
    # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])


def cal_tempo_loss_in_beat(pred_x, true_x, note_locations, start_index):
    previous_beat = -1
    num_notes = pred_x.shape[1]
    start_beat = note_locations[start_index].beat
    num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1


    pred_beat_tempo = torch.zeros([num_beats, NUM_TEMPO_PARAM]).to(DEVICE)
    true_beat_tempo = torch.zeros([num_beats, NUM_TEMPO_PARAM]).to(DEVICE)
    for i in range(num_notes):
        current_beat = note_locations[i+start_index].beat
        if current_beat > previous_beat:
            previous_beat = current_beat
            pred_beat_tempo[current_beat-start_beat] = pred_x[0,i,QPM_INDEX:QPM_INDEX + NUM_TEMPO_PARAM]
            true_beat_tempo[current_beat-start_beat] = true_x[0,i,QPM_INDEX:QPM_INDEX + NUM_TEMPO_PARAM]

    tempo_loss = criterion(pred_beat_tempo, true_beat_tempo)
    return tempo_loss


def make_slicing_indexes_by_measure(num_notes, measure_numbers):
    slice_indexes = []
    if num_notes < TIME_STEPS:
        slice_indexes.append((0, num_notes))
    else:
        first_end_measure = measure_numbers[TIME_STEPS]
        last_measure = measure_numbers[-1]
        if first_end_measure < last_measure - 1:
            first_note_after_the_measure = measure_numbers.index(first_end_measure+1)
            slice_indexes.append((0, first_note_after_the_measure))
            second_end_start_measure = measure_numbers[num_notes - TIME_STEPS]
            first_note_of_the_measure = measure_numbers.index(second_end_start_measure)
            slice_indexes.append((first_note_of_the_measure, num_notes))

            if num_notes > TIME_STEPS * 2:
                first_start = random.randrange(int(TIME_STEPS/2), int(TIME_STEPS*1.5))
                start_measure = measure_numbers[first_start]
                end_measure = start_measure

                while end_measure < second_end_start_measure:
                    start_note = measure_numbers.index(start_measure)
                    if start_note+TIME_STEPS < num_notes:
                        end_measure = measure_numbers[start_note+TIME_STEPS]
                    else:
                        break
                    end_note = measure_numbers.index(end_measure-1)
                    slice_indexes.append((start_note, end_note))

                    if end_measure > start_measure + 2:
                        start_measure = end_measure - 2
                    elif end_measure > start_measure + 1:
                        start_measure = end_measure - 1
                    else:
                        start_measure - end_measure
        else:
            slice_indexes.append((0, num_notes))

    return slice_indexes

def sigmoid(x, gain=1):
  return 1 / (1 + math.exp(-gain*x))



### training

if args.sessMode == 'train':
    if not args.trainTrill:
        model_parameters = filter(lambda p: p.requires_grad, MODEL.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    else:
        model_parameters = filter(lambda p: p.requires_grad, trill_model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)


    best_prime_loss = float("inf")
    best_second_loss = float("inf")
    best_trill_loss = float("inf")
    start_epoch = 0


    if args.resumeTraining and not args.trainTrill:
        if os.path.isfile('prime_' + args.modelCode + args.resume):
            print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
            # model_codes = ['prime', 'trill']
            filename = 'prime_' + args.modelCode + args.resume
            checkpoint = torch.load(filename)
            # args.start_epoch = checkpoint['epoch']
            # best_valid_loss = checkpoint['best_valid_loss']
            MODEL.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            start_epoch = checkpoint['epoch'] - 1
            best_prime_loss = checkpoint['best_valid_loss']
            print('Best valid loss was ', best_prime_loss)


    # load data
    print('Loading the training data...')
    training_data_name = args.dataName + ".dat"
    if not os.path.isfile(training_data_name):
        training_data_name = '/mnt/ssd1/jdasam_data/' + training_data_name
    with open(training_data_name, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    # perform_num = len(complete_xy)
    if args.trainingLoss == 'MSE':
        tempo_stats = [MEANS[1][0], STDS[1][0]]
    else:
        tempo_stats = [0,0]

    # train_perf_num = int(perform_num * training_ratio)
    train_xy = complete_xy['train']
    test_xy = complete_xy['valid']
    print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))

    print(train_xy[0][0][0])

    if args.trainTrill:
        train_model = trill_model
    else:
        train_model = MODEL

    # total_step = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        print('current training step is ', NUM_UPDATED)
        tempo_loss_total = []
        vel_loss_total = []
        dev_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_total = []
        for xy_tuple in train_xy:
            train_x = xy_tuple[0]
            train_y = xy_tuple[1]
            if args.trainingLoss == 'CE':
                train_y = categorize_value_to_vector(train_y, BINS)
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            edges = xy_tuple[4]

            data_size = len(train_x)
            graphs = edges_to_matrix(edges, data_size)
            measure_numbers = [x.measure for x in note_locations]
            # graphs = edges_to_sparse_tensor(edges)
            total_batch_num = int(math.ceil(data_size / (TIME_STEPS * batch_size)))

            key_lists = [0]
            key = 0
            for i in range(num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(num_key_augmentation+1):
                key = key_lists[i]
                temp_train_x = key_augmentation(train_x, key)
                slice_indexes = make_slicing_indexes_by_measure(data_size, measure_numbers)
                kld_weight = sigmoid((NUM_UPDATED - 9e4) / 9e3) * 0.02

                for slice_idx in slice_indexes:
                    tempo_loss, vel_loss, dev_loss, pedal_loss, trill_loss, kld = \
                        batch_time_step_run(temp_train_x, train_y, graphs, note_locations, align_matched, slice_idx, model=train_model, kld_weight=kld_weight)
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
                    # print(tempo_loss)
                    tempo_loss_total.append(tempo_loss.item())
                    vel_loss_total.append(vel_loss.item())
                    dev_loss_total.append(dev_loss.item())
                    pedal_loss_total.append(pedal_loss.item())
                    trill_loss_total.append(trill_loss.item())
                    kld_total.append(kld.item())
                    NUM_UPDATED += 1

        print('Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
              .format(epoch + 1, num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
                      np.mean(dev_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))


        ## Validation
        tempo_loss_total =[]
        vel_loss_total =[]
        deviation_loss_total =[]
        trill_loss_total =[]
        pedal_loss_total = []
        for xy_tuple in test_xy:
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            edges = xy_tuple[4]
            graphs = edges_to_matrix(edges, len(test_x))
            if LOSS_TYPE == 'CE':
                test_y = categorize_value_to_vector(test_y, BINS)

            batch_x = torch.Tensor(test_x).view((1, -1, SCORE_INPUT)).to(DEVICE)
            batch_y = torch.Tensor(test_y).view((1, -1, TOTAL_OUTPUT)).to(DEVICE)
            # input_y = torch.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(DEVICE)
            align_matched = torch.Tensor(align_matched).view(1, -1, 1).to(DEVICE)
            outputs = perform_xml(batch_x, batch_y, graphs, note_locations, tempo_stats, valid_y=batch_y)

            # valid_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM:-num_trill_param], batch_y[:,:,NUM_TEMPO_PARAM:-num_trill_param], align_matched)
            tempo_loss = cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
            if LOSS_TYPE =='CE':
                vel_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM:NUM_TEMPO_PARAM+len(BINS[1])], batch_y[:,:,NUM_TEMPO_PARAM:NUM_TEMPO_PARAM+len(BINS[1])], align_matched)
                deviation_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM+len(BINS[1]):NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2])],
                                        batch_y[:,:,NUM_TEMPO_PARAM+len(BINS[1]):NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2])])
                pedal_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2]):-num_trill_param],
                                        batch_y[:,:,NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2]):-num_trill_param])
                trill_loss = criterion(outputs[:,:,-num_trill_param:], batch_y[:,:,-num_trill_param:])
            else:
                vel_loss = criterion(outputs[:, :, 1], batch_y[:, :, 1], align_matched)
                deviation_loss = criterion(outputs[:, :, 2], batch_y[:, :, 2], align_matched)
                pedal_loss = criterion(outputs[:, :, 3:-num_trill_param], batch_y[:, :, 3:-num_trill_param], align_matched)
                trill_loss = criterion(outputs[:, :, -num_trill_param:], batch_y[:, :, -num_trill_param:], align_matched)

            # valid_loss_total.append(valid_loss.item())
            tempo_loss_total.append(tempo_loss.item())
            vel_loss_total.append(vel_loss.item())
            deviation_loss_total.append(deviation_loss.item())
            pedal_loss_total.append(pedal_loss.item())
            trill_loss_total.append(trill_loss.item())

        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss / 4 + mean_pedal_loss * 3) / 5.25

        print("Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss , mean_vel_loss,
                      mean_deviation_loss, mean_pedal_loss, mean_trill_loss))

        is_best = mean_valid_loss < best_prime_loss
        best_prime_loss = min(mean_valid_loss, best_prime_loss)


        is_best_trill = mean_trill_loss < best_trill_loss
        best_trill_loss = min(mean_trill_loss, best_trill_loss)

        if args.trainTrill:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': trill_model.state_dict(),
                'best_valid_loss': best_trill_loss,
                'optimizer': trill_optimizer.state_dict(),
            }, is_best_trill, model_name='trill')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': MODEL.state_dict(),
                'best_valid_loss': best_prime_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, model_name='prime')


    #end of epoch


elif args.sessMode in ['test', 'testAll', 'encode', 'encodeAll']:
### test session
    if os.path.isfile('prime_' + args.modelCode + args.resume):
        print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
        # model_codes = ['prime', 'trill']
        filename = 'prime_' + args.modelCode + args.resume
        checkpoint = torch.load(filename)
        # args.start_epoch = checkpoint['epoch']
        # best_valid_loss = checkpoint['best_valid_loss']
        MODEL.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))

        trill_filename = 'trill_' + args.trillCode + args.resume
        checkpoint = torch.load(trill_filename, DEVICE)
        trill_model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(trill_filename, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    if args.sessMode == 'test':
        load_file_and_generate_performance(args.testPath)
    elif args.sessMode=='testAll':
        MODEL.sequence_iteration = 10
        path_list = cons.emotion_data_path
        emotion_list = cons.emotion_key_list
        perform_z_by_list = encode_all_emotionNet_data(path_list, emotion_list)
        test_list = cons.test_piece_list
        for piece in test_list:
            path = '/mnt/ssd1/yoojin_projects/performNN/test_pieces/' + piece[0] + '/'
            composer = piece[1]
            for perform_z_pair in perform_z_by_list:
                load_file_and_generate_performance(path, composer, z=perform_z_pair)
            load_file_and_generate_performance(path, composer, z=0)

    elif args.sessMode == 'encode':
        perform_z, qpm_primo = load_file_and_encode_style(args.testPath, args.perfName, args.composer)
        print(perform_z)
        with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
            pickle.dump(perform_z, f, protocol=2)


# elif args.sessMode=='plot':
#     if os.path.isfile(args.resume):
#         print("=> loading checkpoint '{}'".format(args.resume))
#         checkpoint = torch.load(args.resume)
#         # args.start_epoch = checkpoint['epoch']
#         best_valid_loss = checkpoint['best_valid_loss']
#         MODEL.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint '{}' (epoch {})"
#               .format(args.resume, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))
#
#
#     with open(args.dataName + ".dat", "rb") as f:
#         u = pickle._Unpickler(f)
#         u.encoding = 'latin1'
#         # p = u.load()
#         # complete_xy = pickle.load(f)
#         complete_xy = u.load()
#
#     with open(args.dataName + "_stat.dat", "rb") as f:
#         u = pickle._Unpickler(f)
#         u.encoding = 'latin1'
#         MEANS, STDS = u.load()
#
#     perform_num = len(complete_xy)
#     tempo_stats = [MEANS[1][0], STDS[1][0]]
#
#     train_perf_num = int(perform_num * training_ratio)
#     train_xy = complete_xy[:train_perf_num]
#     test_xy = complete_xy[train_perf_num:]
#
#     n_tuple = 0
#     for xy_tuple in test_xy:
#         n_tuple += 1
#         train_x = xy_tuple[0]
#         train_y = xy_tuple[1]
#         prev_feature = xy_tuple[2]
#         note_locations = xy_tuple[3]
#
#         data_size = len(train_x)
#         total_batch_num = int(math.ceil(data_size / (TIME_STEPS * batch_size)))
#         batch_size=1
#         for step in range(total_batch_num - 1):
#             batch_start = step * batch_size * TIME_STEPS
#             batch_end = (step + 1) * batch_size * TIME_STEPS
#             batch_x = Variable(
#                 torch.Tensor(train_x[batch_start:batch_end]))
#             batch_y = train_y[batch_start:batch_end]
#             # print(batch_x.shape, batch_y.shape)
#             # input_y = Variable(
#             #     torch.Tensor(prev_feature[step * batch_size * time_steps:(step + 1) * batch_size * time_steps]))
#             # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
#             batch_x = batch_x.view((batch_size, TIME_STEPS, SCORE_INPUT)).to(DEVICE)
#             # is_beat_batch = is_beat_list[batch_start:batch_end]
#             # batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
#             # input_y = input_y.view((batch_size, time_steps, num_output)).to(device)
#
#             # hidden = model.init_hidden(1)
#             # final_hidden = model.init_final_layer(1)
#             # outputs, hidden, final_hidden = model(batch_x, input_y, hidden, final_hidden)
#             #
#             if args.trainTrill:
#                 input_y = torch.zeros(1, 1, TOTAL_OUTPUT)
#             else:
#                 input_y = torch.zeros(1, 1, TOTAL_OUTPUT - num_trill_param)
#
#             input_y[0] = batch_y[0][0]
#             input_y = input_y.view((1, 1, TOTAL_OUTPUT)).to(DEVICE)
#             outputs = perform_xml(batch_x, input_y, note_locations, tempo_stats)
#             outputs = torch.Tensor(outputs).view((1, -1, TOTAL_OUTPUT))
#
#             outputs = outputs.cpu().detach().numpy()
#             # batch_y = batch_y.cpu().detach().numpy()
#             batch_y = np.asarray(batch_y).reshape((1, -1, TOTAL_OUTPUT))
#             plt.figure(figsize=(10, 7))
#             for i in range(4):
#                 plt.subplot(411+i)
#                 plt.plot(batch_y[0, :, i])
#                 plt.plot(outputs[0, :, i])
#             plt.savefig('images/piece{:d},seg{:d}.png'.format(n_tuple, step))
#             plt.close()