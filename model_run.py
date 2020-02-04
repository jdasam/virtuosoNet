import th
import pickle
import _pickle as cPickle
import argparse
import math
import numpy as np
import shutil
import os
import matplotlib
import warnings
matplotlib.use('Agg')

from .parser import get_parser
import pyScoreParser.xml_matching as xml_matching
import pyScoreParser.performanceWorm as perf_worm
import data_process as dp
import copy
import random
from . import nnModel
from . import model_parameters as param
import model_constants as cons
import sys
import style_analysis

sys.modules['xml_matching'] = xml_matching


# TODO: move xml_matching.
def should_be_moved():
    raise(NotImplementedError)

xml_matching.read_score_perform_pair = should_be_moved

#>>>>>>>>>>>>>>>> to parser
LOSS_TYPE = args.trainingLoss
HIERARCHY = False
IN_HIER = False
HIER_MEAS = False
HIER_BEAT = False
HIER_MODEL = None
RAND_TRAIN = args.randomTrain
TRILL = False

if 'measure' in args.modelCode or 'beat' in args.modelCode:
HIERARCHY = True
elif 'note' in args.modelCode:
IN_HIER = True  # In hierarchy mode
if HIERARCHY or IN_HIER:
if 'measure' in args.modelCode or 'measure' in args.hierCode:
    HIER_MEAS = True
elif 'beat' in args.modelCode or 'beat' in args.hierCode:
    HIER_BEAT = True

if 'trill' in args.modelCode:
TRILL = True


### parameters
learning_rate = 0.0003
TIME_STEPS = 200
VALID_STEPS = 10000
DELTA_WEIGHT = 1
NUM_UPDATED = 0
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5
KLD_MAX = 0.02
KLD_SIG = 15e4

batch_size = 1
#<<<<<<<<<<<<<<<< to parser


#>>>>>>>>>>>>>>>> to parser.get_name

print('Learning Rate: {}, Time_steps: {}, Delta weight: {}, Weight decay: {}, Grad clip: {}, KLD max: {}, KLD sig step: {}'.format
    (learning_rate, TIME_STEPS, DELTA_WEIGHT, WEIGHT_DECAY, GRAD_CLIP, KLD_MAX, KLD_SIG))
num_epochs = 100
num_key_augmentation = 1
#<<<<<<<<<<<<<<<< to parser.get_name

#>>>>>>>>>>>>>>>> remove
NUM_INPUT = 78
NUM_PRIME_PARAM = 11
if HIERARCHY:
NUM_OUTPUT = 2
elif TRILL:
NUM_INPUT += NUM_PRIME_PARAM
NUM_OUTPUT = 5
else:
NUM_OUTPUT = 11
if IN_HIER:
NUM_INPUT += 2
#<<<<<<<<<<<<<<<< remove

#>>>>>>>>>>>>>>>> remove or constants
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

QPM_INDEX = 0
# VOICE_IDX = 11
TEMPO_IDX = 26
QPM_PRIMO_IDX = 4
TEMPO_PRIMO_IDX = -2
GRAPH_KEYS = ['onset', 'forward', 'melisma', 'rest']
if args.slurEdge:
GRAPH_KEYS.append('slur')
if args.voiceEdge:
GRAPH_KEYS.append('voice')
N_EDGE_TYPE = len(GRAPH_KEYS) * 2
# mean_vel_start_index = 7
# vel_vec_start_index = 33
#<<<<<<<<<<<<<<<< remove or constants

#>>>>>>>>>>>>>>>> dataset
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
#<<<<<<<<<<<<<<<<< dataset

def main():
th.cuda.set_device(args.device)
DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

if args.sessMode == 'train' and not args.resumeTraining:
NET_PARAM = param.initialize_model_parameters_by_code(args.modelCode)
NET_PARAM.num_edge_types = N_EDGE_TYPE
NET_PARAM.training_args = args
param.save_parameters(NET_PARAM, args.modelCode + '_param')
elif args.resumeTraining:
NET_PARAM = param.load_parameters(args.modelCode + '_param')
else:
NET_PARAM = param.load_parameters(args.modelCode + '_param')
TrillNET_Param = param.load_parameters(args.trillCode + '_param')
# if not hasattr(NET_PARAM, 'num_edge_types'):
#     NET_PARAM.num_edge_types = 10
# if not hasattr(TrillNET_Param, 'num_edge_types'):
#     TrillNET_Param.num_edge_types = 10
TRILL_MODEL = nnModel.TrillRNN(TrillNET_Param, DEVICE).to(DEVICE)


if 'isgn' in args.modelCode:
MODEL = nnModel.ISGN(NET_PARAM, DEVICE).to(DEVICE)
elif 'han' in args.modelCode:
if 'ar' in args.modelCode:
    step_by_step = True
else:
    step_by_step = False
MODEL = nnModel.HAN_Integrated(NET_PARAM, DEVICE, step_by_step).to(DEVICE)
elif 'trill' in args.modelCode:
MODEL = nnModel.TrillRNN(NET_PARAM, DEVICE).to(DEVICE)
else:
print('Error: Unclassified model code')
# Model = nnModel.HAN_VAE(NET_PARAM, DEVICE, False).to(DEVICE)

optimizer = th.optim.Adam(MODEL.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)


if LOSS_TYPE == 'MSE':
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
        return th.zeros(1).to(DEVICE)
    return th.sum(((target - pred) ** 2) * aligned_status) / data_size
elif LOSS_TYPE == 'CE':
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


def save_checkpoint(state, is_best, filename=args.modelCode, model_name='prime'):
warnings.warn('moved to utils.py', DeprecationWarning)
save_name = model_name + '_' + filename + '_checkpoint.pth.tar'
th.save(state, save_name)
if is_best:
    best_name = model_name + '_' + filename + '_best.pth.tar'
    shutil.copyfile(save_name, best_name)



def edges_to_matrix(edges, num_notes):
warnings.warn('moved to graph.py', DeprecationWarning)
if not MODEL.is_graph:
    return None
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
matrix = th.Tensor(matrix)
return matrix


def edges_to_matrix_short(edges, slice_index):
warnings.warn('moved to graph.py', DeprecationWarning)
if not MODEL.is_graph:
    return None
num_keywords = len(GRAPH_KEYS)
# TODO: No ref for slice_idx
num_notes = slice_idx[1] - slice_idx[0]

matrix = np.zeros((N_EDGE_TYPE, num_notes, num_notes))
start_edge_index = xml_matching.binary_index_for_edge(edges, slice_index[0])
end_edge_index = xml_matching.binary_index_for_edge(edges, slice_index[1] + 1)
for i in range(start_edge_index, end_edge_index):
    edg = edges[i]
    if edg[2] not in GRAPH_KEYS:
        continue
    if edg[1] >= slice_index[1]:
        continue
    edge_type = GRAPH_KEYS.index(edg[2])
    matrix[edge_type, edg[0]-slice_index[0], edg[1]-slice_index[0]] = 1
    if edge_type != 0:
        matrix[edge_type+num_keywords, edg[1]-slice_index[0], edg[0]-slice_index[0]] = 1
    else:
        matrix[edge_type, edg[1]-slice_index[0], edg[0]-slice_index[0]] = 1
matrix[num_keywords, :,:] = np.identity(num_notes)
matrix = th.Tensor(matrix)

return matrix


def edges_to_sparse_tensor(edges):
warnings.warn('moved to graph.py', DeprecationWarning)
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

    edge_list = th.LongTensor(edge_list)
edge_type_list = th.FloatTensor(edge_type_list)

matrix = th.sparse.FloatTensor(edge_list.t(), edge_type_list)

return matrix

#>>>>>>>>>>>>>> to dataset.py probably remove
def categorize_value_to_vector(y, bins):
warnings.warn('deprecated', DeprecationWarning)
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
#<<<<<<<<<<<< probably remove

def scale_model_prediction_to_original(prediction, MEANS, STDS):
warnings.warn('moved to inference.py', DeprecationWarning)
for i in range(len(STDS)):
    for j in range(len(STDS[i])):
        if STDS[i][j] < 1e-4:
            STDS[i][j] = 1
prediction = np.squeeze(np.asarray(prediction.cpu()))
num_notes = len(prediction)
if LOSS_TYPE == 'MSE':
    for i in range(11):
        prediction[:, i] *= STDS[1][i]
        prediction[:, i] += MEANS[1][i]
    for i in range(11, 15):
        prediction[:, i] *= STDS[1][i+4]
        prediction[:, i] += MEANS[1][i+4]
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

return prediction

def load_file_and_generate_performance(path_name, composer=args.composer, z=args.latent, start_tempo=args.startTempo, return_features=False):
warnings.warn('moved to inference.py', DeprecationWarning)
vel_pair = (int(args.velocity.split(',')[0]), int(args.velocity.split(',')[1]))
test_x, xml_notes, xml_doc, edges, note_locations = xml_matching.read_xml_to_array(path_name, MEANS, STDS,
                                                                                    start_tempo, composer,
                                                                                    vel_pair)
batch_x = th.Tensor(test_x)
num_notes = len(test_x)
input_y = th.zeros(1, num_notes, NUM_OUTPUT).to(DEVICE)

if type(z) is dict:
    initial_z = z['z']
    qpm_change = z['qpm']
    z = z['key']
    batch_x[:,QPM_PRIMO_IDX] = batch_x[:,QPM_PRIMO_IDX] + qpm_change
else:
    initial_z = 'zero'

if IN_HIER:
    batch_x = batch_x.to(DEVICE).view(1, -1, HIER_MODEL.input_size)
    graph = edges_to_matrix(edges, batch_x.shape[1])
    MODEL.is_teacher_force = False
    if type(initial_z) is list:
        hier_z = initial_z[0]
        final_z = initial_z[1]
    else:
        # hier_z = [z] * HIER_MODEL_PARAM.encoder.size
        hier_z = 'zero'
        final_z = initial_z
    hier_input_y = th.zeros(1, num_notes, HIER_MODEL.output_size)
    hier_output, _ = run_model_in_steps(batch_x, hier_input_y, graph, note_locations, initial_z=hier_z, model=HIER_MODEL)
    if 'measure' in args.hierCode:
        hierarchy_numbers = [x.measure for x in note_locations]
    else:
        hierarchy_numbers = [x.section for x in note_locations]
    hier_output_spanned = HIER_MODEL.span_beat_to_note_num(hier_output, hierarchy_numbers, len(test_x), 0)
    combined_x = th.cat((batch_x, hier_output_spanned), 2)
    prediction, _ = run_model_in_steps(combined_x, input_y, graph, note_locations, initial_z=final_z, model=MODEL)
else:
    if type(initial_z) is list:
        initial_z = initial_z[0]
    batch_x = batch_x.to(DEVICE).view(1, -1, NUM_INPUT)
    graph = edges_to_matrix(edges, batch_x.shape[1])
    prediction, _ = run_model_in_steps(batch_x, input_y, graph, note_locations, initial_z=initial_z, model=MODEL)


trill_batch_x = th.cat((batch_x, prediction), 2)
trill_prediction, _ = run_model_in_steps(trill_batch_x, th.zeros(1, num_notes, cons.num_trill_param), graph, note_locations, model=TRILL_MODEL)

prediction = th.cat((prediction, trill_prediction), 2)
prediction = scale_model_prediction_to_original(prediction, MEANS, STDS)

output_features = xml_matching.model_prediction_to_feature(prediction)
output_features = xml_matching.add_note_location_to_features(output_features, note_locations)
if return_features:
    return output_features

output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                        predicted=True)
output_midi, midi_pedals = xml_matching.xml_notes_to_midi(output_xml)
piece_name = path_name.split('/')
save_name = 'test_result/' + piece_name[-2] + '_by_' + args.modelCode + '_z' + str(z)

perf_worm.plot_performance_worm(output_features, save_name + '.png')
xml_matching.save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_name + '.mid',
                                            bool_pedal=args.boolPedal, disklavier=args.disklavier)


def load_file_and_encode_style(path, perf_name, composer_name):
warnings.warn('moved to dataset.py', DeprecationWarning)
test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(path, perf_name, composer_name, MEANS, STDS)
qpm_primo = test_x[0][4]

test_x, test_y = handle_data_in_tensor(test_x, test_y, hierarchy_test=IN_HIER)
edges = edges_to_matrix(edges, test_x.shape[0])

if IN_HIER:
    test_x = test_x.view((1, -1, HIER_MODEL.input_size))
    hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
    perform_z_high = encode_performance_style_vector(test_x, hier_y, edges, note_locations, model=HIER_MODEL)
    hier_outputs, _ = run_model_in_steps(test_x, hier_y, edges, note_locations, model=HIER_MODEL)
    if HIER_MEAS:
        hierarchy_numbers = [x.measure for x in note_locations]
    elif HIER_BEAT:
        hierarchy_numbers = [x.beat for x in note_locations]
    hier_outputs_spanned = HIER_MODEL.span_beat_to_note_num(hier_outputs, hierarchy_numbers, test_x.shape[1], 0)
    input_concat = th.cat((test_x, hier_outputs_spanned), 2)
    batch_y = test_y[1].view(1, -1, MODEL.output_size)
    perform_z_note = encode_performance_style_vector(input_concat, batch_y, edges, note_locations, model=MODEL)
    perform_z = [perform_z_high, perform_z_note]

else:
    batch_x = test_x.view((1, -1, NUM_INPUT))
    batch_y = test_y.view((1, -1, NUM_OUTPUT))
    perform_z = encode_performance_style_vector(batch_x, batch_y, edges, note_locations)
    perform_z = [perform_z]

return perform_z, qpm_primo


def load_all_file_and_encode_style(parsed_data, measure_only=False, emotion_data=False):
warnings.warn('moved to dataset.py', DeprecationWarning)
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
            test_x, test_y = handle_data_in_tensor(piece_test_x[j], piece_test_y[j], hierarchy_test=IN_HIER)
            edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
            test_x = test_x.view((1, -1, HIER_MODEL.input_size))
            hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
            perform_z_high = encode_performance_style_vector(test_x, hier_y, edges, piece_note_locations[j], model=HIER_MODEL)
        else:
            test_x, test_y = handle_data_in_tensor(piece_test_x[j], piece_test_y[j], hierarchy_test=False)
            edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
            test_x = test_x.view((1, -1, MODEL.input_size))
            test_y = test_y.view(1, -1, MODEL.output_size)
            perform_z_high = encode_performance_style_vector(test_x, test_y, edges, piece_note_locations[j],
                                                                model=MODEL)
        # perform_z_high = perform_z_high.reshape(-1).cpu().numpy()
        # piece_z.append(perform_z_high)
        # perf_name_list.append(piece_perf_name[j])

        perform_z_high = [z.reshape(-1).cpu().numpy() for z in perform_z_high]
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

def encode_performance_style_vector(input, input_y, edges, note_locations, model=MODEL):
warnings.warn('moved to utils.py', DeprecationWarning)
with th.no_grad():
    model_eval = model.eval()
    if edges is not None:
        edges = edges.to(DEVICE)
    encoded_z = model_eval(input, input_y, edges,
                            note_locations=note_locations, start_index=0, return_z=True)
return encoded_z


def encode_all_emotionNet_data(path_list, style_keywords):
warnings.warn('moved to dataset.py', DeprecationWarning)
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
            perform_z_li, qpm_primo = load_file_and_encode_style(path, perf_name, composer_name)
            perform_z_li = [th.mean(th.stack(z), 0, True) for z in perform_z_li]
            indiv_perform_z.append(perform_z_li)
            indiv_qpm.append(qpm_primo)
        for i in range(1, num_style):
            for j in range(num_model):
                indiv_perform_z[i][j] = indiv_perform_z[i][j] - indiv_perform_z[0][j]
            indiv_qpm[i] = indiv_qpm[i] - indiv_qpm[0]
        perform_z_list_by_subject.append(indiv_perform_z)
        qpm_list_by_subject.append(indiv_qpm)
for i in range(num_style):
    z_by_models = []
    for j in range(num_model):
        emotion_mean_z = []
        for z_list in perform_z_list_by_subject:
            emotion_mean_z.append(z_list[i][j])
        mean_perform_z = th.mean(th.stack(emotion_mean_z), 0, True)
        z_by_models.append(mean_perform_z)
    if i is not 0:
        emotion_qpm = []
        for qpm_change in qpm_list_by_subject:
            emotion_qpm.append(qpm_change[i])
        mean_qpm_change = np.mean(emotion_qpm)
    else:
        mean_qpm_change = 0
    print(style_keywords[i], z_by_models, mean_qpm_change)
    perform_z_by_emotion.append({'z': z_by_models, 'key': style_keywords[i], 'qpm': mean_qpm_change})

return perform_z_by_emotion
    # with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
    #     pickle.dump(mean_perform_z, f, protocol=2)


def run_model_in_steps(input, input_y, edges, note_locations, initial_z=False, model=MODEL):
warnings.warn('moved to utils.py', DeprecationWarning)
num_notes = input.shape[1]
with th.no_grad():  # no need to track history in validation
    model_eval = model.eval()
    total_output = []
    total_z = []
    measure_numbers = [x.measure for x in note_locations]
    slice_indexes = dp.make_slicing_indexes_by_measure(num_notes, measure_numbers, steps=VALID_STEPS, overlap=False)
        if edges is not None:
            edges = edges.to(DEVICE)

        for slice_idx in slice_indexes:
            batch_start, batch_end = slice_idx
            if edges is not None:
                batch_graph = edges[:, batch_start:batch_end, batch_start:batch_end]
            else:
                batch_graph = None
            batch_input = input[:, batch_start:batch_end, :].view(1,-1,model.input_size)
            batch_input_y = input_y[:, batch_start:batch_end, :].view(1,-1,model.output_size)
            temp_outputs, perf_mu, perf_var, _ = model_eval(batch_input, batch_input_y, batch_graph,
                                                                           note_locations=note_locations, start_index=batch_start, initial_z=initial_z)
            total_z.append((perf_mu, perf_var))
            total_output.append(temp_outputs)

        outputs = th.cat(total_output, 1)
        return outputs, total_z


def batch_time_step_run(data, model, batch_size=batch_size):
    warnings.warn('moved to utils.py', DeprecationWarning)
    batch_start, batch_end = training_data['slice_idx']
    batch_x, batch_y = handle_data_in_tensor(data['x'][batch_start:batch_end], data['y'][batch_start:batch_end])

    batch_x = batch_x.view((batch_size, -1, NUM_INPUT))
    batch_y = batch_y.view((batch_size, -1, NUM_OUTPUT))

    align_matched = th.Tensor(data['align_matched'][batch_start:batch_end]).view((batch_size, -1, 1)).to(DEVICE)
    pedal_status = th.Tensor(data['pedal_status'][batch_start:batch_end]).view((batch_size, -1, 1)).to(DEVICE)

    if training_data['graphs'] is not None:
        edges = training_data['graphs']
        if edges.shape[1] == batch_end - batch_start:
            edges = edges.to(DEVICE)
        else:
            edges = edges[:, batch_start:batch_end, batch_start:batch_end].to(DEVICE)
    else:
        edges = training_data['graphs']

    prime_batch_x = batch_x
    if HIERARCHY:
        prime_batch_y = batch_y
    else:
        prime_batch_y = batch_y[:, :, 0:NUM_PRIME_PARAM]

    model_train = model.train()
    outputs, perform_mu, perform_var, total_out_list \
        = model_train(prime_batch_x, prime_batch_y, edges, note_locations, batch_start)

    if HIERARCHY:
        if HIER_MEAS:
            hierarchy_numbers = [x.measure for x in note_locations]
        elif HIER_BEAT:
            hierarchy_numbers = [x.beat for x in note_locations]
        tempo_in_hierarchy = MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, batch_start, 0)
        dynamics_in_hierarchy = MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, batch_start, 1)
        tempo_loss = criterion(outputs[:,:,0:1], tempo_in_hierarchy)
        vel_loss = criterion(outputs[:,:,1:2], dynamics_in_hierarchy)
        if args.deltaLoss and outputs.shape[1] > 1:
            vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
            vel_true_delta = dynamics_in_hierarchy[:, 1:, :] - dynamics_in_hierarchy[:, :-1, :]

            vel_loss += criterion(vel_out_delta, vel_true_delta) * DELTA_WEIGHT
            vel_loss /= 1 + DELTA_WEIGHT

        total_loss = tempo_loss + vel_loss
    elif TRILL:
        trill_bool = batch_x[:, :, is_trill_index_concated:is_trill_index_concated + 1]
        if th.sum(trill_bool) > 0:
            total_loss = criterion(outputs, batch_y, trill_bool)
        else:
            return th.zeros(1), th.zeros(1), th.zeros(1),  th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1)

    else:
        if 'isgn' in args.modelCode and args.intermediateLoss:
            total_loss = th.zeros(1).to(DEVICE)
            for out in total_out_list:
                if model.is_baseline:
                    tempo_loss = criterion(out[:, :, 0:1],
                                           prime_batch_y[:, :, 0:1], align_matched)
                else:
                    tempo_loss = cal_tempo_loss_in_beat(out, prime_batch_y, note_locations, batch_start)
                vel_loss = criterion(out[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX],
                                     prime_batch_y[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX], align_matched)
                dev_loss = criterion(out[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX],
                                     prime_batch_y[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX], align_matched)
                articul_loss = criterion(out[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX+1],
                                         prime_batch_y[:,:,PEDAL_PARAM_IDX:PEDAL_PARAM_IDX+1], pedal_status)
                pedal_loss = criterion(out[:, :, PEDAL_PARAM_IDX+1:], prime_batch_y[:, :, PEDAL_PARAM_IDX+1:],
                                       align_matched)

                total_loss += (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11
            total_loss /= len(total_out_list)
        else:
            if model.is_baseline:
                tempo_loss = criterion(outputs[:, :, 0:1],
                                     prime_batch_y[:, :, 0:1], align_matched)
            else:
                tempo_loss = cal_tempo_loss_in_beat(outputs, prime_batch_y, note_locations, batch_start)
            vel_loss = criterion(outputs[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX],
                                 prime_batch_y[:, :, VEL_PARAM_IDX:DEV_PARAM_IDX], align_matched)
            dev_loss = criterion(outputs[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX],
                                 prime_batch_y[:, :, DEV_PARAM_IDX:PEDAL_PARAM_IDX], align_matched)
            articul_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX + 1],
                                     prime_batch_y[:, :, PEDAL_PARAM_IDX:PEDAL_PARAM_IDX + 1], pedal_status)
            pedal_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX + 1:], prime_batch_y[:, :, PEDAL_PARAM_IDX + 1:],
                                   align_matched)
            total_loss = (tempo_loss + vel_loss + dev_loss + articul_loss + pedal_loss * 7) / 11

    if isinstance(perform_mu, bool):
        perform_kld = th.zeros(1)
    else:
        perform_kld = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
        total_loss += perform_kld * kld_weight
    optimizer.zero_grad()
    total_loss.backward()
    th.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if HIERARCHY:
        return tempo_loss, vel_loss, th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), perform_kld
    elif TRILL:
        return th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), th.zeros(1), total_loss, th.zeros(1)
    else:
        return tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, th.zeros(1), perform_kld

    # loss = criterion(outputs, batch_y)
    # tempo_loss = criterion(prime_outputs[:, :, 0], prime_batch_y[:, :, 0])

#>>>>>>>>>>> utils.py not decided
def cal_tempo_loss_in_beat(pred_x, true_x, note_locations, start_index):
    previous_beat = -1

    num_notes = pred_x.shape[1]
    start_beat = note_locations[start_index].beat
    num_beats = note_locations[num_notes+start_index-1].beat - start_beat + 1

    pred_beat_tempo = th.zeros([num_beats, NUM_TEMPO_PARAM]).to(DEVICE)
    true_beat_tempo = th.zeros([num_beats, NUM_TEMPO_PARAM]).to(DEVICE)
    for i in range(num_notes):
        current_beat = note_locations[i+start_index].beat
        if current_beat > previous_beat:
            previous_beat = current_beat
            if 'baseline' in args.modelCode:
                for j in range(i, num_notes):
                    if note_locations[j+start_index].beat > current_beat:
                        break
                if not i == j:
                    pred_beat_tempo[current_beat - start_beat] = th.mean(pred_x[0, i:j, QPM_INDEX])
                    true_beat_tempo[current_beat - start_beat] = th.mean(true_x[0, i:j, QPM_INDEX])
            else:
                pred_beat_tempo[current_beat-start_beat] = pred_x[0,i,QPM_INDEX:QPM_INDEX + NUM_TEMPO_PARAM]
                true_beat_tempo[current_beat-start_beat] = true_x[0,i,QPM_INDEX:QPM_INDEX + NUM_TEMPO_PARAM]

    tempo_loss = criterion(pred_beat_tempo, true_beat_tempo)
    if args.deltaLoss and pred_beat_tempo.shape[0] > 1:
        prediction_delta = pred_beat_tempo[1:] - pred_beat_tempo[:-1]
        true_delta = true_beat_tempo[1:] - true_beat_tempo[:-1]
        delta_loss = criterion(prediction_delta, true_delta)

        tempo_loss = (tempo_loss + delta_loss * DELTA_WEIGHT) / (1 + DELTA_WEIGHT)

    return tempo_loss


def handle_data_in_tensor(x, y, hierarchy_test=False):
    x = th.Tensor(x)
    y = th.Tensor(y)
    if HIER_MEAS:
        hierarchy_output = y[:,cons.MEAS_TEMPO_IDX:cons.MEAS_TEMPO_IDX+2]
    elif HIER_BEAT:
        hierarchy_output = y[:,cons.BEAT_TEMPO_IDX:cons.BEAT_TEMPO_IDX+2]

    if hierarchy_test:
        y = y[:, :NUM_PRIME_PARAM]
        return x.to(DEVICE), (hierarchy_output.to(DEVICE), y.to(DEVICE))

    if HIERARCHY:
        y = hierarchy_output
    elif IN_HIER:
        x = th.cat((x,hierarchy_output), 1)
        y = y[:, :NUM_PRIME_PARAM]
    elif TRILL:
        x = th.cat((x, y[:, :NUM_PRIME_PARAM]), 1)
        y = y[: ,-num_trill_param:]
    else:
        y = y[:, :NUM_PRIME_PARAM]

    return x.to(DEVICE), y.to(DEVICE)


def sigmoid(x, gain=1):
  return 1 / (1 + math.exp(-gain*x))


class TraningSample():
def __init__(self, index):
    self.index = index
    self.slice_indexes = None
#<<<<<<<<<<< not decided

### training

if args.sessMode == 'train':
model_parameters = filter(lambda p: p.requires_grad, MODEL.parameters())
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
        checkpoint = th.load(filename,  map_location=DEVICE)
        best_valid_loss = checkpoint['best_valid_loss']
        MODEL.load_state_dict(checkpoint['state_dict'])
        MODEL.device = DEVICE
        optimizer.load_state_dict(checkpoint['optimizer'])
        NUM_UPDATED = checkpoint['training_step']
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
    # u = pickle._Unpickler(f)
    # u.encoding = 'latin1'
    u = cPickle.Unpickler(f)
    # p = u.load()
    # complete_xy = pickle.load(f)
    complete_xy = u.load()

train_xy = complete_xy['train']
test_xy = complete_xy['valid']
print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))
print('training sample example', train_xy[0][0][0])

train_model = MODEL

# total_step = len(train_loader)
for epoch in range(start_epoch, num_epochs):
    print('current training step is ', NUM_UPDATED)
    tempo_loss_total = []
    vel_loss_total = []
    dev_loss_total = []
    articul_loss_total = []
    pedal_loss_total = []
    trill_loss_total = []
    kld_total = []

    if RAND_TRAIN:
        num_perf_data = len(train_xy)
        remaining_samples = []
        for i in range(num_perf_data):
            remaining_samples.append(TraningSample(i))
            while len(remaining_samples) > 0:
                new_index = random.randrange(0, len(remaining_samples))
                selected_sample = remaining_samples[new_index]
                train_x = train_xy[selected_sample.index][0]
                train_y = train_xy[selected_sample.index][1]
                if args.trainingLoss == 'CE':
                    train_y = categorize_value_to_vector(train_y, BINS)
                note_locations = train_xy[selected_sample.index][2]
                align_matched = train_xy[selected_sample.index][3]
                pedal_status = train_xy[selected_sample.index][4]
                edges = train_xy[selected_sample.index][5]
                data_size = len(train_x)

                if selected_sample.slice_indexes is None:
                    measure_numbers = [x.measure for x in note_locations]
                    if HIER_MEAS and HIERARCHY:
                        selected_sample.slice_indexes = dp.make_slice_with_same_measure_number(data_size,
                                                                                                     measure_numbers,
                                                                                                         measure_steps=TIME_STEPS)
                    elif HIER_BEAT and HIERARCHY:
                        beat_numbers = [x.beat for x in note_locations]
                        selected_sample.slice_indexes = dp.make_slice_with_same_measure_number(data_size,
                                                                                               beat_numbers,
                                                                                               measure_steps=TIME_STEPS)

                    else:
                        selected_sample.slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=TIME_STEPS)

                num_slice = len(selected_sample.slice_indexes)
                selected_idx = random.randrange(0,num_slice)
                slice_idx = selected_sample.slice_indexes[selected_idx]

                if MODEL.is_graph:
                    graphs = edges_to_matrix_short(edges, slice_idx)
                else:
                    graphs = None

                key_lists = [0]
                key = 0
                for i in range(num_key_augmentation):
                    while key in key_lists:
                        key = random.randrange(-5, 7)
                    key_lists.append(key)

                for i in range(num_key_augmentation+1):
                    key = key_lists[i]
                    temp_train_x = dp.key_augmentation(train_x, key)
                    kld_weight = sigmoid((NUM_UPDATED - KLD_SIG) / (KLD_SIG/10)) * KLD_MAX

                    training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                     'note_locations': note_locations,
                                     'align_matched': align_matched, 'pedal_status': pedal_status,
                                     'slice_idx': slice_idx, 'kld_weight': kld_weight}

                    tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                        batch_time_step_run(training_data, model=train_model)
                    tempo_loss_total.append(tempo_loss.item())
                    vel_loss_total.append(vel_loss.item())
                    dev_loss_total.append(dev_loss.item())
                    articul_loss_total.append(articul_loss.item())
                    pedal_loss_total.append(pedal_loss.item())
                    trill_loss_total.append(trill_loss.item())
                    kld_total.append(kld.item())
                    NUM_UPDATED += 1

                del selected_sample.slice_indexes[selected_idx]
                if len(selected_sample.slice_indexes) == 0:
                    # print('every slice in the sample is trained')
                    del remaining_samples[new_index]

        else:
            for xy_tuple in train_xy:
                train_x = xy_tuple[0]
                train_y = xy_tuple[1]
                if args.trainingLoss == 'CE':
                    train_y = categorize_value_to_vector(train_y, BINS)
                note_locations = xy_tuple[2]
                align_matched = xy_tuple[3]
                pedal_status = xy_tuple[4]
                edges = xy_tuple[5]

                data_size = len(note_locations)
                if MODEL.is_graph:
                    graphs = edges_to_matrix(edges, data_size)
                else:
                    graphs = None
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
                    temp_train_x = dp.key_augmentation(train_x, key)
                    slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=TIME_STEPS)
                    kld_weight = sigmoid((NUM_UPDATED - KLD_SIG) / (KLD_SIG/10)) * KLD_MAX

                    for slice_idx in slice_indexes:
                        training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                         'note_locations': note_locations,
                                         'align_matched': align_matched, 'pedal_status': pedal_status,
                                         'slice_idx': slice_idx, 'kld_weight': kld_weight}

                        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                            batch_time_step_run(training_data, model=train_model)
                        tempo_loss_total.append(tempo_loss.item())
                        vel_loss_total.append(vel_loss.item())
                        dev_loss_total.append(dev_loss.item())
                        articul_loss_total.append(articul_loss.item())
                        pedal_loss_total.append(pedal_loss.item())
                        trill_loss_total.append(trill_loss.item())
                        kld_total.append(kld.item())
                        NUM_UPDATED += 1

        print('Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
              .format(epoch + 1, num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
                      np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))


        ## Validation
        tempo_loss_total =[]
        vel_loss_total =[]
        deviation_loss_total =[]
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_loss_total = []

        for xy_tuple in test_xy:
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]
            edges = xy_tuple[5]
            if MODEL.is_graph:
                graphs = edges_to_matrix(edges, len(test_x))
            else:
                graphs = None
            if LOSS_TYPE == 'CE':
                test_y = categorize_value_to_vector(test_y, BINS)

            batch_x, batch_y = handle_data_in_tensor(test_x, test_y)
            batch_x = batch_x.view(1, -1, NUM_INPUT)
            batch_y = batch_y.view(1, -1, NUM_OUTPUT)
            # input_y = th.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(DEVICE)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(DEVICE)
            pedal_status = th.Tensor(pedal_status).view(1,-1,1).to(DEVICE)
            outputs, total_z = run_model_in_steps(batch_x, batch_y, graphs, note_locations)

            # valid_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM:-num_trill_param], batch_y[:,:,NUM_TEMPO_PARAM:-num_trill_param], align_matched)
            if HIERARCHY:
                if HIER_MEAS:
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif HIER_BEAT:
                    hierarchy_numbers = [x.beat for x in note_locations]
                tempo_y = MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 0)
                vel_y = MODEL.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 1)

                tempo_loss = criterion(outputs[:, :, 0:1], tempo_y)
                vel_loss = criterion(outputs[:, :, 1:2], vel_y)
                if args.deltaLoss:
                    tempo_out_delta = outputs[:, 1:, 0:1] - outputs[:, :-1, 0:1]
                    tempo_true_delta = tempo_y[:, 1:, :] - tempo_y[:, :-1, :]
                    vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
                    vel_true_delta = vel_y[:, 1:, :] - vel_y[:, :-1, :]

                    tempo_loss += criterion(tempo_out_delta, tempo_true_delta) * DELTA_WEIGHT
                    vel_loss += criterion(vel_out_delta, vel_true_delta) * DELTA_WEIGHT

                deviation_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                trill_loss = th.zeros(1)

                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
            elif TRILL:
                trill_bool = batch_x[:,:,is_trill_index_concated] == 1
                trill_bool = trill_bool.float().view(1,-1,1).to(DEVICE)
                trill_loss = criterion(outputs, batch_y, trill_bool)

                tempo_loss = th.zeros(1)
                vel_loss = th.zeros(1)
                deviation_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                kld_loss = th.zeros(1)
                kld_loss_total.append(kld_loss.item())

            else:
                tempo_loss = cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
                if LOSS_TYPE =='CE':
                    vel_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM:NUM_TEMPO_PARAM+len(BINS[1])], batch_y[:,:,NUM_TEMPO_PARAM:NUM_TEMPO_PARAM+len(BINS[1])], align_matched)
                    deviation_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM+len(BINS[1]):NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2])],
                                            batch_y[:,:,NUM_TEMPO_PARAM+len(BINS[1]):NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2])])
                    pedal_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2]):-num_trill_param],
                                            batch_y[:,:,NUM_TEMPO_PARAM+len(BINS[1])+len(BINS[2]):-num_trill_param])
                    trill_loss = criterion(outputs[:,:,-num_trill_param:], batch_y[:,:,-num_trill_param:])
                else:
                    vel_loss = criterion(outputs[:, :, VEL_PARAM_IDX], batch_y[:, :, VEL_PARAM_IDX], align_matched)
                    deviation_loss = criterion(outputs[:, :, DEV_PARAM_IDX], batch_y[:, :, DEV_PARAM_IDX], align_matched)
                    articul_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX], batch_y[:, :, PEDAL_PARAM_IDX], pedal_status)
                    pedal_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX+1:], batch_y[:, :, PEDAL_PARAM_IDX+1:], align_matched)
                    trill_loss = th.zeros(1)
                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())

            # valid_loss_total.append(valid_loss.item())
            tempo_loss_total.append(tempo_loss.item())
            vel_loss_total.append(vel_loss.item())
            deviation_loss_total.append(deviation_loss.item())
            articul_loss_total.append(articul_loss.item())
            pedal_loss_total.append(pedal_loss.item())
            trill_loss_total.append(trill_loss.item())

        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_articul_loss = np.mean(articul_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)
        mean_kld_loss = np.mean(kld_loss_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss + mean_articul_loss + mean_pedal_loss * 7 + mean_kld_loss * kld_weight) / (11 + kld_weight)

        print("Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss , mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss))

        is_best = mean_valid_loss < best_prime_loss
        best_prime_loss = min(mean_valid_loss, best_prime_loss)

        is_best_trill = mean_trill_loss < best_trill_loss
        best_trill_loss = min(mean_trill_loss, best_trill_loss)

        if args.trainTrill:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': MODEL.state_dict(),
                'best_valid_loss': best_trill_loss,
                'optimizer': optimizer.state_dict(),
                'training_step': NUM_UPDATED
            }, is_best_trill, model_name='trill')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': MODEL.state_dict(),
                'best_valid_loss': best_prime_loss,
                'optimizer': optimizer.state_dict(),
                'training_step': NUM_UPDATED
            }, is_best, model_name='prime')


    #end of epoch

else:
# elif args.sessMode in ['test', 'testAll', 'testAllzero', 'encode', 'encodeAll', 'evaluate', 'correlation', 'analysis']:
### test session
    if os.path.isfile('prime_' + args.modelCode + args.resume):
        print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
        # model_codes = ['prime', 'trill']
        filename = 'prime_' + args.modelCode + args.resume
        print('device is ', args.device)
        th.cuda.set_device(args.device)
        if th.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        checkpoint = th.load(filename, map_location=map_location)
        # args.start_epoch = checkpoint['epoch']
        # best_valid_loss = checkpoint['best_valid_loss']
        MODEL.load_state_dict(checkpoint['state_dict'])
        # MODEL.num_graph_iteration = 10
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        # NUM_UPDATED = checkpoint['training_step']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # trill_filename = args.trillCode + args.resume
        trill_filename = args.trillCode + '_best.pth.tar'
        checkpoint = th.load(trill_filename, map_location=map_location)
        TRILL_MODEL.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(trill_filename, checkpoint['epoch']))

        if IN_HIER:
            HIER_MODEL_PARAM = param.load_parameters(args.hierCode + '_param')
            HIER_MODEL = nnModel.HAN_Integrated(HIER_MODEL_PARAM, DEVICE, True).to(DEVICE)
            filename = 'prime_' + args.hierCode + args.resume
            checkpoint = th.load(filename, map_location=DEVICE)
            HIER_MODEL.load_state_dict(checkpoint['state_dict'])
            print("=> high-level model loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    MODEL.is_teacher_force = False

    if args.sessMode == 'test':
        random.seed(0)
        load_file_and_generate_performance(args.testPath)
    elif args.sessMode=='testAll':
        path_list = cons.emotion_data_path
        emotion_list = cons.emotion_key_list
        perform_z_by_list = encode_all_emotionNet_data(path_list, emotion_list)
        test_list = cons.test_piece_list
        for piece in test_list:
            path = './test_pieces/' + piece[0] + '/'
            composer = piece[1]
            if len(piece) == 3:
                start_tempo = piece[2]
            else:
                start_tempo = 0
            for perform_z_pair in perform_z_by_list:
                load_file_and_generate_performance(path, composer, z=perform_z_pair, start_tempo=start_tempo)
            load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)
    elif args.sessMode == 'testAllzero':
        test_list = cons.test_piece_list
        for piece in test_list:
            path = './test_pieces/' + piece[0] + '/'
            composer = piece[1]
            if len(piece) == 3:
                start_tempo = piece[2]
            else:
                start_tempo = 0
            random.seed(0)
            load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)

    elif args.sessMode == 'encode':
        perform_z, qpm_primo = load_file_and_encode_style(args.testPath, args.perfName, args.composer)
        print(perform_z)
        with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
            pickle.dump(perform_z, f, protocol=2)

    elif args.sessMode =='evaluate':
        test_data_name = args.dataName + "_test.dat"
        if not os.path.isfile(test_data_name):
            test_data_name = '/mnt/ssd1/jdasam_data/' + test_data_name
        with open(test_data_name, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            # p = u.load()
            # complete_xy = pickle.load(f)
            complete_xy = u.load()

        tempo_loss_total = []
        vel_loss_total = []
        deviation_loss_total = []
        trill_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        kld_total = []

        prev_perf_x = complete_xy[0][0]
        prev_perfs_worm_data = []
        prev_reconstructed_worm_data = []
        prev_zero_predicted_worm_data = []
        piece_wise_loss = []
        human_correlation_total = []
        human_correlation_results = xml_matching.CorrelationResult()
        model_correlation_total = []
        model_correlation_results = xml_matching.CorrelationResult()
        zero_sample_correlation_total = []
        zero_sample_correlation_results= xml_matching.CorrelationResult()



        for xy_tuple in complete_xy:
            current_perf_index = complete_xy.index(xy_tuple)
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]
            edges = xy_tuple[5]
            graphs = edges_to_matrix(edges, len(test_x))
            if LOSS_TYPE == 'CE':
                test_y = categorize_value_to_vector(test_y, BINS)

            if xml_matching.check_feature_pair_is_from_same_piece(prev_perf_x, test_x):
                piece_changed = False
                # current_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(test_y, note_locations=note_locations, momentum=0.2)
                # for prev_worm in prev_perfs_worm_data:
                #     tempo_r, _ = xml_matching.cal_correlation(current_perf_worm_data[0], prev_worm[0])
                #     dynamic_r, _ = xml_matching.cal_correlation(current_perf_worm_data[1], prev_worm[1])
                #     human_correlation_results.append_result(tempo_r, dynamic_r)
                # prev_perfs_worm_data.append(current_perf_worm_data)
            else:
                piece_changed = True

            if piece_changed or current_perf_index == len(complete_xy)-1:
                prev_perf_x = test_x
                if piece_wise_loss:
                    piece_wise_loss_mean = np.mean(np.asarray(piece_wise_loss), axis=0)
                    tempo_loss_total.append(piece_wise_loss_mean[0])
                    vel_loss_total.append(piece_wise_loss_mean[1])
                    deviation_loss_total.append(piece_wise_loss_mean[2])
                    articul_loss_total.append(piece_wise_loss_mean[3])
                    pedal_loss_total.append(piece_wise_loss_mean[4])
                    trill_loss_total.append(piece_wise_loss_mean[5])
                    kld_total.append(piece_wise_loss_mean[6])
                piece_wise_loss = []

                # human_correlation_total.append(human_correlation_results)
                # human_correlation_results = xml_matching.CorrelationResult()
                #
                # for predict in prev_reconstructed_worm_data:
                #     for human in prev_perfs_worm_data:
                #         tempo_r, _ = xml_matching.cal_correlation(predict[0], human[0])
                #         dynamic_r, _ = xml_matching.cal_correlation(predict[1], human[1])
                #         model_correlation_results.append_result(tempo_r, dynamic_r)
                #
                # model_correlation_total.append(model_correlation_results)
                # model_correlation_results = xml_matching.CorrelationResult()
                #
                # for zero in prev_zero_predicted_worm_data:
                #     for human in prev_perfs_worm_data:
                #         tempo_r, _ = xml_matching.cal_correlation(zero[0], human[0])
                #         dynamic_r, _ = xml_matching.cal_correlation(zero[1], human[1])
                #         zero_sample_correlation_results.append_result(tempo_r, dynamic_r)
                #
                # zero_sample_correlation_total.append(zero_sample_correlation_results)
                # zero_sample_correlation_results = xml_matching.CorrelationResult()
                #
                # prev_reconstructed_worm_data = []
                # prev_zero_predicted_worm_data = []
                # prev_perfs_worm_data = []
                #
                # print('Human Correlation: ', human_correlation_total[-1])
                # print('Reconst Correlation: ', model_correlation_total[-1])
                # print('Zero Sampled Correlation: ', zero_sample_correlation_total[-1])

            batch_x, batch_y = handle_data_in_tensor(test_x, test_y, hierarchy_test=IN_HIER)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(DEVICE)
            pedal_status = th.Tensor(pedal_status).view(1, -1, 1).to(DEVICE)

            if IN_HIER:
                batch_x = batch_x.view((1, -1, HIER_MODEL.input_size))
                hier_y = batch_y[0].view(1, -1, HIER_MODEL.output_size)
                hier_outputs, _ = run_model_in_steps(batch_x, hier_y, graphs, note_locations, model=HIER_MODEL)
                if HIER_MEAS:
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif HIER_BEAT:
                    hierarchy_numbers = [x.beat for x in note_locations]
                hier_outputs_spanned = HIER_MODEL.span_beat_to_note_num(hier_outputs, hierarchy_numbers, batch_x.shape[1], 0)
                input_concat = th.cat((batch_x, hier_outputs_spanned),2)
                batch_y = batch_y[1].view(1,-1, MODEL.output_size)
                outputs, perform_z = run_model_in_steps(input_concat, batch_y, graphs, note_locations, model=MODEL)

                # make another prediction with random sampled z
                zero_hier_outputs, _ = run_model_in_steps(batch_x, hier_y, graphs, note_locations, model=HIER_MODEL,
                                                        initial_z='zero')
                zero_hier_spanned = HIER_MODEL.span_beat_to_note_num(zero_hier_outputs, hierarchy_numbers, batch_x.shape[1], 0)
                zero_input_concat = th.cat((batch_x, zero_hier_spanned),2)
                zero_prediction, _ = run_model_in_steps(zero_input_concat, batch_y, graphs, note_locations, model=MODEL)

            else:
                batch_x = batch_x.view((1, -1, NUM_INPUT))
                batch_y = batch_y.view((1, -1, NUM_OUTPUT))
                outputs, perform_z = run_model_in_steps(batch_x, batch_y, graphs, note_locations)

                # make another prediction with random sampled z
                zero_prediction, _ = run_model_in_steps(batch_x, batch_y, graphs, note_locations, model=MODEL,
                                                     initial_z='zero')

            output_as_feature = outputs.view(-1, NUM_OUTPUT).cpu().numpy()
            predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(output_as_feature, note_locations,
                                                                                momentum=0.2)
            zero_prediction_as_feature = zero_prediction.view(-1, NUM_OUTPUT).cpu().numpy()
            zero_predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(zero_prediction_as_feature, note_locations,
                                                                                     momentum=0.2)

            prev_reconstructed_worm_data.append(predicted_perf_worm_data)
            prev_zero_predicted_worm_data.append(zero_predicted_perf_worm_data)

            # for prev_worm in prev_perfs_worm_data:
            #     tempo_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[0], prev_worm[0])
            #     dynamic_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[1], prev_worm[1])
            #     model_correlation_results.append_result(tempo_r, dynamic_r)
            # print('Model Correlation: ', model_correlation_results)

            # valid_loss = criterion(outputs[:,:,NUM_TEMPO_PARAM:-num_trill_param], batch_y[:,:,NUM_TEMPO_PARAM:-num_trill_param], align_matched)
            if MODEL.is_baseline:
                tempo_loss = criterion(outputs[:, :, 0], batch_y[:, :, 0], align_matched)
            else:
                tempo_loss = cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
            if LOSS_TYPE == 'CE':
                vel_loss = criterion(outputs[:, :, NUM_TEMPO_PARAM:NUM_TEMPO_PARAM + len(BINS[1])],
                                     batch_y[:, :, NUM_TEMPO_PARAM:NUM_TEMPO_PARAM + len(BINS[1])], align_matched)
                deviation_loss = criterion(
                    outputs[:, :, NUM_TEMPO_PARAM + len(BINS[1]):NUM_TEMPO_PARAM + len(BINS[1]) + len(BINS[2])],
                    batch_y[:, :, NUM_TEMPO_PARAM + len(BINS[1]):NUM_TEMPO_PARAM + len(BINS[1]) + len(BINS[2])])
                pedal_loss = criterion(
                    outputs[:, :, NUM_TEMPO_PARAM + len(BINS[1]) + len(BINS[2]):-num_trill_param],
                    batch_y[:, :, NUM_TEMPO_PARAM + len(BINS[1]) + len(BINS[2]):-num_trill_param])
                trill_loss = criterion(outputs[:, :, -num_trill_param:], batch_y[:, :, -num_trill_param:])
            else:
                vel_loss = criterion(outputs[:, :, VEL_PARAM_IDX], batch_y[:, :, VEL_PARAM_IDX], align_matched)
                deviation_loss = criterion(outputs[:, :, DEV_PARAM_IDX], batch_y[:, :, DEV_PARAM_IDX],
                                           align_matched)
                articul_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX], batch_y[:, :, PEDAL_PARAM_IDX],
                                         pedal_status)
                pedal_loss = criterion(outputs[:, :, PEDAL_PARAM_IDX + 1:], batch_y[:, :, PEDAL_PARAM_IDX + 1:],
                                       align_matched)
                trill_loss = th.zeros(1)

            piece_kld = []
            for z in perform_z:
                perform_mu, perform_var = z
                kld = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                piece_kld.append(kld)
            piece_kld = th.mean(th.stack(piece_kld))

            piece_wise_loss.append((tempo_loss.item(), vel_loss.item(), deviation_loss.item(), articul_loss.item(), pedal_loss.item(), trill_loss.item(), piece_kld.item()))



        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_articul_loss = np.mean(articul_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)
        mean_kld_loss = np.mean(kld_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss / 2 + mean_pedal_loss * 8) / 10.5

        print("Test Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss, mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss, mean_kld_loss))
        # num_piece = len(model_correlation_total)
        # for i in range(num_piece):
        #     if len(human_correlation_total) > 0:
        #         print('Human Correlation: ', human_correlation_total[i])
        #         print('Model Correlation: ', model_correlation_total[i])


    elif args.sessMode == 'correlation':
        with open('selected_corr_30.dat', "rb") as f:
            u = pickle._Unpickler(f)
            selected_corr = u.load()
        model_cor = []
        for piece_corr in selected_corr:
            if piece_corr is None or piece_corr==[]:
                continue
            path = piece_corr[0].path_name
            composer_name = copy.copy(path).split('/')[1]
            output_features = load_file_and_generate_performance(path, composer_name, 'zero', return_features=True)
            for slice_corr in piece_corr:
                slc_idx = slice_corr.slice_index
                sliced_features = output_features[slc_idx[0]:slc_idx[1]]
                tempos, dynamics = perf_worm.cal_tempo_and_velocity_by_beat(sliced_features)
                model_correlation_results = xml_matching.CorrelationResult()
                model_correlation_results.path_name = slice_corr.path_name
                model_correlation_results.slice_index = slice_corr.slice_index
                human_tempos = slice_corr.tempo_features
                human_dynamics = slice_corr.dynamic_features
                for i in range(slice_corr.num_performance):
                    tempo_r, _ = xml_matching.cal_correlation(tempos, human_tempos[i])
                    dynamic_r, _ = xml_matching.cal_correlation(dynamics, human_dynamics[i])
                    model_correlation_results._append_result(tempo_r, dynamic_r)
                print(model_correlation_results)
                model_correlation_results.tempo_features = copy.copy(slice_corr.tempo_features)
                model_correlation_results.dynamic_features = copy.copy(slice_corr.dynamic_features)
                model_correlation_results.tempo_features.append(tempos)
                model_correlation_results.dynamic_features.append(dynamics)

                save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_', 10) + '_note{}-{}_by_{}.png'.format(slc_idx[0], slc_idx[1], args.modelCode)
                perf_worm.plot_human_model_features_compare(model_correlation_results.tempo_features, save_name)
                model_cor.append(model_correlation_results)

        with open(args.modelCode + "_cor.dat", "wb") as f:
            pickle.dump(model_cor, f, protocol=2)

    elif args.sessMode == 'analysis':
        # total_perf_z = total_perf_names = []
        # dir_name = 'test_pieces/performScore_test'
        # piece_list = os.listdir(dir_name)
        # for piece in piece_list:
        #     piece_path = os.path.join(dir_name, piece) + '/'
        #     perform_z_list, perf_names = load_all_file_and_encode_style(piece_path, 'Chopin')
        #     total_perf_z.append(perform_z_list)
        #     total_perf_names.append(perf_names)

        # with open("style_parsed_emotion.dat", "rb") as f:
        with open("style_parsed.dat", "rb") as f:
            # u = pickle._Unpickler(f)
            u = cPickle.Unpickler(f)
            total_data = u.load()
        measure_only = True
        is_emotion_data = False

        total_perf_z, total_perf_names = load_all_file_and_encode_style(total_data, measure_only=measure_only, emotion_data=is_emotion_data)
        if ',' in total_perf_names[0]:
            total_perf_names = [x.split(',')[0] for x in total_perf_names]
        else:
            total_perf_names = [x.split('.')[-1] for x in total_perf_names]



        with open("style_z_encoded.dat", "wb") as f:
            pickle.dump({'z': total_perf_z ,'name': total_perf_names}, f, protocol=2)

        # selected_performers = ['Biret', 'Pollini', 'Lisiecki']
        # total_perf_z, total_perf_names = style_analysis.filter_z_by_name(total_perf_z, total_perf_names, selected_performers)
        embedded_z = style_analysis.embedd_tsne(total_perf_z, total_perf_names)
        embedd_pca = style_analysis.embedd_pca(total_perf_z, total_perf_names)
        with open("style_tsne_z.dat", "wb") as f:
            pickle.dump({'z': embedded_z ,'name': total_perf_names}, f, protocol=2)

    elif args.sessMode == 'horowitz':
        # this code is for making Horowitz-like performance from data

        with open("chopin105.dat", "rb") as f:
            # u = pickle._Unpickler(f)
            u = cPickle.Unpickler(f)
            total_data = u.load()
        total_perf_z, total_perf_names = load_all_file_and_encode_style(total_data, measure_only=False, emotion_data=False)
        total_perf_names = [x.split(',')[0] for x in total_perf_names]
        horowitz_z = []
        for z, name in zip(total_perf_z, total_perf_names):
            if name == 'Horowitz':
                horowitz_z.append(z)
        horowitz_z = np.asarray(horowitz_z)
        horowitz_z = np.mean(horowitz_z, 0)

        path_list = cons.emotion_data_path
        emotion_list = cons.emotion_key_list
        perform_z_by_list = encode_all_emotionNet_data(path_list, emotion_list)
        for z_dict in perform_z_by_list:
            z_dict['z'][0] = th.Tensor(horowitz_z).to(DEVICE)
            load_file_and_generate_performance('test_pieces/schumann/', 'Schumann', z=z_dict, start_tempo=45)
