import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import argparse
import math
import numpy as np
import shutil
import os
import xml_matching
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import perf_worm
import copy
import random



parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test")
# parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./chopin_cleaned/Chopin_Ballade/3/", help="folder path of test mat")
# parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
parser.add_argument("-data", "--dataName", type=str, default="score_test", help="dat file name")
parser.add_argument("--resume", type=str, default="eval_model_best.pth.tar", help="best model path")
parser.add_argument("-tempo", "--startTempo", type=int, default=0, help="start tempo. zero to use xml first tempo")
parser.add_argument("-trill", "--trainTrill", type=bool, default=False, help="train trill")
parser.add_argument("--beatTempo", type=bool, default=True, help="cal tempo from beat level")
parser.add_argument("-voice", "--voiceNet", type=bool, default=True, help="network in voice level")


args = parser.parse_args()


class NetParams:
    class Param:
        def __init__(self):
            self.size = 0
            self.layer = 0

    def __init__(self):
        self.note = self.Param()
        self.beat = self.Param()
        self.measure = self.Param()
        self.final = self.Param()
        self.voice = self.Param()
        self.sum = self.Param()

### parameters
NET_PARAM = NetParams()

NET_PARAM.note.layer = 2
NET_PARAM.note.size = 64
NET_PARAM.beat.layer = 1
NET_PARAM.beat.size = 16
NET_PARAM.measure.layer = 1
NET_PARAM.measure.size= 8
NET_PARAM.final.layer = 1
NET_PARAM.final.size = 16
NET_PARAM.voice.layer = 2
NET_PARAM.voice.size = 16
NET_PARAM.sum.layer = 2
NET_PARAM.sum.size = 64

learning_rate = 0.0003
time_steps = 300
num_epochs = 150
num_key_augmentation = 3

input_size = 41
output_size = 16
training_ratio = 0.75
DROP_OUT = 0.5


num_trill_param = 5
is_trill_index = -9

QPM_INDEX = 0
VOICE_IDX = 11
TEMPO_IDX = 25
qpm_primo_index = 5
tempo_primo_index = -2
num_tempo_info = 3

batch_size = 1
valid_batch_size = 50

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

NET_PARAM.final.input = NET_PARAM.note.size * 2 + NET_PARAM.beat.size *2 + NET_PARAM.measure.size * 2
# if args.trainTrill is False:
#     NET_PARAM.final.input -= num_trill_param
if args.voiceNet:
    NET_PARAM.final.input += NET_PARAM.voice.size * 2

### Model


class HAN(nn.Module):
    def __init__(self, input_size, network_parameters, num_output, num_trill_param=5):
        super(HAN, self).__init__()
        self.input_size = input_size
        self.num_layers = network_parameters.note.layer
        self.hidden_size = network_parameters.note.size
        self.num_beat_layers = network_parameters.beat.layer
        self.beat_hidden_size = network_parameters.beat.size
        self.num_measure_layers = network_parameters.measure.layer
        self.measure_hidden_size = network_parameters.measure.size
        self.final_hidden_size = network_parameters.final.size
        self.num_voice_layers = network_parameters.voice.layer
        self.voice_hidden_size = network_parameters.voice.size
        self.summarize_layers = network_parameters.sum.layer
        self.summarize_size = network_parameters.sum.size
        self.final_input = NET_PARAM.final.input
        self.output_size = num_output


        self.lstm = nn.LSTM(input_size+num_output, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        self.output_lstm = nn.LSTM(self.final_input, self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        # if args.trainTrill:
        #     self.output_lstm = nn.LSTM((self.hidden_size + self.beat_hidden_size + self.measure_hidden_size) *2 + num_output + num_tempo_info,
        #                                self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        # else:
        #     self.output_lstm = nn.LSTM(
        #         (self.hidden_size + self.beat_hidden_size + self.measure_hidden_size) * 2 + num_output - num_trill_param + num_tempo_info,
        #         self.final_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.beat_attention = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.beat_hidden = nn.LSTM(self.hidden_size*2, self.beat_hidden_size, self.num_beat_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        self.measure_attention = nn.Linear(self.beat_hidden_size*2, self.beat_hidden_size*2)
        self.measure_hidden = nn.LSTM(self.beat_hidden_size*2, self.measure_hidden_size, self.num_measure_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(self.final_hidden_size + 9, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

        self.softmax = nn.Softmax(dim=0)
        self.trill_fc = nn.Linear(self.final_hidden_size, num_trill_param)
        self.sigmoid = nn.Sigmoid()
        self.beat_tempo_forward = nn.LSTM(self.beat_hidden_size*2+1+3+3, self.beat_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.beat_tempo_fc = nn.Linear(self.beat_hidden_size, 1)
        self.voice_net = nn.LSTM(input_size+num_output, self.voice_hidden_size, self.num_voice_layers, batch_first=True, bidirectional=True, dropout=DROP_OUT)
        self.summarize_net = nn.LSTM(self.final_input, self.summarize_size, self.summarize_layers, batch_first=True, bidirectional=True)

    def forward(self, x, years, final_hidden, note_locations, start_index,
                hidden_out = False, beat_hidden_spanned = False, measure_hidden_spanned = False, beat_tempos = False, beat_changed=False, voice_out=False):
        beat_numbers = [x.beat for x in note_locations]
        measure_numbers = [x.measure for x in note_locations]
        voice_numbers = [x.voice for x in note_locations]

        hidden = self.init_hidden(x.size(0))
        beat_hidden = self.init_beat_layer(x.size(0))
        measure_hidden = self.init_measure_layer(x.size(0))

        hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        beat_nodes = self.make_beat_node(hidden_out, beat_numbers, start_index)
        beat_hidden_out, beat_hidden = self.beat_hidden(beat_nodes, beat_hidden)
        measure_nodes = self.make_measure_node(beat_hidden_out, measure_numbers, beat_numbers, start_index)
        measure_hidden_out, measure_hidden = self.measure_hidden(measure_nodes, measure_hidden)
        num_notes = hidden_out.shape[1]
        beat_hidden_spanned = self.span_beat_to_note_num(beat_hidden_out, beat_numbers, num_notes, start_index)
        measure_hidden_spanned = self.span_beat_to_note_num(measure_hidden_out, measure_numbers, num_notes, start_index)

        if args.voiceNet:
            temp_voice_numbers = voice_numbers[start_index:start_index+x.size(1)]
            if temp_voice_numbers == []:
                temp_voice_numbers = voice_numbers[start_index:]
            max_voice = max(temp_voice_numbers)
            voice_hidden = self.init_voice_layer(1, max_voice)
            voice_out, voice_hidden = self.run_voice_net(x, voice_hidden, temp_voice_numbers, max_voice)
            out_combined = torch.cat((hidden_out, beat_hidden_spanned, measure_hidden_spanned, voice_out,), 2)
        else:
            out_combined = torch.cat((hidden_out, beat_hidden_spanned, measure_hidden_spanned), 2)

        out, final_hidden = self.output_lstm(out_combined, final_hidden)
        out = torch.cat((out[0,-1,:],years))

        out = self.fc(out)

        return out, final_hidden

    def sum_with_attention(self, hidden, attention_net):
        attention = attention_net(hidden)
        attention = self.softmax(attention)
        upper_node = hidden * attention
        upper_node = torch.sum(upper_node, dim=0)

        return upper_node


    def make_beat_node(self, hidden_out, beat_number, start_index):
        beat_nodes = []
        prev_beat = beat_number[start_index]
        beat_notes_start = 0
        beat_notes_end = 0
        num_notes = hidden_out.shape[1]
        for note_index in range(num_notes):
            actual_index = start_index + note_index
            if beat_number[actual_index] > prev_beat:
                #new beat start
                beat_notes_end = note_index
                corresp_hidden = hidden_out[0, beat_notes_start:beat_notes_end, :]
                beat = self.sum_with_attention(corresp_hidden, self.beat_attention)
                beat_nodes.append(beat)

                beat_notes_start = note_index
                prev_beat = beat_number[actual_index]

        last_hidden =  hidden_out[0, beat_notes_end:, :]
        beat = self.sum_with_attention(last_hidden, self.beat_attention)
        beat_nodes.append(beat)

        beat_nodes = torch.stack(beat_nodes).view(1,-1,self.hidden_size*2)
        # beat_nodes = torch.Tensor(beat_nodes)

        return beat_nodes

    def make_measure_node(self, beat_out, measure_number, beat_number, start_index):
        measure_nodes = []
        prev_measure = measure_number[start_index]
        measure_beats_start = 0
        measure_beats_end = 0
        num_beats = beat_out.shape[1]
        start_beat = beat_number[start_index]
        for beat_index in range(num_beats):
            current_beat = start_beat + beat_index
            current_note_index = beat_number.index(current_beat)

            if measure_number[current_note_index] > prev_measure:
                # new beat start
                measure_beats_end = beat_index
                corresp_hidden = beat_out[0, measure_beats_start:measure_beats_end, :]
                measure = self.sum_with_attention(corresp_hidden, self.measure_attention)
                measure_nodes.append(measure)

                measure_beats_start = beat_index
                prev_measure = measure_number[beat_index]

        last_hidden = beat_out[0, measure_beats_end:, :]
        measure = self.sum_with_attention(last_hidden, self.measure_attention)
        measure_nodes.append(measure)

        measure_nodes = torch.stack(measure_nodes).view(1,-1,self.beat_hidden_size*2)

        return measure_nodes

    def span_beat_to_note_num(self, beat_out, beat_number, num_notes, start_index):
        start_beat = beat_number[start_index]
        num_beat = beat_out.shape[1]
        span_mat = torch.zeros(1, num_notes, num_beat)
        node_size = beat_out.shape[2]
        for i in range(num_notes):
            beat_index = beat_number[start_index+i] - start_beat
            span_mat[0,i,beat_index] = 1
        span_mat = span_mat.to(device)

        spanned_beat = torch.bmm(span_mat, beat_out)
        return spanned_beat

    def note_tempo_infos_to_beat(self, y, beat_numbers, start_index, index=None):
        beat_tempos = []
        num_notes = y.size(1)
        prev_beat = -1
        for i in range(num_notes):
            cur_beat = beat_numbers[start_index+i]
            if cur_beat > prev_beat:
                if index is None:
                    beat_tempos.append(y[0,i,:])
                else:
                    beat_tempos.append(y[0,i,index])
                prev_beat = cur_beat
        num_beats = len(beat_tempos)
        beat_tempos = torch.stack(beat_tempos).view(1,num_beats,-1)
        return beat_tempos

    def run_voice_net(self, batch_x, voice_hidden, voice_numbers, max_voice):
        num_notes = batch_x.size(1)
        output = torch.zeros(1, batch_x.size(1), self.voice_hidden_size * 2).to(device)
        voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1,max_voice+1):
            voice_x_bool = voice_numbers == i
            num_voice_notes = torch.sum(voice_x_bool)
            if num_voice_notes > 0:
                span_mat = torch.zeros(num_notes, num_voice_notes)
                note_index_in_voice = 0
                for j in range(num_notes):
                    if voice_x_bool[j] ==1:
                        span_mat[j, note_index_in_voice] = 1
                        note_index_in_voice += 1
                span_mat = span_mat.view(1,num_notes,-1).to(device)
                voice_x = batch_x[0,voice_x_bool,:].view(1,-1, self.input_size + self.output_size)
                ith_hidden = voice_hidden[i-1]

                ith_voice_out, ith_hidden = self.voice_net(voice_x)
                output += torch.bmm(span_mat, ith_voice_out)
        return output, voice_hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, h0)

    def init_final_layer(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.final_hidden_size).to(device)
        return (h0, h0)

    def init_beat_layer(self, batch_size):
        h0 = torch.zeros(self.num_beat_layers * 2, batch_size, self.beat_hidden_size).to(device)
        return (h0, h0)

    def init_measure_layer(self, batch_size):
        h0 = torch.zeros(self.num_measure_layers * 2, batch_size, self.measure_hidden_size).to(device)
        return (h0, h0)

    def init_beat_tempo_forward(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.beat_hidden_size).to(device)
        return (h0, h0)

    def init_voice_layer(self, batch_size, max_voice):
        layers = []
        for i in range(max_voice):
            h0 = torch.zeros(self.num_voice_layers * 2, batch_size, self.voice_hidden_size).to(device)
            layers.append((h0,h0))
        return layers




# model = BiRNN(input_size, hidden_size, num_layers, num_output).to(device)
model = HAN(input_size, NET_PARAM, output_size).to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state, is_best, filename='eval_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'eval_model_best.pth.tar')

def key_augmentation(data_x, key_change):
    # key_change = 0
    data_x_aug = copy.deepcopy(data_x)
    pitch_start_index = 12
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

    return data_x_aug


def perform_xml(input, input_y, note_locations, tempo_stats, start_tempo='0', valid_y = None):
    with torch.no_grad():  # no need to track history in sampling
        input.view((1,-1,input_size))
        # num_notes = input.shape[1]
        final_hidden = model.init_final_layer(1)
        tempo_hidden = model.init_beat_tempo_forward(1)

        # hidden_output, hidden = model(input, False, hidden, final_hidden)
        hidden_output, beat_output, measure_output, voice_output =\
            model(batch_x, False, final_hidden, tempo_hidden, note_locations, 0)


        # print(input_y.shape)
        piece_length = input.shape[1]
        outputs = []

        previous_tempo = start_tempo
        # print(10 ** (previous_tempo * tempo_stats[1] + tempo_stats[0]))
        save_tempo = 0
        num_added_tempo = 0
        # previous_position = input[0,0,7] #xml_position of first note
        prev_beat = 0
        for i in range(piece_length):
            # is_beat = is_beat_list[i]
            beat = note_locations[i].beat
            # print(is_beat)
            if beat > prev_beat and num_added_tempo > 0: # is_beat and
                prev_beat = beat
                previous_tempo = save_tempo / num_added_tempo
                save_tempo =0
                num_added_tempo = 0
                # print(10 ** (previous_tempo * tempo_stats[1] + tempo_stats[0]))
                beat_changed= True
            else:
                beat_changed = False

            input_y = input_y.cpu()
            # print(previous_tempo)
            input_y[0, 0, 0] = previous_tempo
            input_y = input_y.to(device)
            if isinstance(valid_y, torch.Tensor) and i < 100:
                input_y = valid_y[0,i-1,:].to(device).view(1,1,-1)

            note_feature = input[0,i,:].view(1,1,input_size).to(device)
            # print(hidden_output.shape)
            temp_hidden_output = hidden_output[0, i, :].view(1, 1, -1)
            temp_beat_output = beat_output[0, i, :].view(1, 1, -1)
            temp_measure_output = measure_output[0, i, :].view(1, 1, -1)
            if args.voiceNet:
                temp_voice_output = voice_output[0,i,:].view(1,1,-1)
            else:
                temp_voice_output = 0

            # output, _, final_hidden = model(note_feature, input_y, hidden, final_hidden, temp_hidden_output)
            output, final_hidden, tempo_hidden = model(note_feature, input_y, final_hidden, tempo_hidden, note_locations, i,
                                            hidden_out=temp_hidden_output,
                                            beat_hidden_spanned = temp_beat_output, measure_hidden_spanned=temp_measure_output,
                                            beat_changed= beat_changed, voice_out=temp_voice_output)

            output_for_save = output.cpu().detach().numpy()
            input_y = output
            ## change tempo of input_y
            # if is_beat:
            #     if input[0, i, 6] > previous_position:
            #         save_tempo = output_for_save[0][0][0] #save qpm of this beat
            #
            save_tempo += output_for_save[0][0][0]
            num_added_tempo += 1
            outputs.append(output_for_save)

        return outputs



def batch_time_step_run(x,y, score, note_locations, step, batch_size=batch_size, time_steps=time_steps, model=model, validation=False):

    if step < total_batch_num - 1:
        batch_start = step * batch_size * time_steps
        batch_end = (step + 1) * batch_size * time_steps
        batch_x = torch.Tensor(x[batch_start:batch_end]).view((batch_size, time_steps, input_size)).to(device)
        batch_x2 = torch.Tensor(y[batch_start:batch_end]).view((batch_size, time_steps, output_size)).to(device)
        # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
        batch_x = torch.cat((batch_x,batch_x2), 2)


    else:
        # num_left_data = data_size % batch_size*time_steps
        batch_start = -(batch_size * time_steps)
        batch_x = torch.Tensor(x[batch_start:]).view((batch_size, time_steps, input_size)).to(device)
        batch_x2 = torch.Tensor(y[batch_start:]).view((batch_size, time_steps, output_size)).to(device)
        batch_y = torch.Tensor(score[batch_start:])
        # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
        batch_x = torch.cat((batch_x, batch_x2), 2)

    batch_y = torch.Tensor(score).to(device)
    score = batch_y[0]
    years = batch_y[1:]

    final_hidden = model.init_final_layer(batch_x.size(0))
    if validation:
        temp_model = model.eval()
    else:
        temp_model = model.train()
    outputs, final_hidden = temp_model(batch_x, years, final_hidden, note_locations, batch_start)
    loss = criterion(outputs, score)


    return outputs, loss


### training

if args.sessMode == 'train':
    # load data
    print('Loading the training data...')
    with open(args.dataName + ".dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    with open(args.dataName + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        means, stds = u.load()

    perform_num = len(complete_xy)
    tempo_stats = [means[1][0], stds[1][0]]

    train_perf_num = int(perform_num * training_ratio)
    train_xy = complete_xy[:train_perf_num]
    test_xy = complete_xy[train_perf_num:]
    print('number of performance: ', perform_num, 'number of test perf: ', len(test_xy))

    print(train_xy[0][0][0])
    best_valid_loss = float("inf")
    # total_step = len(train_loader)
    for epoch in range(num_epochs):
        loss_total = []
        for xy_tuple in train_xy:
            train_x = xy_tuple[0]
            train_y = xy_tuple[1]
            prev_feature = xy_tuple[2]
            note_locations = xy_tuple[3]
            score = xy_tuple[4]


            data_size = len(train_x)
            total_batch_num = int(math.ceil(data_size / (time_steps * batch_size)))

            key_lists = [0]
            key = 0
            for i in range(num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(num_key_augmentation):
                key = key_lists[0]
                temp_train_x = key_augmentation(train_x, key)

                for step in range(total_batch_num):
                    outputs, loss = \
                        batch_time_step_run(temp_train_x, train_y, score, note_locations, step)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer.step()
                    loss_total.append(loss.item())

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, np.mean(loss_total)))


        ## Validation
        valid_loss_total = []
        correct_guess = 0
        wrong_guess = 0
        for xy_tuple in test_xy:
            train_x = xy_tuple[0]
            train_y = xy_tuple[1]
            prev_feature = xy_tuple[2]
            note_locations = xy_tuple[3]
            score = xy_tuple[4]

            data_size = len(train_x)
            total_batch_num = int(math.ceil(data_size / (time_steps * batch_size)))

            key_lists = [0]
            key = 0
            for i in range(num_key_augmentation):
                while key in key_lists:
                    key = random.randrange(-5, 7)
                key_lists.append(key)

            for i in range(num_key_augmentation):
                key = key_lists[0]
                temp_train_x = key_augmentation(train_x, key)

                for step in range(total_batch_num):
                    outputs, loss = \
                        batch_time_step_run(temp_train_x, train_y, score, note_locations, step, validation=True)
                    valid_loss_total.append(loss.item())
                    if round(outputs.item()) == score[0]:
                        correct_guess +=1
                    else:
                        wrong_guess +=1



        mean_valid_loss = np.mean(valid_loss_total)
        accuracy = correct_guess / (wrong_guess+correct_guess)
        print("Valid Loss= {:.4f}, Accuracy is {:.4f}"
              .format(mean_valid_loss, accuracy ))

        is_best = mean_valid_loss < best_valid_loss
        # if np.mean(valid_loss_total) < best_valid_loss:
        #     best_valid_loss = np.mean(valid_loss_total)
        #     get_worse_count = 0
        # else:
        #     get_worse_count += 1
        #
        # if get_worse_count > 5:
        #     break

        best_valid_loss = min(mean_valid_loss, best_valid_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_valid_loss': best_valid_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    #end of epoch



elif args.sessMode=='test':
### test session
    with open(args.dataName + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        means, stds = u.load()
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    path_name = args.testPath
    test_x, xml_notes, xml_doc, note_locations = xml_matching.read_xml_to_array(path_name, means, stds, args.startTempo)
    batch_x = torch.Tensor(test_x).to(device)
    batch_x = batch_x.view(1, -1, input_size)

    for i in range(len(stds)):
        for j in range(len(stds[i])):
            if stds[i][j] < 1e-4:
                stds[i][j] = 1
    #
    # test_x = np.asarray(test_x)
    # timestep_quantize_num = int(math.ceil(test_x.shape[0] / time_steps))
    # padding_size = timestep_quantize_num * time_steps - test_x.shape[0]
    # test_x_padded = np.pad(test_x, ((0, padding_size), (0, 0)), 'constant')
    # batch_x = test_x_padded.reshape((-1, time_steps, input_size))
    # batch_x = Variable(torch.from_numpy(batch_x)).float().to(device)
    # tempos = xml_doc.get_tempos()

    if args.startTempo == 0:
        start_tempo = xml_notes[0].state_fixed.qpm / 60 * xml_notes[0].state_fixed.divisions
        start_tempo = math.log(start_tempo, 10)
        start_tempo_norm = (start_tempo - means[1][0]) / stds[1][0]
    else:
        start_tempo = math.log(args.startTempo, 10)
    start_tempo_norm = (start_tempo - means[1][0]) / stds[1][0]

    if args.trainTrill:
        input_y = torch.zeros(1, 1, output_size)
    else:
        input_y = torch.zeros(1, 1, output_size - num_trill_param)
    # input_y[0,0,0] = start_tempo
    # # input_y[0,0,1] = 1
    # # input_y[0,0,2] = 64
    # for i in range(output_size - num_trill_param):
    #     input_y[0,0,i] -= means[1][i]
    #     input_y[0,0,i] /= stds[1][i]
    #
    input_y[0,0,0] = start_tempo_norm
    input_y = input_y.to(device)
    tempo_stats = [means[1][0], stds[1][0]]

    prediction = perform_xml(batch_x, input_y, note_locations, tempo_stats, start_tempo=start_tempo_norm)
    # outputs = outputs.view(-1, num_output)
    prediction = np.squeeze(np.asarray(prediction))

    # prediction = outputs.cpu().detach().numpy()
    for i in range(output_size - num_trill_param):
        prediction[:, i] *= stds[1][i]
        prediction[:, i] += means[1][i]

    output_features = []
    # for i in range(100):
    #     pred = prediction[i]
    #     print(pred[0:4])
    num_notes = len(xml_notes)
    for i in range(num_notes):
        pred = prediction[i]
        # feat = {'IOI_ratio': pred[0], 'articulation': pred[1], 'loudness': pred[2], 'xml_deviation': 0,
        feat = xml_matching.MusicFeature()
        feat.qpm = pred[0]
        feat.articulation = pred[1]
        feat.velocity = pred[2]
        feat.xml_deviation = pred[3]
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

        if args.trainTrill:
            feat.trill_param = pred[11:16]
            feat.trill_param[0] = round(feat.trill_param[0])
            feat.trill_param[1] = round(feat.trill_param[1])
            feat.trill_param[2] = round(feat.trill_param[2])
            feat.trill_param[3] = round(feat.trill_param[3])
            feat.trill_param[4] = round(feat.trill_param[4])
        else:
            feat.trill_param = [0] * 5

        #
        # feat.passed_second = pred[0]
        # feat.duration_second = pred[1]
        # feat.pedal_refresh_time = pred[3]
        # feat.pedal_cut_time = pred[4]
        # feat.pedal_at_start = pred[5]
        # feat.pedal_at_end = pred[6]
        # feat.soft_pedal = pred[7]
        # feat.pedal_refresh = pred[8]
        # feat.pedal_cut = pred[9]


        # feat = {'qpm': pred[0], 'articulation': pred[1], 'loudness': pred[2], 'xml_deviation': pred[3],
        #         'pedal_at_start': pred[6], 'pedal_at_end': pred[7], 'soft_pedal': pred[8],
        #         'pedal_refresh_time': pred[4], 'pedal_cut_time': pred[5], 'pedal_refresh': pred[9],
        #         'pedal_cut': pred[10]}
        output_features.append(feat)
    num_notes = len(xml_notes)
    perf_worm.plot_performance_worm(output_features, path_name + 'perfWorm.png')

    # output_xml = xml_matching.apply_perform_features(xml_notes, output_features)
    output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time= 1, predicted=True)
    # output_xml = xml_matching.apply_time_position_features(xml_notes, output_features, start_time=1)

    output_midi = xml_matching.xml_notes_to_midi(output_xml)

    xml_matching.save_midi_notes_as_piano_midi(output_midi, path_name + 'performed_by_nn.mid', bool_pedal=False, disklavier=True)



elif args.sessMode=='plot':
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


    with open(args.dataName + ".dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    with open(args.dataName + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        means, stds = u.load()

    perform_num = len(complete_xy)
    tempo_stats = [means[1][0], stds[1][0]]

    train_perf_num = int(perform_num * training_ratio)
    train_xy = complete_xy[:train_perf_num]
    test_xy = complete_xy[train_perf_num:]

    n_tuple = 0
    for xy_tuple in test_xy:
        n_tuple += 1
        train_x = xy_tuple[0]
        train_y = xy_tuple[1]
        prev_feature = xy_tuple[2]
        note_locations = xy_tuple[3]

        data_size = len(train_x)
        total_batch_num = int(math.ceil(data_size / (time_steps * batch_size)))
        batch_size=1
        for step in range(total_batch_num - 1):
            batch_start = step * batch_size * time_steps
            batch_end = (step + 1) * batch_size * time_steps
            batch_x = Variable(
                torch.Tensor(train_x[batch_start:batch_end]))
            batch_y = train_y[batch_start:batch_end]
            # print(batch_x.shape, batch_y.shape)
            # input_y = Variable(
            #     torch.Tensor(prev_feature[step * batch_size * time_steps:(step + 1) * batch_size * time_steps]))
            # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
            batch_x = batch_x.view((batch_size, time_steps, input_size)).to(device)
            # is_beat_batch = is_beat_list[batch_start:batch_end]
            # batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
            # input_y = input_y.view((batch_size, time_steps, num_output)).to(device)

            # hidden = model.init_hidden(1)
            # final_hidden = model.init_final_layer(1)
            # outputs, hidden, final_hidden = model(batch_x, input_y, hidden, final_hidden)
            #
            if args.trainTrill:
                input_y = torch.zeros(1, 1, output_size)
            else:
                input_y = torch.zeros(1, 1, output_size - num_trill_param)

            input_y[0] = batch_y[0][0]
            input_y = input_y.view((1, 1, output_size)).to(device)
            outputs = perform_xml(batch_x, input_y, note_locations, tempo_stats, start_tempo=batch_y[0][0])
            outputs = torch.Tensor(outputs).view((1, -1, output_size))

            outputs = outputs.cpu().detach().numpy()
            # batch_y = batch_y.cpu().detach().numpy()
            batch_y = np.asarray(batch_y).reshape((1, -1, output_size))
            plt.figure(figsize=(10, 7))
            for i in range(4):
                plt.subplot(411+i)
                plt.plot(batch_y[0, :, i])
                plt.plot(outputs[0, :, i])
            # plt.subplot(412)
            # plt.plot(np.arange(0, time_steps), np.vstack((batch_y[0, :, 1], outputs[0, :, 1])))
            # plt.subplot(413)
            # plt.plot(np.arange(0, time_steps), np.vstack((batch_y[0, :, 2], outputs[0, :, 2])))
            # plt.subplot(414)
            # plt.plot(np.arange(0, time_steps), np.vstack((batch_y[0, :, 3], outputs[0, :, 3])))
            # os.mkdir('images')
            plt.savefig('images/piece{:d},seg{:d}.png'.format(n_tuple, step))
            plt.close()
