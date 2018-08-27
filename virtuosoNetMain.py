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


parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test")
# parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./mxp/testdata/chopin10-3/", help="folder path of test mat")
# parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
parser.add_argument("-data", "--dataName", type=str, default="tempo_primo", help="dat file name")
parser.add_argument("--resume", type=str, default="model_best.pth.tar", help="best model path")
parser.add_argument("-tempo", "--startTempo", type=int, default=0, help="start tempo. zero to use xml first tempo")

args = parser.parse_args()

### parameters
train_x = Variable(torch.Tensor())
input_size = 28
hidden_size = 128
final_hidden = 128
num_layers = 4
num_output = 11
training_ratio = 0.95
learning_rate = 0.0005
num_epochs = 150

time_steps = 35
batch_size = 20
valid_batch_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Model

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_output):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.final_hidden_size = final_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.output_lstm = nn.LSTM(hidden_size * 2 + num_output, final_hidden, num_layers=1, batch_first=True, bidirectional=False)
        # self.fc = nn.Linear(hidden_size * 2, num_output)  # 2 for bidirection
        self.fc = nn.Linear(final_hidden, num_output)

    def forward(self, x, y, hidden, final_hidden, hidden_out = False):
        # Set initial state\s
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        #
        # h1 = torch.zeros(1, x.size(0), self.final_hidden_size).to(device)
        # c1 = torch.zeros(1, x.size(0), self.final_hidden_size).to(device)
        # Forward propagate LSTM
        if not isinstance(hidden_out, torch.Tensor):
            hidden_out, hidden = self.lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        if not isinstance(y, torch.Tensor):
            return hidden_out, hidden

        out_combined = torch.cat((hidden_out,y), 2)
        out, final_hidden = self.output_lstm(out_combined, final_hidden)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out, hidden, final_hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, h0)

    def init_final_layer(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.final_hidden_size).to(device)
        return (h0, h0)


model = BiRNN(input_size, hidden_size, num_layers, num_output).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def perform_xml(input, input_y, is_beat_list, tempo_stats, start_tempo='0'):
    with torch.no_grad():  # no need to track history in sampling
        input.view((1,-1,input_size))
        # num_notes = input.shape[1]
        hidden = model.init_hidden(1)
        final_hidden = model.init_final_layer(1)

        hidden_output, hidden = model(input, False, hidden, final_hidden)

        # print(input_y.shape)
        piece_length = input.shape[1]
        outputs = []

        previous_tempo = start_tempo
        # print(10 ** (previous_tempo * tempo_stats[1] + tempo_stats[0]))
        save_tempo = 0
        num_added_tempo = 0
        previous_position = input[0,0,7] #xml_position of first note
        for i in range(piece_length):
            is_beat = is_beat_list[i]
            # print(is_beat)
            if is_beat and input[0,i,7] > previous_position: # is_beat and
                previous_position = input[0,i,7]
                previous_tempo = save_tempo\
                                 / num_added_tempo
                save_tempo =0
                num_added_tempo = 0
                # print(10 ** (previous_tempo * tempo_stats[1] + tempo_stats[0]))

            input_y = input_y.cpu()
            # print(previous_tempo)
            input_y[0, 0, 0] = previous_tempo
            input_y = input_y.to(device)

            note_feature = input[0,i,:].view(1,1,input_size).to(device)
            # print(hidden_output.shape)
            temp_hidden_output = hidden_output[0,i,:].view(1,1,-1)
            output, _, final_hidden = model(note_feature, input_y, hidden, final_hidden, temp_hidden_output)
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



def batch_time_step_run(x,y,prev_feature, step, batch_size=batch_size, time_steps=time_steps, model=model):

    if step < total_batch_num - 1:
        batch_start = step * batch_size * time_steps
        batch_end = (step + 1) * batch_size * time_steps
        batch_x = Variable(torch.Tensor(x[batch_start:batch_end]))
        batch_y = Variable(torch.Tensor(y[batch_start:batch_end]))
        input_y = Variable(torch.Tensor(prev_feature[batch_start:batch_end]))
        # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
        batch_x = batch_x.view((batch_size, time_steps, input_size)).to(device)
        batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
        input_y = input_y.view((batch_size, time_steps, num_output)).to(device)
        hidden = model.init_hidden(batch_x.size(0))
        final_hidden = model.init_final_layer(batch_x.size(0))
    else:
        # num_left_data = data_size % batch_size*time_steps
        batch_start = -(batch_size * time_steps)
        batch_x = Variable(
            torch.Tensor(x[batch_start:]))
        batch_y = Variable(
            torch.Tensor(y[batch_start:]))
        input_y = Variable(
            torch.Tensor(prev_feature[batch_start:]))
        # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
        batch_x = batch_x.view((batch_size, time_steps, input_size)).to(device)
        batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
        input_y = input_y.view((batch_size, time_steps, num_output)).to(device)
        hidden = model.init_hidden(batch_x.size(0))
        final_hidden = model.init_final_layer(batch_x.size(0))

    outputs, hidden, final_hidden = model(batch_x, input_y, hidden, final_hidden)
    loss = criterion(outputs, batch_y)
    ioi_loss = criterion(outputs[:, :, 0], batch_y[:, :, 0])
    art_loss = criterion(outputs[:, :, 1], batch_y[:, :, 1])
    vel_loss = criterion(outputs[:, :, 2], batch_y[:, :, 2])
    dev_loss = criterion(outputs[:, :, 3], batch_y[:, :, 3])

    return outputs, loss, ioi_loss, art_loss, vel_loss, dev_loss


### training

if args.sessMode == 'train':
    # load data
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
        main_loss_total = []
        ioi_loss_total =[]
        art_loss_total =[]
        vel_loss_total =[]
        dev_loss_total =[]
        for xy_tuple in train_xy:
            train_x = xy_tuple[0]
            train_y = xy_tuple[1]
            prev_feature = xy_tuple[2]

            data_size = len(train_x)
            total_batch_num = int(math.ceil(data_size / (time_steps * batch_size)))

            for step in range(total_batch_num):
                outputs, loss, ioi_loss, art_loss, vel_loss, dev_loss = batch_time_step_run(train_x, train_y, prev_feature, step)
                # if step < total_batch_num - 1:
                #     batch_start = step * batch_size * time_steps
                #     batch_end = (step+1)*batch_size*time_steps
                #     batch_x = Variable(torch.Tensor(train_x[batch_start:batch_end]))
                #     batch_y = Variable(torch.Tensor(train_y[batch_start:batch_end]))
                #     input_y =  Variable(torch.Tensor(prev_feature[batch_start:batch_end]))
                #     # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
                #     batch_x = batch_x.view((batch_size, time_steps, input_size)).to(device)
                #     batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
                #     input_y = input_y.view((batch_size, time_steps, num_output)).to(device)
                #     hidden = model.init_hidden(batch_x.size(0))
                #     final_hidden = model.init_final_layer(batch_x.size(0))
                # else:
                #     # num_left_data = data_size % batch_size*time_steps
                #     batch_start = -(batch_size * time_steps +1)
                #     batch_x = Variable(
                #         torch.Tensor(train_x[batch_start:]))
                #     batch_y = Variable(
                #         torch.Tensor(train_y[batch_start:]))
                #     input_y = Variable(
                #         torch.Tensor(prev_feature[batch_start:]))
                #     # input_y = torch.cat((zero_tensor, batch_y[0:batch_size * time_steps-1]), 0).view((batch_size, time_steps,num_output)).to(device)
                #     batch_x = batch_x.view((batch_size, time_steps, input_size)).to(device)
                #     batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
                #     input_y = input_y.view((batch_size, time_steps, num_output)).to(device)
                #     hidden = model.init_hidden(batch_x.size(0))
                #     final_hidden = model.init_final_layer(batch_x.size(0))


                # outputs, hidden, final_hidden = model(batch_x, input_y, hidden, final_hidden)
                # loss = criterion(outputs, batch_y)
                # ioi_loss = criterion(outputs[:,:,0], batch_y[:,:,0])
                # art_loss = criterion(outputs[:,:,1], batch_y[:,:,1])
                # vel_loss = criterion(outputs[:,:,2], batch_y[:,:,2])
                # dev_loss = criterion(outputs[:,:,3], batch_y[:,:,3])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                loss_total.append(loss.item())
                ioi_loss_total.append(ioi_loss.item())
                art_loss_total.append(art_loss.item())
                vel_loss_total.append(vel_loss.item())
                dev_loss_total.append(dev_loss.item())

        print('Epoch [{}/{}], Loss: {:.4f}, IOI: {:.4f}, Art: {:.4f}, Vel: {:.4f}, Dev: {:.4f}'
              .format(epoch + 1, num_epochs, np.mean(loss_total),
                      np.mean(ioi_loss_total), np.mean(art_loss_total), np.mean(vel_loss_total), np.mean(dev_loss_total)) )


        ## Validation
        valid_loss_total = []
        ioi_loss_total =[]
        art_loss_total =[]
        vel_loss_total =[]
        dev_loss_total =[]

        for xy_tuple in test_xy:
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            prev_feature = xy_tuple[2]
            is_beat_list = xy_tuple[3]
            batch_x = Variable(torch.Tensor(test_x)).view((1, -1, input_size)).to(device)
            batch_y = Variable(torch.Tensor(test_y)).view((1, -1, num_output))
            input_y = Variable(torch.Tensor(prev_feature)).view((1,-1,num_output)).to(device)
            # hidden = model.init_hidden(1)
            # final_hidden = model.init_final_layer(1)
            # outputs, hidden, final_hidden = model(batch_x, input_y, hidden, final_hidden)

            batch_x = Variable(torch.Tensor(test_x)).view((1, -1, input_size)).to(device)
            #
            input_y = torch.zeros(num_output)

            input_y[0] = test_y[0][0]
            input_y[2] = -0.25
            input_y = input_y.view((1,1, num_output)).to(device)
            outputs = perform_xml(batch_x, input_y, is_beat_list, tempo_stats, start_tempo=test_y[0][0])
            outputs = torch.Tensor(outputs).view((1,-1,num_output))
            valid_loss = criterion(outputs, batch_y)
            ioi_loss = criterion(outputs[:,:,0], batch_y[:,:,0])
            art_loss = criterion(outputs[:,:,1], batch_y[:,:,1])
            vel_loss = criterion(outputs[:,:,2], batch_y[:,:,2])
            dev_loss = criterion(outputs[:,:,3], batch_y[:,:,3])

            valid_loss_total.append(valid_loss.item())
            ioi_loss_total.append(ioi_loss.item())
            art_loss_total.append(art_loss.item())
            vel_loss_total.append(vel_loss.item())
            dev_loss_total.append(dev_loss.item())

        mean_valid_loss = np.mean(valid_loss_total)
        print("Valid Loss= {:.4f} , IOI: {:.4f}, Art: {:.4f}, Vel: {:.4f}, Dev: {:.4f}"
              .format(mean_valid_loss,  np.mean(ioi_loss_total), np.mean(art_loss_total),
                      np.mean(vel_loss_total), np.mean(dev_loss_total)))

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
    test_x, xml_notes, xml_doc, is_beat_list = xml_matching.read_xml_to_array(path_name, means, stds, args.startTempo)
    batch_x = Variable(torch.FloatTensor(test_x)).to(device)
    batch_x = batch_x.view(1, -1, input_size)

    for i in range(len(stds)):
        for j in range(len(stds[i])):
            if stds[i][j] == 0:
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

    input_y = torch.zeros(1, num_output).view((1, 1, num_output))
    input_y[0,0,0] = start_tempo
    input_y[0,0,1] = 1
    input_y[0,0,2] = 64
    for i in range(3,num_output):
        input_y[0,0,i] = 0
    for i in range(num_output):
        input_y[0,0,i] -= means[1][i]
        input_y[0,0,i] /= stds[1][i]

    input_y[0,0,0] = 0
    input_y = input_y.to(device)
    tempo_stats = [means[1][0], stds[1][0]]

    prediction = perform_xml(batch_x, input_y, is_beat_list, tempo_stats, start_tempo=start_tempo_norm)
    # outputs = outputs.view(-1, num_output)
    prediction = np.squeeze(np.asarray(prediction))

    # prediction = outputs.cpu().detach().numpy()
    for i in range(num_output):
        prediction[:, i] *= stds[1][i]
        prediction[:, i] += means[1][i]

    output_features = []
    for pred in prediction:
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
        print(10 ** feat.qpm)
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
        is_beat_list = xy_tuple[3]
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
            is_beat_batch = is_beat_list[batch_start:batch_end]
            # batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)
            # input_y = input_y.view((batch_size, time_steps, num_output)).to(device)

            # hidden = model.init_hidden(1)
            # final_hidden = model.init_final_layer(1)
            # outputs, hidden, final_hidden = model(batch_x, input_y, hidden, final_hidden)
            #
            input_y = torch.zeros(num_output)

            input_y[0] = batch_y[0][0]
            input_y[2] = -0.25
            input_y = input_y.view((1, 1, num_output)).to(device)
            outputs = perform_xml(batch_x, input_y, is_beat_list, tempo_stats, start_tempo=batch_y[0][0])
            outputs = torch.Tensor(outputs).view((1, -1, num_output))

            outputs = outputs.cpu().detach().numpy()
            # batch_y = batch_y.cpu().detach().numpy()
            batch_y = np.asarray(batch_y).reshape((1,-1,num_output))
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
