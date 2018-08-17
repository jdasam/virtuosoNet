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

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test")
# parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./mxp/testdata/chopin10-3/", help="folder path of test mat")
# parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
parser.add_argument("-data", "--dataName", type=str, default="chopin_cleaned_cont_pedal", help="dat file name")
parser.add_argument("--resume", type=str, default="model_best.pth.tar", help="best model path")

args = parser.parse_args()

### parameters
train_x = Variable(torch.Tensor())
input_size = 55
hidden_size = 64
num_layers = 2
num_output = 11
training_ratio = 0.8
learning_rate = 0.001
num_epochs = 30

time_steps = 200
batch_size = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Model

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_output):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_output)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
model = BiRNN(input_size, hidden_size, num_layers, num_output).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


### training

if args.sessMode == 'train':
    # load data
    with open(args.dataName + ".dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()
    perform_num = len(complete_xy)

    train_perf_num = int(perform_num * training_ratio)
    train_xy = complete_xy[:train_perf_num]
    test_xy = complete_xy[train_perf_num:]

    print(train_xy[0][0][0])
    best_valid_loss = float("inf")
    # total_step = len(train_loader)
    for epoch in range(num_epochs):
        loss_total = []
        for xy_tuple in train_xy:
            train_x = xy_tuple[0]
            train_y = xy_tuple[1]

            data_size = len(train_x)
            total_batch_num = int(math.ceil(data_size / (time_steps * batch_size)))

            for step in range(total_batch_num - 1):
                batch_x = Variable(torch.FloatTensor(train_x[step*batch_size*time_steps:(step+1)*batch_size*time_steps]))
                batch_y = Variable(torch.FloatTensor(train_y[step * batch_size * time_steps:(step + 1) * batch_size * time_steps]))
                batch_x = batch_x.view((batch_size, time_steps, input_size)).to(device)
                batch_y = batch_y.view((batch_size, time_steps, num_output)).to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total.append(loss.item())

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, np.mean(loss_total) ))

        valid_loss_total = []

        for xy_tuple in test_xy:
            test_x = np.asarray(xy_tuple[0])
            test_y = np.asarray(xy_tuple[1])
            timestep_quantize_num = int(math.ceil(test_x.shape[0] / time_steps))
            padding_size = timestep_quantize_num * time_steps - test_x.shape[0]
            # print(test_x.shape, padding_size)
            test_x_padded = np.pad(test_x, ((0, padding_size), (0, 0)), 'constant')
            test_y_padded = np.pad(test_y, ((0, padding_size), (0, 0)), 'constant')
            # print(test_x_padded.shape)
            # print(data_size, batch_size, total_batch_num)

            # print(batch_x.shape, batch_y.shape)
            batch_x = test_x_padded.reshape((-1, time_steps, input_size))
            batch_y = test_y_padded.reshape((-1, time_steps, num_output))
            batch_x = Variable(torch.from_numpy(batch_x)).float().to(device)
            batch_y = Variable(torch.from_numpy(batch_y)).float().to(device)
            outputs = model(batch_x)
            valid_loss = criterion(outputs, batch_y)
            valid_loss_total.append(valid_loss.item())
        mean_valid_loss = np.mean(valid_loss_total)
        print("Valid Loss= " + \
              "{:.4f}".format(mean_valid_loss))

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

else:
### test session
    with open(args.dataName + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        means, stds = u.load()
    # print(means, stds)
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
    test_x, xml_notes = xml_matching.read_xml_to_array(path_name, means, stds)
    batch_x = Variable(torch.FloatTensor(test_x)).to(device)
    batch_x = batch_x.view(1, -1, input_size)
    print(batch_x.shape)
    #
    # test_x = np.asarray(test_x)
    # timestep_quantize_num = int(math.ceil(test_x.shape[0] / time_steps))
    # padding_size = timestep_quantize_num * time_steps - test_x.shape[0]
    # test_x_padded = np.pad(test_x, ((0, padding_size), (0, 0)), 'constant')
    # batch_x = test_x_padded.reshape((-1, time_steps, input_size))
    # batch_x = Variable(torch.from_numpy(batch_x)).float().to(device)


    outputs = model(batch_x)
    outputs = outputs.view(-1, num_output)

    prediction = outputs.cpu().detach().numpy()
    for i in range(11):
        prediction[:, i] *= stds[1][i]
        prediction[:, i] += means[1][i]

    output_features = []
    for pred in prediction:
        feat = {'IOI_ratio': pred[0], 'articulation': pred[1], 'loudness': pred[2], 'xml_deviation': pred[3],
                'pedal_at_start': pred[6], 'pedal_at_end': pred[7], 'soft_pedal': pred[8],
                'pedal_refresh_time': pred[4], 'pedal_cut_time': pred[5], 'pedal_refresh': pred[9],
                'pedal_cut': pred[10]}
        output_features.append(feat)

    output_xml = xml_matching.apply_perform_features(xml_notes, output_features)
    output_midi = xml_matching.xml_notes_to_midi(output_xml)

    xml_matching.save_midi_notes_as_piano_midi(output_midi, path_name + 'performed_by_nn.mid', bool_pedal=True, disklavier=True)