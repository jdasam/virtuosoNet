from __future__ import print_function
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.keras as keras
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import random
import argparse
import xml_matching
import midi_utils.midi_utils as midi_utils
from mxp import MusicXMLDocument



parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test")
# parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./mxp/testdata/chopin10-3/", help="folder path of test mat")
# parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
args = parser.parse_args()

# Training Parameters
learning_rate = 0.001
# training_steps = 10000
training_epochs = 20
batch_size = 4
display_step = 200
training_ratio = 0.8

# Network Parameters
num_input = 3
timesteps = 100 # timesteps
num_hidden = 64 # hidden layer num of features
num_output = 3 # loudness, articulation, ioi
input_length = 11


with open("pairs_entire3.dat", "rb") as f:
    dataset = pickle.load(f)

def make_windowed_data(features,input_length):
    feature_array = np.asarray(features)
    windowed_feature = []
    left_margin = (input_length-1)/2
    right_margin = (input_length - 1) / 2 +1

    # print(left_margin, right_margin)
    for i in range(feature_array.shape[0]):
        if i >= left_margin and i+right_margin<feature_array.shape[0]:
            temp_windowed = feature_array[i-left_margin:i+right_margin,:]
        elif i <left_margin:
            padding = left_margin-i
            temp_windowed = feature_array[:i+right_margin,:]
            temp_windowed = np.pad(temp_windowed, ((padding,0), (0,0)) , 'constant')
        else:
            padding = (i+right_margin) - feature_array.shape[0]
            temp_windowed = feature_array[i-left_margin:feature_array.shape[0],:]
            temp_windowed = np.pad(temp_windowed, ((0, padding), (0,0)) , 'constant')
        if not temp_windowed.shape[0] == input_length:
            print(temp_windowed.shape)
        windowed_feature.append(temp_windowed)
    windowed_feature = np.asarray(windowed_feature)
    # print(windowed_feature.shape)
    return windowed_feature




complete_xy = []
for piece in dataset:
    for perform in piece:
        train_x = []
        train_y = []
        for feature in perform:
            if not feature['IOI_ratio'] == None:
                train_x.append( [ feature['pitch_interval'],feature['duration_ratio'],feature['beat_position']  ] )
                # train_x.append( [ feature['pitch_interval'],feature['duration_ratio'] ] )
                train_y.append([ feature['IOI_ratio'], feature['articulation'] ,feature['loudness'] ])
        # windowed_train_x = make_windowed_data(train_x, input_length )
        complete_xy.append([train_x, train_y])

def get_mean_and_sd(performances, target_data, target_dimension):
    sum = 0
    squared_sum = 0
    count = 0
    for perf in performances:
        samples = perf[target_data]
        for sample in samples:
            value = sample[target_dimension]
            sum += value
            squared_sum += value*value
            count += 1
    data_mean = sum / count
    data_std = (squared_sum/count - data_mean **2) ** 0.5
    return data_mean, data_std

complete_xy_normalized = []
means = [[],[]]
stds = [[],[]]
for i1 in (0, 1):
    for i2 in range(3):
        mean_value, std_value = get_mean_and_sd(complete_xy, i1, i2)
        means[i1].append(mean_value)
        stds[i1].append(std_value)

for performance in complete_xy:
    complete_xy_normalized.append([])
    for index1 in (0, 1):
        complete_xy_normalized[-1].append([])
        for sample in performance[index1]:
            new_sample = []
            for index2 in (0, 1, 2):
                new_sample.append((sample[index2]-means[index1][index2])/stds[index1][index2])
            complete_xy_normalized[-1][index1].append(new_sample)

complete_xy_orig = complete_xy
print(len(complete_xy), len(complete_xy))
complete_xy = complete_xy_normalized

random.shuffle(complete_xy)




# complete_xy = np.asarray(complete_xy)
# perform_num = complete_xy.shape[0]
perform_num = len(complete_xy)
train_perf_num = int(perform_num * training_ratio)
train_xy = complete_xy[:train_perf_num]
test_xy = complete_xy[train_perf_num:]



# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, timesteps, num_output])


def frame_wise_projection(input, feature_dim, out_dim):
    with tf.variable_scope('projection') as scope:
        kernel = tf.get_variable('weight', shape=[1, feature_dim, 1, out_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)
        output = tf.squeeze(output, axis=2, name='output')
    return output


def keras_frame_wise_projection(input, feature_dim, out_dim):
    with tf.variable_scope('projection') as scope:
        kernel = tf.get_variable('weight', shape=[1, feature_dim, 1, out_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)
        output = tf.squeeze(output, axis=2, name='output')
    return output

def RNN(input, use_peepholes=False):
    with tf.variable_scope('rnn'):
        layers = 2
        n_units = [num_hidden] * layers

        fw = []
        bw = []
        for n in xrange(layers):
            with tf.variable_scope('layer_%d' % n):
                fw_cell = tf.contrib.rnn.LSTMCell(n_units[n], forget_bias=1.0, use_peepholes=use_peepholes)
                bw_cell = tf.contrib.rnn.LSTMCell(n_units[n], forget_bias=1.0, use_peepholes=use_peepholes)
                # fw_cell = tf.contrib.rnn.DropoutWrapper(cell=fw_cell, output_keep_prob=keep_prob[0])
                # bw_cell = tf.contrib.rnn.DropoutWrapper(cell=bw_cell, output_keep_prob=keep_prob[0])
                fw.append(fw_cell)
                bw.append(bw_cell)

        outputs, state_fw, state_bw = \
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw, bw, input, dtype='float32')
    expand = tf.expand_dims(outputs, axis =-1)
    print(expand.shape)
    hypothesis = frame_wise_projection(expand, 2*num_hidden , num_output)
    print(hypothesis.shape)
    # hypothesis =tf.matmul(outputs, weights['out']) + biases['out']
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    '''
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    '''
    return hypothesis, cost, train_op, tf.train.Saver(max_to_keep=1)

hypothesis, cost, train_op, saver = RNN(X)
init = tf.global_variables_initializer()

# Start training

if args.sessMode == 'train':
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for epoch in range(training_epochs):
            loss_total = []
            for xy_tuple in train_xy:
                train_x = np.asarray(xy_tuple[0])
                train_y = np.asarray(xy_tuple[1])
                data_size = train_x.shape[0]
                total_batch_num = int(math.ceil(data_size / (timesteps *batch_size )))
                for step in range(total_batch_num-1):
                    batch_x = train_x[step*batch_size*timesteps:(step+1)*batch_size*timesteps]
                    batch_y = train_y[step*batch_size*timesteps:(step+1)*batch_size*timesteps]

                    batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                    batch_y = batch_y.reshape((batch_size, timesteps, num_output ))

                    loss , _ = sess.run([cost, train_op], feed_dict={X: batch_x, Y: batch_y})
                    loss_total.append(loss)

            print("Epoch " + str(epoch) + ", Epoch Loss= " + \
                "{:.4f}".format(np.mean(loss_total)))

        print("Optimization Finished!")
        saver.save(sess, 'save_temp/save')

#
# elif args.sessMode == 'test':
#     with tf.Session() as sess:
        saver.restore(sess, 'save_temp/save')
        # test
        n_tuple=0
        for xy_tuple in test_xy:
            n_tuple +=1
            print(n_tuple)
            train_x = np.asarray(xy_tuple[0])
            train_y = np.asarray(xy_tuple[1])
            data_size = train_x.shape[0]
            total_batch_num = int(math.ceil(data_size / (timesteps * batch_size)))
            # print(data_size, batch_size, total_batch_num)

            for step in range(total_batch_num - 1):
                batch_x = train_x[step * batch_size * timesteps:(step + 1) * batch_size * timesteps]
                batch_y = train_y[step * batch_size * timesteps:(step + 1) * batch_size * timesteps]

                # print(batch_x.shape, batch_y.shape)
                batch_x = batch_x.reshape((batch_size, timesteps, num_input))

                prediction = sess.run(hypothesis, feed_dict={X: batch_x})
                prediction = prediction.reshape((-1, num_output))
                plt.figure(figsize=(10,7))
                plt.subplot(211)
                plt.plot(batch_y)
                plt.subplot(212)
                plt.plot(prediction)
                plt.savefig('images/piece{:d},seg{:d}.png'.format(n_tuple, step))

        # Calculate accuracy for 128 mnist test images
        # test_len = 128
        # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        # test_label = mnist.test.labels[:test_len]
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

else:
    with tf.Session() as sess:
        saver.restore(sess, 'save_temp/save')
        #load test piece
        path_name = args.testPath
        xml_name = path_name + 'xml.xml'
        midi_name = path_name + 'midi.mid'
        xml_object = MusicXMLDocument(xml_name)
        xml_notes = xml_matching.extract_notes(xml_object, melody_only=True)
        midi_file = midi_utils.to_midi_zero(midi_name)
        midi_notes = midi_file.instruments[0].notes
        match_list = xml_matching.matchXMLtoMIDI(xml_notes, midi_notes)
        score_pairs = xml_matching.make_xml_midi_pair(xml_notes, midi_notes, match_list)
        measure_positions = xml_matching.extract_measure_position(xml_object)
        features = xml_matching.extract_perform_features(xml_notes, score_pairs, measure_positions)

        test_x = []
        for feat in features:
            if not feat['pitch_interval'] == None:
                test_x.append([  feat['pitch_interval'],feat['duration_ratio'],feat['beat_position'] ] )
            else:
                test_x.append( [0, 0, feat['beat_position'] ])

        test_x = np.asarray(test_x)



        # test
        timestep_quantize_num = int(math.ceil(test_x.shape[0] / timesteps))
        padding_size = timestep_quantize_num * timesteps - test_x.shape[0]
        print(test_x.shape)
        test_x_padded = np.pad(test_x, ((0,padding_size), (0,0)), 'constant')
        print(test_x_padded.shape)
        # print(data_size, batch_size, total_batch_num)

            # print(batch_x.shape, batch_y.shape)
        batch_x = test_x_padded.reshape((-1, timesteps, num_input))

        prediction = sess.run(hypothesis, feed_dict={X: batch_x})
        prediction = prediction.reshape((-1, num_output))
        print(prediction.shape)
        prediction = np.delete(prediction, range(prediction.shape[0]-padding_size,prediction.shape[0]), 0   )
        print(prediction.shape)

        prediction[:,0] *= stds[1][0]
        prediction[:,1] *= stds[1][1]
        prediction[:,2] *= stds[1][2]

        prediction[:,0] += means[1][0]
        prediction[:,1] += means[1][1]
        prediction[:,2] += means[1][2]

        prediction = np.transpose(prediction)
        new_midi = xml_matching.applyIOI(xml_notes, midi_notes, features, prediction)

        xml_matching.save_midi_notes_as_piano_midi(new_midi, path_name + 'performed_by_nn.mid')