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
import os


parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test")
# parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./mxp/testdata/chopin10-3/", help="folder path of test mat")
# parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
parser.add_argument("-data", "--dataName", type=str, default="chopin_cleaned_small", help="dat file name")
args = parser.parse_args()

# Training Parameters
learning_rate = 0.001
# training_steps = 10000
training_epochs = 40
batch_size = 4
display_step = 200
training_ratio = 0.8

# Network Parameters
num_input = 8+40+7  #
timesteps = 200 # timesteps
num_hidden = 64 # hidden layer num of features
num_output = 4 + 7 # ioi, articulation, loudness, onset_deviation, pedals(7)




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
        for n in range(layers):
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
    sigmoid_layer = tf.sigmoid(hypothesis)
    print(sigmoid_layer.shape)

    combined_hypothesis = tf.concat([hypothesis[:,:,0:7], sigmoid_layer[:,:,7:12]], axis=2)
    print(combined_hypothesis.shape)
    # hypothesis =tf.matmul(outputs, weights['out']) + biases['out']
    # cost = tf.reduce_mean(tf.square(hypothesis - Y))
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
    with open(args.dataName+".dat", "rb") as f:
        complete_xy = pickle.load(f)
    perform_num = len(complete_xy)

    train_perf_num = int(perform_num * training_ratio)
    train_xy = complete_xy[:train_perf_num]
    test_xy = complete_xy[train_perf_num:]
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        best_valid_loss = float("inf")
        get_worse_count = 0
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
                    # print(batch_x.shape, batch_y.shape)
                    loss , _, hypoth = sess.run([cost, train_op, hypothesis], feed_dict={X: batch_x, Y: batch_y})
                    loss_total.append(loss)

            print("Epoch " + str(epoch) + ", Epoch Loss= " + \
                "{:.4f}".format(np.mean(loss_total)))

            valid_loss_total = []
            for xy_tuple in test_xy:
                test_x = np.asarray(xy_tuple[0])
                test_y = np.asarray(xy_tuple[1])
                timestep_quantize_num = int(math.ceil(test_x.shape[0] / timesteps))
                padding_size = timestep_quantize_num * timesteps - test_x.shape[0]
                # print(test_x.shape, padding_size)
                test_x_padded = np.pad(test_x, ((0, padding_size), (0, 0)), 'constant')
                test_y_padded = np.pad(test_y, ((0, padding_size), (0, 0)), 'constant')
                # print(test_x_padded.shape)
                # print(data_size, batch_size, total_batch_num)

                # print(batch_x.shape, batch_y.shape)
                batch_x = test_x_padded.reshape((-1, timesteps, num_input))
                batch_y = test_y_padded.reshape((-1, timesteps, num_output))

                valid_loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
                valid_loss_total.append(valid_loss)
            print("Valid Loss= " + \
                  "{:.4f}".format(np.mean(valid_loss_total)))

            if np.mean(valid_loss_total) < best_valid_loss:
                best_valid_loss = np.mean(valid_loss_total)
                get_worse_count = 0
            else:
                get_worse_count += 1

            if get_worse_count >5:
                break

        print("Optimization Finished!")
        saver.save(sess,  args.dataName+'_save_temp/save')

#
# elif args.sessMode == 'test':
#     with tf.Session() as sess:
        saver.restore(sess,  args.dataName+'_save_temp/save')
        # test
        n_tuple=0
        for xy_tuple in test_xy:
            n_tuple +=1
            train_x = np.asarray(xy_tuple[0])
            train_y = np.asarray(xy_tuple[1])
            data_size = train_x.shape[0]
            total_batch_num = int(math.ceil(data_size / (timesteps * batch_size)))

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

# test session
else:
    with open(args.dataName+"_stat.dat", "rb") as f:
        means, stds = pickle.load(f)
        # print(means, stds)
    with tf.Session() as sess:
        saver.restore(sess, args.dataName+'_save_temp/save')
        #load test piece
        path_name = args.testPath
        # xml_name = path_name + 'xml.xml'
        # midi_name = path_name + 'midi.mid'
        xml_name = path_name + 'musicxml_cleaned.musicxml'
        midi_name = path_name + 'midi_cleaned.mid'

        if not os.path.isfile(xml_name):
            xml_name = path_name + 'xml.xml'
            midi_name = path_name + 'midi.mid'


        xml_object = MusicXMLDocument(xml_name)
        xml_notes = xml_matching.extract_notes(xml_object, melody_only=False, grace_note=True)
        directions = xml_matching.extract_directions(xml_object)
        xml_notes = xml_matching.apply_directions_to_notes(xml_notes, directions)
        midi_file = midi_utils.to_midi_zero(midi_name)
        midi_file = midi_utils.add_pedal_inf_to_notes(midi_file)
        midi_notes = midi_file.instruments[0].notes
        match_list = xml_matching.matchXMLtoMIDI(xml_notes, midi_notes)
        score_pairs = xml_matching.make_xml_midi_pair(xml_notes, midi_notes, match_list)
        measure_positions = xml_matching.extract_measure_position(xml_object)
        features = xml_matching.extract_perform_features(xml_notes, score_pairs, measure_positions)

        test_x = []
        for feat in features:
            # if not feat['pitch_interval'] == None:
            test_x.append([ (feat['pitch']-means[0][0])/stds[0][0],  (feat['pitch_interval']-means[0][1])/stds[0][1] ,
                            (feat['duration'] - means[0][2]) / stds[0][2],(feat['duration_ratio']-means[0][3])/stds[0][3],
                            (feat['beat_position']-means[0][4])/stds[0][4], (feat['voice']-means[0][5])/stds[0][5],
                            feat['xml_position'], feat['grace_order']]
                          + feat['tempo'] + feat['dynamic'] + feat['notation'])
            # else:
            #     test_x.append( [(feat['pitch']-means[0][0])/stds[0][0], 0,  (feat['duration'] - means[0][2]) / stds[0][2], 0,
            #                     (feat['beat_position']-means[0][4])/stds[0][4]]
            #                    + feat['tempo'] + feat['dynamic'] + feat['notation'] )

        test_x = np.asarray(test_x)
        print('test_x shape is', test_x.shape)


        # test
        timestep_quantize_num = int(math.ceil(test_x.shape[0] / timesteps))
        padding_size = timestep_quantize_num * timesteps - test_x.shape[0]
        print(test_x.shape, padding_size)
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


        for i in range(11):
            prediction[:,i] *= stds[1][i]
            prediction[:,i] += means[1][i]


        # pred_length = prediction.shape[0]
        output_features= []
        for pred in prediction:
            feat = {'IOI_ratio': pred[0], 'articulation':pred[1], 'loudness':pred[2], 'xml_deviation':pred[3],
                    'pedal_at_start': pred[6], 'pedal_at_end': pred[7], 'soft_pedal': pred[8],
                    'pedal_refresh_time': pred[4], 'pedal_cut_time': pred[5], 'pedal_refresh': pred[9],
                    'pedal_cut': pred[10] }
            output_features.append(feat)
        # prediction = np.transpose(prediction)
        # feature['pedal_at_start'] = pairs[i]['midi'].pedal_at_start
        # feature['pedal_at_end'] = pairs[i]['midi'].pedal_at_end
        # feature['pedal_refresh'] = int(pairs[i]['midi'].pedal_refresh)
        # feature['pedal_refresh_time'] = int(pairs[i]['midi'].pedal_refresh_time)
        # feature['pedal_cut'] = int(pairs[i]['midi'].pedal_cut)
        # feature['pedal_cut_time'] = int(pairs[i]['midi'].pedal_cut)
        # feature['soft_pedal'] = pairs[i]['midi'].soft_pedal

        output_xml = xml_matching.apply_perform_features(xml_notes, output_features)
        output_midi = xml_matching.xml_notes_to_midi(output_xml)

        # new_midi = xml_matching.applyIOI(xml_notes, midi_notes, features, prediction)

        xml_matching.save_midi_notes_as_piano_midi(output_midi, path_name + 'performed_by_nn.mid', bool_pedal=True, disklavier=True)