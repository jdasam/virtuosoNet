from __future__ import division
import pickle
import random
import copy
import pandas
import numpy as np
import argparse
import os

from . import xml_matching as xml_matching
from . import dataset_split as split

parser = argparse.ArgumentParser()
parser.add_argument('--regression', default=True, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()


NUM_TRILL_PARAM = 5
NUM_NORMALIZE_FEATURE = [8, 19, 4]
REGRESSION = args.regression
print('Data type is regression: ', args.regression)

VALID_LIST = split.VALID_LIST
TEST_LIST = split.TEST_LIST

def save_features_as_vector(dataset, num_train, num_valid, save_name):
    complete_xy = []
    num_total_datapoint = 0
    total_notes = 0
    num_piece = 0
    num_perform = 0
    for piece in dataset:
        num_piece += 1
        for perform in piece:
            num_perform +=1
            features = perform['features']
            score = perform['score']
            composer_vec = perform['composer']
            score_graph = perform['graph']

            train_x, train_y = xml_matching.convert_features_to_vector(features, composer_vec)
            align_matched_status = [f.align_matched for f in features]
            pedal_status = [f.articulation_loss_weight for f in features]
            note_locations = [f.note_location for f in features]
            total_notes += len(train_x)

            # windowed_train_x = make_windowed_data(train_x, input_length )
            # complete_xy.append([train_x, train_y, previous_y, beat_numbers, measure_numbers, voice_numbers])
            complete_xy.append([train_x, train_y, note_locations, align_matched_status, pedal_status, score_graph, score])
            # key_changed_num = []
            # for i in range(3):
            #     key_change = 0
            #     while key_change == 0 or key_change in key_changed_num:
            #         key_change = random.randrange(-5, 7)
            #     train_x_aug = key_augmentation(train_x, key_change)
            #     complete_xy.append([train_x_aug, train_y, previous_y, beat_numbers, measure_numbers])
            #     key_changed_num.append(key_change)

    print('Total data point is ', num_total_datapoint)
    print('Number of total piece is ', num_piece, ' and total performance is ', num_perform)
    print('Number of training perform is ', num_train, ' number of valid perform is', num_valid, ' and test performance is ', len(complete_xy) - num_train - num_valid)

    print(total_notes)
    num_input = len(train_x[0])
    num_output = len(train_y[0])

    print(train_x[0])
    print(train_y[0])

    if REGRESSION:
        complete_xy_normalized, means, stds = normalize_features(complete_xy, num_input, num_output, x_only=False)
        complete_xy_orig = complete_xy
        complete_xy = complete_xy_normalized
    else:
        complete_xy_normalized, means, stds = normalize_features(complete_xy, num_input, num_output, x_only=True)
        complete_xy_orig = complete_xy
        complete_xy = complete_xy_normalized
        complete_xy, bins = output_to_categorical(complete_xy)

    complete_xy_train = complete_xy[0:num_train]
    complete_xy_valid = complete_xy[num_train:num_train+num_valid]
    complete_xy_test = complete_xy[num_train+num_valid:]
    random.shuffle(complete_xy_train)
    # random.shuffle(complete_xy_valid)
    # random.shuffle(complete_xy_test)

    for index1 in (0,1):
        for index2 in range(len(stds[index1])):
            std = stds[index1][index2]
            if std == 0:
                print('STD of ' + str(index1) + ',' + str(index2) + ' is zero')

    with open(save_name + ".dat", "wb") as f:
        pickle.dump({'train': complete_xy_train, 'valid': complete_xy_valid}, f, protocol=2)
    with open(save_name + "_test.dat", "wb") as f:
        pickle.dump(complete_xy_test, f, protocol=2)

    if REGRESSION:
        with open(save_name + "_stat.dat", "wb") as f:
            pickle.dump([means, stds], f, protocol=2)
    else:
        with open(save_name + "_stat.dat", "wb") as f:
            pickle.dump([means, stds, bins], f, protocol=2)

    num_output = len(complete_xy[0][1][0])
    print(num_input, num_output)


def get_mean_and_sd(performances, target_data, target_dimension):
    sum = 0
    squared_sum = 0
    count = 0
    for perf in performances:
        samples = perf[target_data]
        for sample in samples:
            value = sample[target_dimension]
            if target_data == 1 and 14 < target_dimension < 19 and value == 0:
                continue
            sum += value
            squared_sum += value * value
            count += 1
    if count != 0:
        data_mean = sum / count
        data_std = (squared_sum / count - data_mean ** 2) ** 0.5
    else:
        data_mean = 0
        data_std = 1
    return data_mean, data_std


def normalize_features(complete_xy, num_input, num_output, x_only=False):
    complete_xy_normalized = []
    means = [[], [], []]
    stds = [[], [], []]
    if x_only:
        index_list = [0]
    else:
        index_list = [0, 1]

    for i1 in index_list:
        for i2 in range(NUM_NORMALIZE_FEATURE[i1]):
            mean_value, std_value = get_mean_and_sd(complete_xy, i1, i2)
            means[i1].append(mean_value)
            stds[i1].append(std_value)
    print(means)
    print(stds)

    for performance in complete_xy:
        complete_xy_normalized.append([])
        for index1 in index_list:
            complete_xy_normalized[-1].append([])
            for sample in performance[index1]:
                new_sample = []
                for index2 in range(NUM_NORMALIZE_FEATURE[index1]):
                    if not (stds[index1][index2] == 0 or isinstance(stds[index1][index2], complex)):
                        if index1 == 1 and 14 < index2 < 19 and sample[index2] == 0:
                            new_sample.append(0)
                        else:
                            new_sample.append((sample[index2] - means[index1][index2]) / stds[index1][index2])
                    else:
                        new_sample.append(0)
                if index1 == 0:
                    new_sample[NUM_NORMALIZE_FEATURE[index1]:num_input] = sample[
                                                                          NUM_NORMALIZE_FEATURE[index1]:num_input]
                else:
                    new_sample[NUM_NORMALIZE_FEATURE[index1]:num_output] = sample[
                                                                           NUM_NORMALIZE_FEATURE[index1]:num_output]
                complete_xy_normalized[-1][index1].append(new_sample)
        if x_only:
            complete_xy_normalized[-1].append(performance[1])

        complete_xy_normalized[-1].append(performance[2])
        complete_xy_normalized[-1].append(performance[3])
        complete_xy_normalized[-1].append(performance[4])
        complete_xy_normalized[-1].append(performance[5])

    return complete_xy_normalized, means, stds


def output_to_categorical(complete_xy):
    num_bins_by_feature = [100, 20, 20, 10, 10, 10]
    pedal_threshold = [-1, 30, 60, 128]
    xy_in_categorical = []
    entire_y = [xy[1] for xy in complete_xy]
    num_notes_of_perf = []
    entire_y_flattened = []

    bins = []

    for perf in entire_y:
        num_notes = len(perf)
        num_notes_of_perf.append(num_notes)
        if entire_y_flattened == []:
            entire_y_flattened = perf
        else:
            entire_y_flattened += perf

    y_as_mat = np.asarray(entire_y_flattened)
    trill_bool = y_as_mat[:,11] != 0

    for i in range(6):
        y_as_mat[:,i], temp_bin = pandas.qcut(y_as_mat[:,i], num_bins_by_feature[i], labels=False, retbins=True, duplicates='drop')
        bins.append(temp_bin)

    for i in range(6,11):
        y_as_mat[:, i] = pandas.cut(y_as_mat[:, i], pedal_threshold, labels=False)
        bins.append(pedal_threshold)

    for i in range(11,15):
        y_as_mat[trill_bool, i], temp_bin = pandas.qcut(y_as_mat[trill_bool, i], 5, labels=False, retbins=True, duplicates='drop')
        bins.append(temp_bin)

    num_perf = len(complete_xy)
    notes_range_index = 0
    for i in range(num_perf):
        num_notes = num_notes_of_perf[i]
        complete_xy[i][1] = y_as_mat[notes_range_index:notes_range_index+num_notes,:]
        notes_range_index += num_notes

    return complete_xy, bins


def key_augmentation(data_x, key_change):
    # key_change = 0
    data_x_aug = copy.deepcopy(data_x)
    pitch_start_index = 13
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


def load_entire_subfolder(path, minimum_perform_limit=0):
    entire_pairs = []
    num_train_pairs = 0
    num_valid_pairs = 0
    num_test_pairs = 0

    midi_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
              f == 'midi_cleaned.mid']
    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        skip = False
        for valid_piece in VALID_LIST:
            if valid_piece in foldername:
                skip = True
                break
        for test_piece in TEST_LIST:
            if test_piece in foldername:
                skip = True
                break
        if not skip:
            xml_name = foldername + 'musicxml_cleaned.musicxml'
            if os.path.isfile(xml_name):
                print(foldername)
                piece_pairs = xml_matching.load_pairs_from_folder(foldername)
                if piece_pairs is not None and len(piece_pairs) > minimum_perform_limit:
                    entire_pairs.append(piece_pairs)
                    num_train_pairs += len(piece_pairs)

    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        for valid_piece in VALID_LIST:
            if valid_piece in foldername:
                xml_name = foldername + 'musicxml_cleaned.musicxml'

                if os.path.isfile(xml_name):
                    print(foldername)
                    piece_pairs = xml_matching.load_pairs_from_folder(foldername)
                    if piece_pairs is not None and len(piece_pairs) > minimum_perform_limit:
                        entire_pairs.append(piece_pairs)
                        num_valid_pairs += len(piece_pairs)
                        print('num valid pairs', num_valid_pairs)

    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        for test_piece in TEST_LIST:
            if test_piece in foldername:
                xml_name = foldername + 'musicxml_cleaned.musicxml'

                if os.path.isfile(xml_name):
                    print(foldername)
                    piece_pairs = xml_matching.load_pairs_from_folder(foldername)
                    if piece_pairs is not None and len(piece_pairs) > minimum_perform_limit:
                        entire_pairs.append(piece_pairs)
                        num_test_pairs += len(piece_pairs)

    print('Number of train pairs: ', num_train_pairs, 'valid pairs: ', num_valid_pairs, 'test pairs: ', num_test_pairs)
    # print('Number of total score notes, performance notes, non matched notes, excluded notes: ', NUM_SCORE_NOTES, NUM_PERF_NOTES, NUM_NON_MATCHED_NOTES, NUM_EXCLUDED_NOTES)
    return entire_pairs, num_train_pairs, num_valid_pairs, num_test_pairs




# xml_matching.check_data_split('chopin_cleaned/')
chopin_pairs, num_train_pairs, num_valid_pairs, num_test_pairs = load_entire_subfolder('pyScoreParser/chopin_cleaned/', 4)
save_features_as_vector(chopin_pairs, num_train_pairs, num_valid_pairs, 'perform_style_set_5')