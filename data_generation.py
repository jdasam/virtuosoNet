from __future__ import division
import pickle
import random
import xml_matching
import math

def save_features_as_vector(dataset, save_name):
    complete_xy = []
    num_total_datapoint = 0
    total_notes = 0
    for piece in dataset:
        for perform in piece:
            train_x = []
            train_y = []
            for feature in perform:
                total_notes += 1
                if not feature['IOI_ratio'] == None:
                    print(feature['grace_order'])
                    train_x.append(
                        [feature['pitch'], feature['pitch_interval'], feature['duration'],
                         feature['duration_ratio'], feature['beat_position'], feature['voice'],
                        feature['xml_position'], feature['grace_order']] + feature['tempo'] + feature['dynamic'] + feature['notation'])
                    # train_x.append( [ feature['pitch_interval'],feature['duration_ratio'] ] )
                    train_y.append([feature['IOI_ratio'], feature['articulation'], feature['loudness'],
                                    feature['xml_deviation'], feature['pedal_at_start'], feature['pedal_at_end'],
                                    feature['soft_pedal'], feature['pedal_refresh_time'], feature['pedal_cut_time'],
                                    feature['pedal_refresh'], feature['pedal_cut']])
                    num_total_datapoint += 1
            # windowed_train_x = make_windowed_data(train_x, input_length )
            complete_xy.append([train_x, train_y])
    print('total data point is ', num_total_datapoint)
    print(total_notes)
    num_input = len(train_x[0])
    num_output = len(train_y[0])

    def get_mean_and_sd(performances, target_data, target_dimension):
        sum = 0
        squared_sum = 0
        count = 0
        for perf in performances:
            samples = perf[target_data]
            for sample in samples:
                value = sample[target_dimension]
                sum += value
                squared_sum += value * value
                count += 1
        data_mean = sum / count
        data_std = (squared_sum / count - data_mean ** 2) ** 0.5
        return data_mean, data_std

    complete_xy_normalized = []
    means = [[], []]
    stds = [[], []]
    num_normalize_feature = [6, 7]
    for i1 in (0, 1):
        for i2 in range(num_normalize_feature[i1]):
            mean_value, std_value = get_mean_and_sd(complete_xy, i1, i2)
            means[i1].append(mean_value)
            stds[i1].append(std_value)

    for performance in complete_xy:
        complete_xy_normalized.append([])
        for index1 in (0, 1):
            complete_xy_normalized[-1].append([])
            for sample in performance[index1]:
                new_sample = []
                for index2 in range(num_normalize_feature[index1]):
                    new_sample.append((sample[index2] - means[index1][index2]) / stds[index1][index2])
                if index1 == 0:
                    new_sample[num_normalize_feature[index1]:num_input] = sample[num_normalize_feature[index1]:num_input]
                else:
                    new_sample[num_normalize_feature[index1]:num_output] = sample[
                                                                          num_normalize_feature[index1]:num_output]
                complete_xy_normalized[-1][index1].append(new_sample)

    complete_xy_orig = complete_xy
    print(len(complete_xy), len(complete_xy))
    complete_xy = complete_xy_normalized
    random.shuffle(complete_xy)

    with open(save_name + ".dat", "wb") as f:
        pickle.dump(complete_xy, f, protocol=2)
    with open(save_name + "_stat.dat", "wb") as f:
        pickle.dump([means, stds], f, protocol=2)



chopin_pairs = xml_matching.load_entire_subfolder('chopin_cleaned/')
save_features_as_vector(chopin_pairs, 'chopin_cleaned_grace')
