from __future__ import division
import pickle
import random
import xml_matching
import copy


def save_features_as_vector(dataset, save_name):
    num_normalize_feature = [8, 11, 11]
    complete_xy = []
    num_total_datapoint = 0
    total_notes = 0
    for piece in dataset:
        for perform in piece:
            train_x = []
            train_y = []
            previous_y = []
            is_beat_list = []
            prev_feat = [0] * num_normalize_feature[1]
            for feature in perform:
                total_notes += 1
                if not feature.qpm == None:
                    train_x.append(
                        [feature.pitch, feature.duration,
                         feature.duration_ratio, feature.beat_position, feature.measure_length, feature.voice,
                        feature.qpm_primo, feature.following_rest,
                        feature.xml_position, feature.grace_order, feature.time_sig_num, feature.time_sig_den]
                        +  feature.pitch_interval + feature.tempo + feature.dynamic + feature.notation + feature.tempo_primo)
                    # train_x.append( [ feature['pitch_interval'],feature['duration_ratio'] ] )
                    # train_y.append([feature['IOI_ratio'], feature['articulation'], feature['loudness'],
                    temp_y = [feature.qpm, feature.articulation, feature.velocity,
                              feature.xml_deviation, feature.pedal_refresh_time, feature.pedal_cut_time,
                              feature.pedal_at_start, feature.pedal_at_end, feature.soft_pedal,
                              feature.pedal_refresh, feature.pedal_cut]
                    # temp_y = [feature.passed_second, feature.duration_second, feature.velocity,
                    #           feature.pedal_refresh_time, feature.pedal_cut_time,
                    #           feature.pedal_at_start, feature.pedal_at_end, feature.soft_pedal,
                    #           feature.pedal_refresh, feature.pedal_cut]
                    train_y.append(temp_y)
                    prev_feat[0] = feature.previous_tempo
                    previous_y.append(prev_feat)
                    prev_feat = copy.copy(temp_y)
                    num_total_datapoint += 1
                    is_beat_list.append(feature.is_beat)
            # windowed_train_x = make_windowed_data(train_x, input_length )
            complete_xy.append([train_x, train_y, previous_y, is_beat_list])
            key_changed_num = []
            for i in range(3):
                key_change = 0
                while key_change == 0 or key_change in key_changed_num:
                    key_change = random.randrange(-5, 7)
                train_x_aug = key_augmentation(train_x, key_change)
                complete_xy.append([train_x_aug, train_y, previous_y, is_beat_list])
                key_changed_num.append(key_change)
    print('total data point is ', num_total_datapoint)
    print(total_notes)
    num_input = len(train_x[0])
    num_output = len(train_y[0])

    print(train_x[0])
    print(train_y[0])
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
    means = [[], [], [], []]
    stds = [[], [], [], []]
    for i1 in (0, 1):
        for i2 in range(num_normalize_feature[i1]):
            mean_value, std_value = get_mean_and_sd(complete_xy, i1, i2)
            means[i1].append(mean_value)
            stds[i1].append(std_value)
    print (means)
    print(stds)
    means[2] = means[1]
    stds[2] = stds[1]

    for performance in complete_xy:
        complete_xy_normalized.append([])
        for index1 in (0, 1, 2):
            complete_xy_normalized[-1].append([])
            for sample in performance[index1]:
                new_sample = []
                for index2 in range(num_normalize_feature[index1]):
                    if not stds[index1][index2] ==0:
                        new_sample.append((sample[index2] - means[index1][index2]) / stds[index1][index2])
                    else:
                        new_sample.append(0)
                if index1 == 0:
                    new_sample[num_normalize_feature[index1]:num_input] = sample[num_normalize_feature[index1]:num_input]
                else:
                    new_sample[num_normalize_feature[index1]:num_output] = sample[
                                                                          num_normalize_feature[index1]:num_output]
                complete_xy_normalized[-1][index1].append(new_sample)
        complete_xy_normalized[-1].append(performance[3])
    complete_xy_orig = complete_xy
    complete_xy = complete_xy_normalized
    random.shuffle(complete_xy)

    for index1 in (0,1):
        for index2 in range(len(stds[index1])):
            std = stds[index1][index2]
            if std == 0:
                print('STD of ' + str(index1) + ',' + str(index2) + ' is zero')


    with open(save_name + ".dat", "wb") as f:
        pickle.dump(complete_xy, f, protocol=2)
    with open(save_name + "_stat.dat", "wb") as f:
        pickle.dump([means, stds], f, protocol=2)

    print(num_input, num_output)


def key_augmentation(data_x, key_change):
    # key_change = 0
    data_x_aug = copy.deepcopy(data_x)
    # while key_change == 0:
    #     key_change = random.randrange(-5, 7)
    for data in data_x_aug:
        data[0] = data[0]+key_change
    return data_x_aug

chopin_pairs = xml_matching.load_entire_subfolder('chopin_cleaned/Chopin_Etude_op_10/1/')
save_features_as_vector(chopin_pairs, 'vectorized_interval_small')
