import pyScoreParser.xml_matching as xml_matching
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import pickle
import os
import copy


def get_features_mean_and_std_of_piece(path):
    perform_data = xml_matching.load_pairs_from_folder(path)
    features = [x['features'] for x in perform_data]
    composer_vec = perform_data[0]['composer']

    total_y = []
    for f in features:
        _, perf_y = xml_matching.convert_features_to_vector(f, composer_vec)
        # total_x.append(perf_x)
        total_y.append(perf_y)

    mean = np.mean(total_y, axis=0)
    std = np.std(total_y, axis =0)

    return total_y, mean, std


def draw_mean_std(means, stds, save_name='feature_test.png'):
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(means)), [x[1] for x in means])

    plt.savefig('means.png')
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.plot(range(len(stds)), [x[1] for x in stds])
    plt.savefig('stds.png')
    plt.close()


def scale_perform_features_by_stats(features, global_scale):
    l = len(features)
    target_feature_end_idx = 14
    target_features = np.asarray([x[:target_feature_end_idx] for x in features])
    target_scale = (global_scale[0][:target_feature_end_idx], global_scale[1][:target_feature_end_idx])
    means = np.broadcast_to(target_scale[0], (l, target_feature_end_idx))
    stds = np.broadcast_to(target_scale[1], (l, target_feature_end_idx))
    return (target_features - means) / stds

def estimate_loss_by_mean_of_performances(perform_features, global_scale):
    # global_scale = (global means, global stds)
    mean_features = np.mean(perform_features, axis=0)
    mean_features = scale_perform_features_by_stats(mean_features, global_scale)
    perform_features = [scale_perform_features_by_stats(x, global_scale) for x in perform_features]
    target_feature_indices = [0, 1]

    for perform in perform_features:
        for i in target_feature_indices:
            squared_error = np.mean((perform[:,i] - mean_features[:,i]) ** 2)
            print('L2 loss of ', i, ' between target and mean: ', squared_error)

    for i in target_feature_indices:
        max_perf_diff = 0
        for j in range(len(perform_features)):
            for k in range(j+1, len(perform_features)):
                squared_error = np.mean((perform_features[j][:,i] - perform_features[k][:,i]) ** 2)
                max_perf_diff = max(max_perf_diff, squared_error)

        print(max_perf_diff)


def load_data_stats(path):
    with open(path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        means, stds = u.load()

        return means, stds


def draw_piano_roll_by_feature(notes, features, save_name='piano_roll.png'):
    fig = plt.figure(figsize=(100, 7))
    ax = fig.add_subplot(111, aspect='auto')
    cmap = cm.get_cmap('YlOrRd')

    note_height = 1 / 88
    total_length = notes[-1].note_duration.xml_position + notes[-1].note_duration.duration
    max_feature_value = max(features)
    for i, note in enumerate(notes):
        note_start = note.note_duration.xml_position / total_length
        note_duration = note.note_duration.duration / total_length
        midi_pitch = (note.pitch[1] - 20) / 88
        color_value = cmap(features[i] / max_feature_value)
        note_rectangle = patches.Rectangle((note_start, midi_pitch), note_duration, note_height, color=color_value)
        ax.add_patch(note_rectangle)


    plt.savefig(save_name)
    plt.close()


def get_entire_sub_folder(path):
    midi_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                 f == 'midi_cleaned.mid']
    folder_list = []
    for midifile in midi_list:
        foldername = os.path.split(midifile)[0] + '/'
        folder_list.append(foldername)

    return folder_list


def draw_std_of_all_piece_in_path(path):
    folder_list = get_entire_sub_folder(path)
    # data_mean, data_stds = load_data_stats('icml_grace_stat.dat')
    # estimate_loss_by_mean_of_performances(features, (data_mean[1], data_stds[1]))
    feat_name = ['temp', 'vel', 'dev']
    for folder in folder_list:
        features, _, stds = get_features_mean_and_std_of_piece(folder)
        if len(features) > 4:
            for i in range(3):
                target_stds = [x[i] for x in stds]
                _, notes = xml_matching.read_xml_to_notes(folder)

                path_split = copy.copy(folder).split('/')
                dataset_folder_name_index = path_split.index('chopin_cleaned')
                piece_name = '_'.join(copy.copy(folder).split('/')[dataset_folder_name_index + 1:]) + feat_name[i] +'.png'
                draw_piano_roll_by_feature(notes, target_stds, piece_name)


def make_mean_performance_midi(path):
    multiple_features, means, stds = get_features_mean_and_std_of_piece(path)
    xml_doc, xml_notes = xml_matching.read_xml_to_notes(path)
    features = xml_matching.model_prediction_to_feature(means)
    xml_notes = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, features, start_time=1, predicted=True)

    output_midi, midi_pedals = xml_matching.xml_notes_to_midi(xml_notes)
    piece_name = path.split('/')
    save_name = 'test_result/' + piece_name[-2] + '_by_mean'

    xml_matching.save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_name + '.mid')

# features, means, stds = get_features_mean_and_std_of_piece('pyScoreParser/chopin_cleaned/Chopin/Etudes_op_10/5/')
# # draw_mean_std(means, stds)
# # print(stds)
#
#
# target_stds = [x[1] for x in stds]
# _, notes = xml_matching.read_xml_to_notes('pyScoreParser/chopin_cleaned/Chopin/Etudes_op_10/5/')
# draw_piano_roll_by_feature(notes, target_stds)

# draw_std_of_all_piece_in_path('pyScoreParser/chopin_cleaned/Beethoven/')

make_mean_performance_midi('pyScoreParser/chopin_cleaned/Chopin/Ballades/1/')