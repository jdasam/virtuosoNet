import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path


def cal_tempo_and_velocity_by_beat(features, note_locations, momentum=0.8):
    tempos = []
    velocities = []
    prev_beat = 0

    tempo_saved = 0
    num_added = 0
    max_velocity = 0
    velocity_saved = 0

    for cur_note_tempo, cur_note_vel, cur_beat in zip( features['beat_tempo'], features['velocity'], note_locations):
        if cur_beat > prev_beat and num_added > 0:
            tempo = tempo_saved / num_added
            velocity = (velocity_saved / num_added + max_velocity) / 2

            if len(tempos)> 0:
                tempo = tempos[-1] * momentum + tempo * (1-momentum)
                velocity = velocities[-1] * momentum + velocity * (1 - momentum)
            tempos.append(tempo)
            velocities.append(velocity)
            tempo_saved = 0
            num_added = 0
            max_velocity = 0
            velocity_saved = 0

        tempo_saved += 10 ** cur_note_tempo
        velocity_saved += cur_note_vel
        num_added += 1
        max_velocity = max(max_velocity, cur_note_vel)
        prev_beat = cur_beat

    if num_added > 0:
        tempo = tempo_saved / num_added
        tempos.append(tempo)
        velocities.append(max_velocity)
    # for i in range(num_notes):
    #     feat = features[i]
    #     if note_locations:
    #         cur_note_tempo = feat[0]
    #         cur_note_vel = feat[1]
    #         cur_beat = note_locations[i].beat
    #     else:
    #         if feat.qpm is None:
    #             continue
    #         else:
    #             cur_note_tempo = feat.qpm
    #             cur_note_vel = feat.velocity
    #         cur_beat = feat.note_location.beat
    #     if cur_beat > prev_beat and num_added > 0:
    #         tempo = tempo_saved / num_added
    #         velocity = (velocity_saved / num_added + max_velocity) / 2

    #         if len(tempos)> 0:
    #             tempo = tempos[-1] * momentum + tempo * (1-momentum)
    #             velocity = velocities[-1] * momentum + velocity * (1 - momentum)
    #         tempos.append(tempo)
    #         velocities.append(velocity)
    #         tempo_saved = 0
    #         num_added = 0
    #         max_velocity = 0
    #         velocity_saved = 0

    #     tempo_saved += 10 ** cur_note_tempo
    #     velocity_saved += cur_note_vel
    #     num_added += 1
    #     max_velocity = max(max_velocity, cur_note_vel)
    #     prev_beat = cur_beat

    # if num_added > 0:
    #     tempo = tempo_saved / num_added
    #     tempos.append(tempo)
    #     velocities.append(max_velocity)

    return tempos, velocities



def plot_performance_worm(features, note_locations, save_name='images/performance_worm.png', save_csv=False, attention_weights=None):
    tempos, velocities = cal_tempo_and_velocity_by_beat(features, note_locations)
    if save_csv:
        if isinstance(save_name, str):
            save_name = Path(save_name)
        csv_path = save_name.parent / (save_name.stem + '_perf_worm.csv')
        beat_change_idx = [0] + np.where(np.diff(note_locations)==1)[0].tolist()
        with open(csv_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            if attention_weights is not None:
                writer.writerow(["note_idx", "tempo", "velocity", "att_0","att_1","att_2","att_3","att_4","att_5","att_6", "att_7"])
                [writer.writerow([i, t,v] + a) for i, t, v, a in zip(beat_change_idx, tempos, velocities, attention_weights)]
            else:
                writer.writerow(["note_idx", "tempo", "velocity"])
                [writer.writerow([i, t,v]) for i, t, v, a in zip(beat_change_idx, tempos, velocities)]

    num_beat = len(tempos)
    plt.figure(figsize=(10, 7))
    color = 'green'
    for i in range(num_beat):
        ratio = i / num_beat
        alpha = 0.05+ratio*0.8
        plt.plot(tempos[i], velocities[i], markersize=(7 + 7*ratio), marker='o', color=color, alpha=alpha)
        if i > 0:
            plt.plot(tempos[i-1:i+1], velocities[i-1:i+1], color=color, alpha=alpha)
    plt.savefig(save_name)
    plt.close()


def plot_normalized_feature(features_list, save_name='feature_test.png'):
    plt.figure(figsize=(12, 7))
    num_beat = len(features_list[0])

    for features in features_list:
        features = np.asarray(features)
        normalized_features = features / np.mean(features)
        # for i in range(1,num_beat):
            # feat = normalized_features[i]
        plt.plot(range(num_beat), normalized_features)

    plt.savefig(save_name)
    plt.close()


def plot_human_model_features_compare(features_list, save_name='feature_test.png'):
    plt.figure(figsize=(12, 7))
    num_beat = len(features_list[0])
    num_performance = len(features_list)

    for i in range(num_performance-1):
        features = features_list[i]
        features = np.asarray(features)
        normalized_features = features / np.mean(features)
        # for i in range(1,num_beat):
        # feat = normalized_features[i]
        plt.plot(range(num_beat), normalized_features, color='gray')

    model_features = np.asarray(features_list[-1])
    normalized_features = model_features / np.mean(model_features)
    plt.plot(range(num_beat), normalized_features, color='red')

    plt.savefig(save_name)
    plt.close()


def plot_model_features_compare(features_list, num_models=4, save_name='feature_test.png'):
    plt.figure(figsize=(12, 6))
    matplotlib.rcParams.update({'font.size': 20})

    plt.xlabel('Beat Index')
    plt.ylabel('Relative Tempo')
    num_beat = len(features_list[0])
    num_performance = len(features_list)
    code = ('ISGN', 'BL', 'HAN', 'G-HAN')

    for i in range(num_performance-num_models):
        features = features_list[i]
        features = np.asarray(features)
        normalized_features = features / np.mean(features)
        # for i in range(1,num_beat):
        # feat = normalized_features[i]
        if i == 0:
            plt.plot(range(num_beat), normalized_features, color='gray', label='Human')
        else:
            plt.plot(range(num_beat), normalized_features, color='gray')

    for i in range(num_models):
        model_features = np.asarray(features_list[-num_models + i])
        normalized_features = model_features / np.mean(model_features)
        plt.plot(range(num_beat), normalized_features, label=code[i], linewidth=2.0)

    plt.legend()

    plt.savefig(save_name)
    plt.close()