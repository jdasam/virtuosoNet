import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cal_tempo_and_velocity_by_beat(features):
    tempos = []
    velocities = []
    prev_beat = 0

    tempo_saved = 0
    num_added = 0
    max_velocity = 0
    velocity_saved = 0
    momentum = 0.8
    for feat in features:
        if feat.qpm is None:
            continue
        cur_beat = feat.beat_index
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


        tempo_saved += 10 ** feat.qpm
        velocity_saved += feat.velocity
        num_added += 1
        max_velocity = max(max_velocity, feat.velocity)
        prev_beat = cur_beat

    if num_added > 0:
        tempo = tempo_saved / num_added
        tempos.append(tempo)
        velocities.append(max_velocity)

    return tempos, velocities



def plot_performance_worm(features, save_name='images/performance_worm.png'):
    tempos, velocities = cal_tempo_and_velocity_by_beat(features)
    # data_points = []
    # num_data = len(tempos)
    #
    # for i in range(num_data):
    #     data = [tempos[i], velocities[i]]
    #     data_points.append(data)
    # plot data
    num_beat = len(tempos)
    plt.figure(figsize=(10, 7))
    for i in range(num_beat):
        ratio = i / num_beat
        plt.plot(tempos[i], velocities[i], markersize=(7 + 7*ratio), marker='o', color='green', alpha=0.05+ratio*0.8)
        if i > 0:
            plt.plot(tempos[i-1:i+1], velocities[i-1:i+1], color='green', alpha=0.05+ratio*0.8)
    plt.savefig(save_name)
    plt.close()
