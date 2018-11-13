import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_name = 'tempo_primo'
feature_index = [0, 0]
save_name = 'feature_histogram.png'


with open(data_name + ".dat", "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    # p = u.load()
    # complete_x()y = pickle.load(f)
    complete_xy = u.load()

index1 = [0, 1]
index2 = [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7,8,9,10]]


for i in index1:
    for j in index2[i]:
        target_features = []
        save_name = 'feature_' + str(i) + '_' + str(j) + '.png'
        for perf in complete_xy:
            features = perf[i]
            for feat in features:
                target_features.append(feat[j])


        plt.figure(figsize=(10, 7))
        n, bins, patches = plt.hist(x=target_features, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)

        plt.savefig(save_name)
        plt.close()
