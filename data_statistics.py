import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_name = 'sigmoid_pedal'
feature_index = [0, 0]
save_name = 'feature_histogram.png'


with open(data_name + ".dat", "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    # p = u.load()
    # complete_x()y = pickle.load(f)
    complete_xy = u.load()

target_features = []
for perf in complete_xy:
    features = perf[feature_index[0]]
    for feat in features:
        target_features.append(feat[feature_index[1]])


plt.figure(figsize=(10, 7))

n, bins, patches = plt.hist(x=target_features, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)

plt.savefig(save_name)
plt.close()
