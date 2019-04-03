import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_name = 'hierarchy_set'
feature_index = [0, 0]
save_name = 'feature_histogram.png'

print('Loading the data...')
with open(data_name + ".dat", "rb") as f:
    u = pickle._Unpickler   (f)
    u.encoding = 'latin1'
    # p = u.load()
    # complete_x()y = pickle.load(f)
    complete_xy = u.load()

# index1 = [0, 1]
# index2 = [[0,1,2,3,4,5,6,7,8,9,10,11], [0,1,2,3,4,5,6,7,8,9,10]]
index1 = [2]
index2 = [[0,1,2,3]]


def plot_and_save_figures():
    for i in range(len(index1)):
        i1 = index1[i]
        for j in index2[i]:
            print('index is ', i1, j)
            target_features = []
            save_name = 'feature_' + str(i1) + '_' + str(j) + '.png'
            for key in ('train', 'valid'):
                performs = complete_xy[key]
                for perf in performs:
                    features = perf[i1]
                    for feat in features:
                        target_features.append(feat[j])


            plt.figure(figsize=(10, 7))
            n, bins, patches = plt.hist(x=target_features, bins='sturges', color='#0504aa',
                                        alpha=0.7, rwidth=0.85)

            plt.savefig(save_name)
            plt.close()

def calculate_ratio_of_feature_true(indexes):
    for index in indexes:
        idx0 = index[0]
        idx1 = index[1]
        for key in ('train', 'valid'):
            num_feature_true = 0
            num_feature_false = 0
            performs = complete_xy[key]
            for perf in performs:
                features = perf[idx0]
                for feat in features:
                    if feat[idx1]:
                        num_feature_true += 1
                    else:
                        num_feature_false +=1

            print('Index[{}, {}], {} set - Number of True notes: {}, Number of False notes: {}, True ratio: {:.2f}%'.format(
                idx0, idx1, key, num_feature_true, num_feature_false, num_feature_true/(num_feature_false+num_feature_true)*100
            ))


plot_and_save_figures()
# target_indexes = [(0,12), (0,13), (0,-11), (0,15)]
# calculate_ratio_of_feature_true(target_indexes)