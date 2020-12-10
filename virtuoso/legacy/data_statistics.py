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


def cal_correlation_of_pairs_in_folder(path):
    features_in_folder = load_pairs_from_folder(path)
    if features_in_folder is None:
        return None
    num_performance = len(features_in_folder)
    if num_performance < 3:
        print('Error: There are only {} performances in the folder'.format(num_performance))
        return None

    num_notes = len(features_in_folder[0]['features'])
    beat_numbers = [x.note_location.beat for x in features_in_folder[0]['features']]
    slice_indexes = make_slicing_indexes_by_beat(beat_numbers, 30, overlap=True)
    correlation_result_total = []

    for slc_idx in slice_indexes:
        correlation_result = CorrelationResult(path, slc_idx)
        correlation_result.num_performance = num_performance

        for features in features_in_folder:
            sliced_features = features['features'][slc_idx[0]:slc_idx[1]]
            tempos, dynamics = perf_worm.cal_tempo_and_velocity_by_beat(sliced_features)
            correlation_result.tempo_features.append(tempos)
            correlation_result.dynamic_features.append(dynamics)
        correlation_result._cal_correlation_of_features()

        min_tempo_r, min_vel_r = correlation_result.cal_minimum()
        if min_tempo_r > 0.7:
            save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_', 10) + '_note{}-{}.png'.format(slc_idx[0], slc_idx[1])
            perf_worm.plot_normalized_feature(correlation_result.tempo_features, save_name)
            correlation_result_total.append(correlation_result)

    return correlation_result_total


class CorrelationResult:
    def __init__(self, path=None, slc_idx=None):
        self.tempo_r = []
        self.dynamic_r = []
        self.path_name = path
        self.slice_index = slc_idx
        self.tempo_features = []
        self.dynamic_features = []
        self.num_performance = 0

    def _append_result(self, tempo_r, velocity_r):
        self.tempo_r.append(tempo_r)
        self.dynamic_r.append(velocity_r)

    def cal_median(self):
        return np.median(self.tempo_r), np.median(self.dynamic_r)

    def cal_minimum(self):
        return min(self.tempo_r), min(self.dynamic_r)

    def cal_maximum(self):
        return max(self.tempo_r), max(self.dynamic_r)

    def _cal_correlation_of_features(self):
        for i in range(self.num_performance-1):
            for j in range(i+1, self.num_performance):
                tempo_r, _ = cal_correlation(self.tempo_features[i], self.tempo_features[j])
                dynamic_r, _ = cal_correlation(self.dynamic_features[i], self.dynamic_features[j])
                self._append_result(tempo_r, dynamic_r)


    def __str__(self):
        if len(self.tempo_r) > 0:
            tempo_r_median, dynamic_r_median = self.cal_median()
            tempo_r_min, dynamic_r_min = self.cal_minimum()
            tempo_r_max, dynamic_r_max = self.cal_maximum()
        else:
            return 'No correlation'

        return 'Piece: {}, Note index : {}, Tempo Med: {:.4f}, Min: {:.4f}, Max: {:.4f} - Dynamic Med: {:.4f}, Min: {:.4f}, Max: {:.4f}'\
            .format(self.path_name, self.slice_index, tempo_r_median, tempo_r_min, tempo_r_max, dynamic_r_median, dynamic_r_min, dynamic_r_max)


def cal_correlation(feat_a, feat_b):
    if len(feat_a) != len(feat_b):
        print('Error: length of two tempos are different, length a: {}, length b: {}'.format(len(feat_a), len(feat_b)))
        return None

    return scipy.stats.pearsonr(feat_a, feat_b)

plot_and_save_figures()
# target_indexes = [(0,12), (0,13), (0,-11), (0,15)]
# calculate_ratio_of_feature_true(target_indexes)