import performanceWorm as perf_worm
import pickle
import numpy as np

code_list = ['isgn_altv_ready_low_kld2', 'han_ar_baseline_ready', 'han_ar_vnet_ready_lowkld', 'han_ar_graph_ready']

total_data = []
for code in code_list:
    with open(code+'_cor.dat', "rb") as f:
        u = pickle._Unpickler(f)
        corr = u.load()
        total_data.append(corr)



num_excerpts = len(total_data[0])
num_model = len(code_list)

# print(num_excerpts)
for j in range(num_excerpts):
# for j in range(1):
    features = total_data[0][j].tempo_features
    for i in range(1, num_model):
        features.append(total_data[i][j].tempo_features[-1])

    # path = total_data[0][j].path_name
    # slc_idx = total_data[0][j].slice_index
    # save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_', 10) + '_note{}-{}.png'.format(slc_idx[0], slc_idx[1])
    # perf_worm.plot_model_features_compare(features, 4, save_name)


for code in code_list:
    with open(code+'_cor.dat', "rb") as f:
        u = pickle._Unpickler(f)
        selected_corr = u.load()
    mean_r = []
    num_passed = 0
    num_high = 0
    num_middle = 0
    num_low = 0

    for cor in selected_corr:
        clean_r = []
        passed = False
        high = False
        middle = False
        low = False
        under_zero = False
        for r in cor.tempo_r:
            if r > 0.7:
                passed = True
            if r > 0.9:
                high = True
            if r > 0.5:
                middle= True
            if r > 0.3:
                low = True
            if not np.isnan(r):
                clean_r.append(r)
        max_r = np.median(clean_r)
        mean_r.append(max_r)
        if passed:
            num_passed += 1
        if high:
            num_high += 1
        if middle:
            num_middle += 1
        if low:
            num_low += 1

    print(np.mean(mean_r))
    print(num_passed)
    print(num_high)
    print(num_middle)
    print(num_low)