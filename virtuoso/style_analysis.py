import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import pickle
import _pickle as cPickle
# from . import pyScoreParser.xml_matching as xml_matching
import os
from .pyScoreParser.midi_utils import utils as utils

def get_emotion_representative_vectors(total_perform_z):
    '''
    total_perform_z: List of dictionary {'E1': List of style vector z in 1D numpy array }
    Out: [style_vectors for Emotion in (E1, E2, E3, E4, E5)]
    '''
    total_z_flattened = torch.Tensor(np.asarray([z for x in total_perform_z for y in ['E1', 'E2', 'E3', 'E4', 'E5'] for z in x[y]]))
    num_pieces = len(total_perform_z)
    num_emotions = 5
    num_sample_per_performances = len(total_perform_z[0]['E1'])
    
    
    total_z_flattened = total_z_flattened.view(num_pieces, num_emotions, num_sample_per_performances, -1)
    abs_mean_by_emotion = torch.mean(total_z_flattened, dim=(0,2))

    orig_emotion_mean = torch.mean(total_z_flattened[:,0,:,:], dim=1)
    normalized_z = total_z_flattened - orig_emotion_mean.unsqueeze(1).unsqueeze(1)
    norm_mean_by_emotion = torch.mean(normalized_z, dim=(0,2))
    
    return abs_mean_by_emotion, norm_mean_by_emotion
    


def embedd_dim_reduction_of_emotion_dataset(total_perform_z, dim_reduction_type='pca'):
    '''
    total_perform_z: List of dictionary {'E1': List of style vector z in 1D numpy array }
    dim_reduction_type (str): string among ['pca', 'umap', 'tsne']
    Out: TSNE embeddings with shape of [Performances X Emotions (5) X Num_Z_Sample_Per_Performance X 2 (t-SNE dimension)]
    '''
    selected_z = np.asarray([z for x in total_perform_z for y in ['E1', 'E2', 'E3', 'E4', 'E5'] for z in x[y]])
    num_sample_per_performances = len(total_perform_z[0]['E1'])
    z_reshaped = selected_z.reshape(len(total_perform_z), 5, num_sample_per_performances, -1)
    mean_z = np.mean(z_reshaped[:,0,:,:], axis=1)
    normalized_z = z_reshaped - mean_z.reshape(len(total_perform_z), 1, 1, -1)
    normalized_z = normalized_z.reshape(len(total_perform_z) * 5 * num_sample_per_performances, -1)
  
    if dim_reduction_type=='pca':
      z_embedded = PCA(n_components=2).fit_transform(selected_z)
      z_normalized = PCA(n_components=2).fit_transform(normalized_z)
    elif dim_reduction_type=='umap':
      from umap import UMAP
      z_embedded = UMAP(n_components=2).fit_transform(selected_z)
      z_normalized = UMAP(n_components=2).fit_transform(normalized_z)
    elif dim_reduction_type=='tsne':
      z_embedded = TSNE(n_components=2).fit_transform(selected_z)
      z_normalized = TSNE(n_components=2).fit_transform(normalized_z)
    else:
      raise Exception(f"Unknown dimension reduction type: {dim_reduction_type}")
    z_embedded = z_embedded.reshape(len(total_perform_z), 5, num_sample_per_performances, -1)
    z_normalized = z_normalized.reshape(len(total_perform_z), 5, num_sample_per_performances, -1)

    return z_embedded, z_normalized

def total_perform_z_to_abs_and_norm(total_perform_z):
    selected_z = np.asarray([z for x in total_perform_z for y in ['E1', 'E2', 'E3', 'E4', 'E5'] for z in x[y]])
    num_sample_per_performances = len(total_perform_z[0]['E1'])
    perform_z_array = selected_z.reshape(len(total_perform_z), 5, num_sample_per_performances, -1)

    mean_z = np.mean(perform_z_array[:,0,:,:], axis=1)
    normalized_z = perform_z_array - mean_z.reshape(len(total_perform_z), 1, 1, -1)

    return perform_z_array, normalized_z

def cross_validate_with_svm(perform_z_array, valid_slices):
    num_sample_per_performances = perform_z_array.shape[-2]
    y = np.zeros_like(perform_z_array)[:,:,:,0]
    for i in range(y.shape[1]):
        y[:,i,:] = i
    perform_z_array = perform_z_array.reshape(-1, num_sample_per_performances, perform_z_array.shape[-1])
    y = y.reshape(-1, num_sample_per_performances)
    confusion_matrix = np.zeros((5,5))
    for valid_ids in valid_slices:
        train_ids = list(set(range(y.shape[0])) - set(valid_ids))
        linear_svc = SVC(kernel='linear')
        linear_svc.fit(perform_z_array[train_ids].reshape(-1,perform_z_array.shape[-1]), y[train_ids].flatten())
        confusion_matrix += cal_svm_accuracy(linear_svc, perform_z_array[valid_ids].reshape(-1,perform_z_array.shape[-1]),y[valid_ids].flatten())

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    confusion_matrix /= np.max(np.sum(confusion_matrix, axis=1))
    return confusion_matrix, accuracy

def get_classification_error_with_svm(total_perform_z, valid_slices):    
    abs_z, norm_z = total_perform_z_to_abs_and_norm(total_perform_z)

    abs_confusion, abs_accuracy = cross_validate_with_svm(abs_z, valid_slices)
    norm_confusion, norm_accuracy = cross_validate_with_svm(norm_z, valid_slices)

    return abs_confusion, abs_accuracy, norm_confusion, norm_accuracy

def cal_svm_accuracy(svc, x, y):
    prediction = svc.predict(x)
    confusion_matrix = np.zeros((5,5))
    for i in range(5):
        ids = np.where(y==i)[0]
        for j in range(5):
            confusion_matrix[i,j] = sum(prediction[ids]==j)
    
    return confusion_matrix

def draw_tsne_for_emotion_data(z_embedded, output_name):
    plt.figure(figsize=(10,10))
    colors = ['black', 'green', 'gold', 'blue', 'red']
    [plt.scatter(z_embedded[:,j,:,0], z_embedded[:,j,:,1] , c=colors[j], s=16) for j in range(5)]
    plt.savefig(output_name)
    plt.close()

def embedd_tsne(z, perf_names):
    # z = np.asarray([x.reshape(-1).cpu().numpy() for x in z])
    z_embedded = TSNE(n_components=2).fit_transform(z)
    plot_tsne_data(z_embedded, perf_names)

    return z_embedded

def embedd_pca(z, perf_names):
    pca2d = PCA(n_components=2)
    embedded_z = pca2d.fit_transform(z)

    plot_tsne_data(embedded_z, perf_names, output_name='pca.png')
    return embedded_z


def plot_tsne_data(data, perf_names, output_name='tsne_test.png'):
    plt.figure(figsize=(10,10))
    # perf_names = [x.split('_')[0] for x in perf_names]
    # perf_names = [x.split('.')[-1] for x in perf_names]
    # perf_names = [x.split(',')[0] for x in perf_names]

    perf_name_dic = list(set(perf_names))
    print('Number of Performer: ', len(perf_name_dic))
    colors = ['black', 'red', 'yellowgreen', 'aqua', 'violet', 'crimson', 'b', 'slateblue', 'magenta', 'lime', 'olive', 'darkgreen', 'gold', 'tomato', 'silver', 'royalblue', 'sienna','slategrey']
    labels = [perf_name_dic.index(x) for x in perf_names]
    for i in range(len(perf_name_dic)):
        corresp_perf = [x==i for x in labels]
        plt.scatter(data[corresp_perf,0], data[corresp_perf,1], label=perf_name_dic[i], c=colors[i], s=16)
    # for i, x in enumerate(data):
    #     performer_class_idx = perf_name_dic.index(perf_names[i])
    #     plt.scatter(x[0], x[1], c=colors[performer_class_idx], label=perf_names[i])

    plt.legend()
    # for i, txt in enumerate(perf_names):
    #     plt.annotate(txt, (data[i,0], data[i,1]))
    plt.savefig(output_name)


def filter_z_by_name(perf_z, perf_name, selected_name):
    new_perf_z = []
    new_perf_name = []
    for z, name in zip(perf_z, perf_name):
        if name in selected_name:
            new_perf_z.append(z)
            new_perf_name.append(name)
    return new_perf_z, new_perf_name


def save_style_data(path, data_stat_name="pedal_refresh", composer_name='Chopin'):
    with open(data_stat_name + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        MEANS, STDS = u.load()

    piece_list = os.listdir(path)
    total_test_x = []
    total_test_y = []
    total_edges = []
    total_note_locations = []
    total_perf_name = []
    total_piece_name = []
    print(piece_list)
    for piece in piece_list:
        piece_path = os.path.join(path, piece) + '/'
        file_list = os.listdir(piece_path)
        perf_name_list = [os.path.splitext(x)[0] for x in file_list if
                          x.endswith('.mid') and not x == 'midi_cleaned.mid' and not x == 'midi.mid']
        piece_x = []
        piece_y = []
        piece_edge = []
        piece_note_locations = []
        piece_perf_name = []
        for perf_name in perf_name_list:
            test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(piece_path, perf_name, composer_name,
                                                                                 MEANS, STDS)
            piece_x.append(test_x)
            piece_y.append(test_y)
            piece_edge.append(edges)
            piece_note_locations.append(note_locations)
            piece_perf_name.append(perf_name)
            total_piece_name.append(piece)

        total_test_x.append(piece_x)
        total_test_y.append(piece_y)
        total_edges.append(piece_edge)
        total_note_locations.append(piece_note_locations)
        total_perf_name.append(piece_perf_name)

    combined_data = [total_test_x, total_test_y, total_edges, total_note_locations, total_perf_name, total_piece_name]
    with open("chopin_parsed_test.dat", "wb") as f:
        pickle.dump(combined_data, f, protocol=2)



def save_emotion_perf_data(path, data_stat_name="pedal_refresh"):
    with open(data_stat_name + "_stat.dat", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        MEANS, STDS = u.load()

    piece_list = utils.find_files_in_subdir(path, '*.musicxml')
    align_list = utils.find_files_in_subdir(path, '*_infer_corresp.txt')
    print(piece_list)
    total_test_x = []
    total_test_y = []
    total_edges = []
    total_note_locations = []
    total_perf_name = []
    for piece in piece_list:
        piece_path = '.'.join(piece.split('.')[0:-1])
        perf_name_list = [x.split('/')[-1][0:-18] for x in align_list if x.split('/')[-1].split('.')[0:3] == piece.split('/')[-1].split('.')[1:4]]

        piece_x = []
        piece_y = []
        piece_edge = []
        piece_note_locations = []
        piece_perf_name = []
        composer_name = piece.split('.')[1]
        if composer_name == 'Mendelssohn':
            composer_name = 'Schubert'
        print(piece, composer_name)
        for perf_name in perf_name_list:
            test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(piece_path, perf_name, composer_name,
                                                                                 MEANS, STDS, search_by_file_name=True)
            piece_x.append(test_x)
            piece_y.append(test_y)
            piece_edge.append(edges)
            piece_note_locations.append(note_locations)
            piece_perf_name.append(perf_name)

        total_test_x.append(piece_x)
        total_test_y.append(piece_y)
        total_edges.append(piece_edge)
        total_note_locations.append(piece_note_locations)
        total_perf_name.append(piece_perf_name)

    combined_data = [total_test_x, total_test_y, total_edges, total_note_locations, total_perf_name]
    with open("style_parsed_emotion007.dat", "wb") as f:
        pickle.dump(combined_data, f, protocol=2)





def load_tsne_and_plot(path):
    with open(path, "rb") as f:
        u = cPickle.Unpickler(f)
        dict = u.load()

    plot_tsne_data(dict['z'], dict['name'])


def load_z_filter_and_plot(path):
    with open(path, "rb") as f:
        u = cPickle.Unpickler(f)
        dict = u.load()

    perf_z, perf_names = dict['z'], dict['name']
    selected_performers = ['Biret', 'Lisiecki', 'Pollini']
    perf_z, perf_names = filter_z_by_name(perf_z, perf_names, selected_performers)
    tsne_z = embedd_tsne(perf_z, perf_names)
    pca_z = embedd_pca(perf_z, perf_names)

    print(tsne_z)