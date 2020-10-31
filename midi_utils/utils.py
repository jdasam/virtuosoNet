from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import pickle
import ntpath
import fnmatch


def save_dict_as_json(config):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    param_path = os.path.join(config.save_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def maybe_make_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def change_name_extension(file_name, new_extension):
    if '.' not in new_extension:
        new_extension = '.' + new_extension
    base = os.path.splitext(file_name)[0]
    return base + new_extension


def split_head_and_tail(file_path):
    head, tail = ntpath.split(file_path)
    return head, tail


def save_obj(obj, name):
    save_name = change_name_extension(name, '.pkl')
    with open(save_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    save_name = change_name_extension(name, '.pkl')
    with open(save_name, 'rb') as f:
        return pickle.load(f)


def find_files_in_subdir(folder, regexp):
    match_files = []
    for root, subdir, base in os.walk(folder):
        for file_name in fnmatch.filter(base, regexp):
            match_files.append(os.path.join(root, file_name))
    return match_files


'''
import numpy as np
import math
from numpy.lib.stride_tricks import as_strided

def array2stack(array, length, hop=None):
  if hop is None:
    hop = length
  assert (array.shape[0] - length) % hop == 0, 'length of array is not fit. l={:d}, length={:d}, hop={:d}' \
    .format(array.shape[0], length, hop)
  strides = array.strides
  stack = as_strided(array, ((array.shape[0] - length) // hop + 1, length, array.shape[1]),
                     (strides[0] * hop, strides[0], strides[1]))
  return stack


def overlap_stack2array(stack):
  # TODO: what if hop != stack.shape[1]//2 ?
  hop = stack.shape[1] // 2
  length = (stack.shape[0] + 1) * hop
  array = np.zeros((length, stack.shape[2]))
  array[:hop // 2, :] = stack[0, :hop // 2, :]
  for n in xrange(stack.shape[0]):
    array[n * hop + hop // 2: n * hop + 3 * hop // 2, :] = stack[n, hop // 2: 3 * hop // 2, :]
  array[(stack.shape[0] - 1) * hop + 3 * hop // 2:, :] = stack[stack.shape[0] - 1, 3 * hop // 2:, :]

  return array


def onset2delayed(onset, delay_len=10):
  rolled_onset = np.zeros(onset.shape)
  for k in range(delay_len):
    temp = np.roll(onset, k, axis=0)
    temp[0, :] = 0
    weight = math.sqrt((delay_len - k) / float(delay_len))
    rolled_onset += temp * weight
  rolled_onset[rolled_onset > 1] = 1
  return rolled_onset


def record_as_text(config, text):
  if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)
  record_txt = config.save_dir + '/' + 'summary.txt'
  f = open(record_txt, 'a')
  f.write(text)
  f.close()


def get_data_list(set_name, set_num=1):
    f = open('data_list/config{:d}_{}.txt'.format(set_num, set_name), 'rb')
    data_list = f.readlines()
    for n in xrange(len(data_list)):
        data_list[n] = data_list[n].replace('\n', '')
    f.close()
    data_list.sort()
    return data_list


def pad2d(feature, seg_len):
    if feature.shape[0] % seg_len != 0:
        pad_len = seg_len - feature.shape[0] % seg_len
        feature = np.pad(feature, ((0, pad_len), (0, 0)), 'constant')
    return feature


def normalize(feature, mean, std):
    return np.divide((feature - mean[None, :]), std[None, :], where=(std[None, :] != 0))
'''