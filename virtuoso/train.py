from pathlib import Path
import os
import random
import math
import copy

import numpy as np
import torch as th
import pickle

from torch.utils.data import DataLoader
from .parser import get_parser
from .utils import categorize_value_to_vector
from . import data_process as dp  # maybe confuse with dynamic programming?
from . import graph
from . import utils
from . import model_constants as const
from . import model
from .dataset import ScorePerformDataset
# from . import inference

def sigmoid(x, gain=1):
  # why not np.sigmoid or something?
  return 1 / (1 + math.exp(-gain*x))

class TraningSample():
    def __init__(self, index):
        self.index = index
        self.slice_indexes = None

def load_model(model, optimizer, device, args):
    # if args.resumeTraining and not args.trainTrill:
    if os.path.isfile('prime_' + args.modelCode + args.resume):
        print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
        # model_codes = ['prime', 'trill']
        filename = 'prime_' + args.modelCode + args.resume
        checkpoint = th.load(filename,  map_location=device)
        best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['state_dict'])
        model.device = device
        optimizer.load_state_dict(checkpoint['optimizer'])
        NUM_UPDATED = checkpoint['training_step']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(filename, checkpoint['epoch']))
        start_epoch = checkpoint['epoch'] - 1
        best_prime_loss = checkpoint['best_valid_loss']
        print('Best valid loss was ', best_prime_loss)

def prepare_train(args):
    train_set = ScorePerformDataset(args.data_path, type="train")
    valid_set = ScorePerformDataset(args.data_path, type="valid")
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_set, 1, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, 1, shuffle=False, num_workers=args.num_workers)
    return train_loader, valid_loader, optimizer

def train(args,
          model,
          device,
          optimizer, 
          num_epochs, 
          bins, 
          time_steps,
          criterion, 
          NUM_INPUT, 
          NUM_OUTPUT):


    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    best_prime_loss = float("inf")
    best_trill_loss = float("inf")
    start_epoch = 0
    NUM_UPDATED = 0
    
    if args.resumeTraining and not args.trainTrill:
        if os.path.isfile('prime_' + args.modelCode + args.resume):
            print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
            # model_codes = ['prime', 'trill']
            filename = 'prime_' + args.modelCode + args.resume
            checkpoint = th.load(filename,  map_location=device)
            best_valid_loss = checkpoint['best_valid_loss']
            model.load_state_dict(checkpoint['state_dict'])
            model.device = device
            optimizer.load_state_dict(checkpoint['optimizer'])
            NUM_UPDATED = checkpoint['training_step']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            start_epoch = checkpoint['epoch'] - 1
            best_prime_loss = checkpoint['best_valid_loss']
            print('Best valid loss was ', best_prime_loss)


    # load data
    print('Loading the training data...')
    training_data_name = args.dataName + ".dat"
    if not os.path.isfile(training_data_name):
        training_data_name = '/mnt/ssd1/jdasam_data/' + training_data_name
    with open(training_data_name, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        # p = u.load()
        # complete_xy = pickle.load(f)
        complete_xy = u.load()

    train_xy = complete_xy['train']
    test_xy = complete_xy['valid']
    print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))
    print('training sample example', train_xy[0][0][0])

    train_model = model

    # total_step = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        print('current training step is ', NUM_UPDATED)
        tempo_loss_total = []
        vel_loss_total = []
        dev_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_total = []

        # if RAND_TRAIN:
        if True:
            num_perf_data = len(train_xy)
            remaining_samples = []
            for i in range(num_perf_data):
                remaining_samples.append(TraningSample(i))
            while len(remaining_samples) > 0:
                new_index = random.randrange(0, len(remaining_samples))
                selected_sample = remaining_samples[new_index]
                train_x = train_xy[selected_sample.index][0]
                train_y = train_xy[selected_sample.index][1]
                if args.trainingLoss == 'CE':
                    train_y = categorize_value_to_vector(train_y, bins)
                note_locations = train_xy[selected_sample.index][2]
                align_matched = train_xy[selected_sample.index][3]
                pedal_status = train_xy[selected_sample.index][4]
                edges = train_xy[selected_sample.index][5]
                data_size = len(train_x)

                if selected_sample.slice_indexes is None:
                    measure_numbers = [x.measure for x in note_locations]
                    if args.hier_meas and args.hierarchy:
                        selected_sample.slice_indexes = dp.make_slice_with_same_measure_number(data_size,
                                                                                               measure_numbers,
                                                                                               measure_steps=time_steps)

                    else:
                        selected_sample.slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=time_steps)

                num_slice = len(selected_sample.slice_inde)
                selected_idx = random.randrange(0,num_slice)
                slice_idx = selected_sample.slice_indexes[selected_idx]

                if model.is_graph:
                    graphs = graph.edges_to_matrix_short(edges, slice_idx)
                else:
                    graphs = None

                key_lists = [0]
                key = 0
                for i in range():
                    while key in key_lists:
                        key = random.randrange(-5, 7)
                    key_lists.append(key)

                for i in range(args.num_key_augmentation+1):
                    key = key_lists[i]
                    temp_train_x = dp.key_augmentation(train_x, key)
                    kld_weight = sigmoid((NUM_UPDATED - args.kld_sig) / (args.kld_sig/10)) * args.kld_max

                    training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                     'note_locations': note_locations,
                                     'align_matched': align_matched, 'pedal_status': pedal_status,
                                     'slice_idx': slice_idx, 'kld_weight': kld_weight}

                    tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                        utils.batch_time_step_run(training_data, model=train_model)
                    tempo_loss_total.append(tempo_loss.item())
                    vel_loss_total.append(vel_loss.item())
                    dev_loss_total.append(dev_loss.item())
                    articul_loss_total.append(articul_loss.item())
                    pedal_loss_total.append(pedal_loss.item())
                    trill_loss_total.append(trill_loss.item())
                    kld_total.append(kld.item())
                    NUM_UPDATED += 1

                del selected_sample.slice_indexes[selected_idx]
                if len(selected_sample.slice_indexes) == 0:
                    # print('every slice in the sample is trained')
                    del remaining_samples[new_index]
        '''
        else:
            for xy_tuple in train_xy:
                train_x = xy_tuple[0]
                train_y = xy_tuple[1]
                if args.trainingLoss == 'CE':
                    train_y = categorize_value_to_vector(train_y, bins)
                note_locations = xy_tuple[2]
                align_matched = xy_tuple[3]
                pedal_status = xy_tuple[4]
                edges = xy_tuple[5]

                data_size = len(note_locations)
                if model.is_graph:
                    graphs = edges_to_matrix(edges, data_size)
                else:
                    graphs = None
                measure_numbers = [x.measure for x in note_locations]
                # graphs = edges_to_sparse_tensor(edges)
                total_batch_num = int(math.ceil(data_size / (time_steps * batch_size)))

                key_lists = [0]
                key = 0
                for i in range(num_key_augmentation):
                    while key in key_lists:
                        key = random.randrange(-5, 7)
                    key_lists.append(key)

                for i in range(num_key_augmentation+1):
                    key = key_lists[i]
                    temp_train_x = dp.key_augmentation(train_x, key)
                    slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers, steps=time_steps)
                    kld_weight = sigmoid((NUM_UPDATED - KLD_SIG) / (KLD_SIG/10)) * KLD_MAX

                    for slice_idx in slice_indexes:
                        training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
                                         'note_locations': note_locations,
                                         'align_matched': align_matched, 'pedal_status': pedal_status,
                                         'slice_idx': slice_idx, 'kld_weight': kld_weight}

                        tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
                            batch_time_step_run(training_data, model=train_model)
                        tempo_loss_total.append(tempo_loss.item())
                        vel_loss_total.append(vel_loss.item())
                        dev_loss_total.append(dev_loss.item())
                        articul_loss_total.append(articul_loss.item())
                        pedal_loss_total.append(pedal_loss.item())
                        trill_loss_total.append(trill_loss.item())
                        kld_total.append(kld.item())
                        NUM_UPDATED += 1
        '''
        print('Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
              .format(epoch + 1, num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
                      np.mean(dev_loss_total), np.mean(articul_loss_total), np.mean(pedal_loss_total), np.mean(trill_loss_total), np.mean(kld_total)))


        ## Validation
        tempo_loss_total =[]
        vel_loss_total =[]
        deviation_loss_total =[]
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_loss_total = []

        for xy_tuple in test_xy:
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]
            edges = xy_tuple[5]
            if model.is_graph:
                graphs = graph.edges_to_matrix(edges, len(test_x))
            else:
                graphs = None
            if args.loss == 'CE':
                test_y = categorize_value_to_vector(test_y, bins)

            batch_x, batch_y = utils.handle_data_in_tensor(test_x, test_y, args, device)
            batch_x = batch_x.view(1, -1, NUM_INPUT)
            batch_y = batch_y.view(1, -1, NUM_OUTPUT)
            # input_y = th.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(device)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(device)
            pedal_status = th.Tensor(pedal_status).view(1,-1,1).to(device)
            outputs, total_z = utils.run_model_in_steps(batch_x, batch_y, args, graphs, note_locations, model, device)

            # valid_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], batch_y[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], align_matched)
            if args.hierarchy:
                if args.hier_meas:
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif args.hier_beat:
                    hierarchy_numbers = [x.beat for x in note_locations]
                tempo_y = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 0)
                vel_y = model.note_tempo_infos_to_beat(batch_y, hierarchy_numbers, 0, 1)

                tempo_loss = criterion(outputs[:, :, 0:1], tempo_y)
                vel_loss = criterion(outputs[:, :, 1:2], vel_y)
                if args.deltaLoss:
                    tempo_out_delta = outputs[:, 1:, 0:1] - outputs[:, :-1, 0:1]
                    tempo_true_delta = tempo_y[:, 1:, :] - tempo_y[:, :-1, :]
                    vel_out_delta = outputs[:, 1:, 1:2] - outputs[:, :-1, 1:2]
                    vel_true_delta = vel_y[:, 1:, :] - vel_y[:, :-1, :]

                    tempo_loss += criterion(tempo_out_delta, tempo_true_delta) * args.delta_weight
                    vel_loss += criterion(vel_out_delta, vel_true_delta) * args.delta_weight

                deviation_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                trill_loss = th.zeros(1)

                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
            elif args.trill:
                trill_bool = batch_x[:,:, const.is_trill_index_concated] == 1
                trill_bool = trill_bool.float().view(1,-1,1).to(device)
                trill_loss = criterion(outputs, batch_y, trill_bool)

                tempo_loss = th.zeros(1)
                vel_loss = th.zeros(1)
                deviation_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                kld_loss = th.zeros(1)
                kld_loss_total.append(kld_loss.item())

            else:
                tempo_loss = utils.cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0, qpm_idx, criterion, args, device)
                if args.loss =='CE':
                    vel_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM:const.NUM_TEMPO_PARAM+len(bins[1])], batch_y[:,:,const.NUM_TEMPO_PARAM:const.NUM_TEMPO_PARAM+len(bins[1])], align_matched)
                    deviation_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM+len(bins[1]):const.NUM_TEMPO_PARAM+len(bins[1])+len(bins[2])],
                                            batch_y[:,:,const.NUM_TEMPO_PARAM+len(bins[1]):const.NUM_TEMPO_PARAM+len(bins[1])+len(bins[2])])
                    pedal_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM+len(bins[1])+len(bins[2]):-const.num_trill_param],
                                            batch_y[:,:,const.NUM_TEMPO_PARAM+len(bins[1])+len(bins[2]):-const.num_trill_param])
                    trill_loss = criterion(outputs[:,:,-const.num_trill_param:], batch_y[:,:,-const.num_trill_param:])
                else:
                    vel_loss = criterion(outputs[:, :, const.VEL_PARAM_IDX], batch_y[:, :, const.VEL_PARAM_IDX], align_matched)
                    deviation_loss = criterion(outputs[:, :, const.DEV_PARAM_IDX], batch_y[:, :, const.DEV_PARAM_IDX], align_matched)
                    articul_loss = criterion(outputs[:, :, const.PEDAL_PARAM_IDX], batch_y[:, :, const.PEDAL_PARAM_IDX], pedal_status)
                    pedal_loss = criterion(outputs[:, :, const.PEDAL_PARAM_IDX+1:], batch_y[:, :, const.PEDAL_PARAM_IDX+1:], align_matched)
                    trill_loss = th.zeros(1)
                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())

            # valid_loss_total.append(valid_loss.item())
            tempo_loss_total.append(tempo_loss.item())
            vel_loss_total.append(vel_loss.item())
            deviation_loss_total.append(deviation_loss.item())
            articul_loss_total.append(articul_loss.item())
            pedal_loss_total.append(pedal_loss.item())
            trill_loss_total.append(trill_loss.item())

        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_articul_loss = np.mean(articul_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)
        mean_kld_loss = np.mean(kld_loss_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss + mean_articul_loss + mean_pedal_loss * 7 + mean_kld_loss * kld_weight) / (11 + kld_weight)

        print("Valid Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss , mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss))

        is_best = mean_valid_loss < best_prime_loss
        best_prime_loss = min(mean_valid_loss, best_prime_loss)

        is_best_trill = mean_trill_loss < best_trill_loss
        best_trill_loss = min(mean_trill_loss, best_trill_loss)

        if args.trainTrill:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_valid_loss': best_trill_loss,
                'optimizer': optimizer.state_dict(),
                'training_step': NUM_UPDATED
            }, is_best_trill, model_name='trill')
        else:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_valid_loss': best_prime_loss,
                'optimizer': optimizer.state_dict(),
                'training_step': NUM_UPDATED
            }, is_best, model_name='prime')


    #end of epoch


# elif args.sessMode in ['test', 'testAll', 'testAllzero', 'encode', 'encodeAll', 'evaluate', 'correlation']:
# ### test session




def train_model(dataset, model, kwargs):
    '''
    train model for specific steps.
    return average loss to monitor
    '''

    raise NotImplementedError

    return average_loss

def validate_model(dataset, model, kwargs):
    '''
    validate model without backprop, with longer sequence length
    return average loss
    '''

    return average_loss