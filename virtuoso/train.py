from pathlib import Path
import os
import shutil
import random
import math
import copy
import time

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
from .dataset import ScorePerformDataset, FeatureCollate
from .logger import Logger
from .loss import LossCalculator
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

def prepare_dataloader(args):
    hier_type = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas']
    curr_type = [x for x in hier_type if getattr(args, x)]

    train_set = ScorePerformDataset(args.data_path, type="train", len_slice=args.len_slice, graph_keys=args.graph_keys, hier_type=curr_type)
    valid_set = ScorePerformDataset(args.data_path, type="valid", len_slice=args.len_valid_slice, graph_keys=args.graph_keys, hier_type=curr_type)
    
    train_loader = DataLoader(train_set, 1, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=FeatureCollate())
    valid_loader = DataLoader(valid_set, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=FeatureCollate())
    return train_loader, valid_loader

def prepare_directories_and_logger(output_dir, log_dir, exp_name):
    print(output_dir, log_dir)
    (output_dir / exp_name / log_dir).mkdir(exist_ok=True, parents=True)
    logger = Logger(output_dir / exp_name / log_dir)
    return logger

def batch_to_device(batch, device):
    batch_x, batch_y, note_locations, align_matched, pedal_status, edges = batch
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    align_matched = align_matched.to(device)
    pedal_status = pedal_status.to(device)
    edges = edges.to(device)
    return batch_x, batch_y, note_locations, align_matched, pedal_status, edges


def train(args,
          model,
          device,
          num_epochs, 
          criterion,
          exp_name,
          ):

    train_loader, valid_loader = prepare_dataloader(args)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger = prepare_directories_and_logger(args.checkpoints_dir, args.logs, exp_name)
    shutil.copy(args.yml_path, args.checkpoints_dir/exp_name)
    loss_calculator = LossCalculator(criterion, args, logger)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    best_valid_loss = float("inf")
    # best_trill_loss = float("inf")
    start_epoch = 0
    iteration = 0
    
    if args.resume_training and not args.trainTrill:
        if os.path.isfile('prime_' + args.modelCode + args.resume):
            print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
            # model_codes = ['prime', 'trill']
            filename = 'prime_' + args.modelCode + args.resume
            checkpoint = th.load(filename,  map_location=device)
            best_valid_loss = checkpoint['best_valid_loss']
            model.load_state_dict(checkpoint['state_dict'])
            model.device = device
            optimizer.load_state_dict(checkpoint['optimizer'])
            num_updated = checkpoint['training_step']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
            start_epoch = checkpoint['epoch'] - 1
            print('Best valid loss was ', best_valid_loss)


    # load data
    print('Loading the training data...')

    for epoch in range(start_epoch, num_epochs):
        print('current training step is ', iteration)
        train_loader.dataset.update_slice_info()
        for _, batch in enumerate(train_loader):
            start =time.perf_counter()
            batch_x, batch_y, note_locations, align_matched, pedal_status, edges = batch_to_device(batch, device)

            outputs, perform_mu, perform_var, total_out_list = model(batch_x, batch_y, edges, note_locations)
            total_loss, loss_dict = loss_calculator(outputs, batch_y, total_out_list, note_locations, align_matched, pedal_status)

            kld_weight = sigmoid((iteration - args.kld_sig) / (args.kld_sig/10)) * args.kld_max
            if isinstance(perform_mu, bool):
                perform_kld = th.zeros(1)
            else:
                perform_kld = -0.5 * \
                    th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                total_loss += perform_kld * kld_weight
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            duration = time.perf_counter() - start
            logger.log_training(total_loss.item(),loss_dict, grad_norm, optimizer.param_groups[0]['lr'], duration, iteration)
            iteration += 1

            if iteration % args.iters_per_checkpoint == 0:
                valid_loss = []
                valid_loss_dict = []
                model.eval()
                with th.no_grad():
                    for _, batch in enumerate(valid_loader):
                        batch_x, batch_y, note_locations, align_matched, pedal_status, edges = batch_to_device(batch, device)

                        outputs, perform_mu, perform_var, total_out_list = model(batch_x, batch_y, edges, note_locations)
                        total_loss, loss_dict = loss_calculator(outputs, batch_y, total_out_list, note_locations, align_matched, pedal_status)
                        valid_loss.append(total_loss.item())
                        valid_loss_dict.append(loss_dict)
                    valid_loss = sum(valid_loss) / len(valid_loss)
                    print('Valid loss: {}'.format(valid_loss))
                model.train()
                logger.log_validation(valid_loss, valid_loss_dict, model, iteration)
                is_best = valid_loss < best_valid_loss
                best_valid_loss = min(best_valid_loss, valid_loss)
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_valid_loss': best_valid_loss,
                    'optimizer': optimizer.state_dict(),
                    'training_step': iteration
                }, is_best)

            # key_lists = [0]
            # key = 0
            # for i in range():
            #     while key in key_lists:
            #         key = random.randrange(-5, 7)
            #     key_lists.append(key)
            # tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
            #         utils.batch_time_step_run(batch, model=train_model)
            # for i in range(args.num_key_augmentation+1):
            #     key = key_lists[i]
            #     temp_train_x = dp.key_augmentation(batch['input'], key)
            #     kld_weight = sigmoid((NUM_UPDATED - args.kld_sig) / (args.kld_sig/10)) * args.kld_max
            #     training_data = {**batch, 'kld_weight':kld_weight}
            #     training_data['input_data'] = temp_train_x
            #     # training_data = {'x': temp_train_x, 'y': train_y, 'graphs': graphs,
            #     #                     'note_locations': note_locations,
            #     #                     'align_matched': align_matched, 'pedal_status': pedal_status,
            #     #                     'slice_idx': slice_idx, 'kld_weight': kld_weight}

            #     tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss, trill_loss, kld = \
            #         utils.batch_time_step_run(training_data, model=train_model)

                


        


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