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
import wandb


from torch.utils.data import DataLoader
from .parser import get_parser
from .utils import categorize_value_to_vector
from . import model_constants as const
from .dataset import ScorePerformDataset, FeatureCollate, MultiplePerformSet, multi_collate, EmotionDataset
from .logger import Logger
from .loss import LossCalculator, cal_multiple_perf_style_loss
from .model_utils import make_higher_node
from .model import SimpleAttention
from . import utils
from .inference import generate_midi_from_xml
from .model_constants import valid_piece_list
from .emotion import validate_style_with_emotion_data
from . import style_analysis as sty
# from . import inference

def sigmoid(x, gain=1):
  # why not np.sigmoid or something?
  return 1 / (1 + math.exp(-gain*x))


def load_model(model, optimizer, device, args):
    # if args.resumeTraining and not args.trainTrill:
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    # model_codes = ['prime', 'trill']
    # filename = 'prime_' + args.modelCode + args.resume
    checkpoint = th.load(args.checkpoint,  map_location='cpu')
    best_valid_loss = checkpoint['best_valid_loss']
    model.load_state_dict(checkpoint['state_dict'])
    model.device = device
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['training_step']
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.checkpoint, checkpoint['epoch']))
    start_epoch = checkpoint['epoch'] - 1
    best_prime_loss = checkpoint['best_valid_loss']
    print(f'Best valid loss was {best_prime_loss}')
    return model, optimizer, start_epoch, iteration, best_valid_loss

def prepare_dataloader(args):
    hier_type = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas', 'meas_note']
    curr_type = [x for x in hier_type if getattr(args, x)]

    train_set = ScorePerformDataset(args.data_path, type="train", len_slice=args.len_slice, len_graph_slice=args.len_graph_slice, graph_keys=args.graph_keys, hier_type=curr_type)
    valid_set = ScorePerformDataset(args.data_path, type="valid", len_slice=args.len_valid_slice, len_graph_slice=args.len_graph_slice, graph_keys=args.graph_keys, hier_type=curr_type)
    emotion_set = EmotionDataset(args.emotion_data_path, type="train", len_slice=args.len_valid_slice * 2, len_graph_slice=args.len_valid_slice * 2, graph_keys=args.graph_keys,)
    multi_perf_set = MultiplePerformSet(args.data_path, type="train", len_slice=args.len_slice, len_graph_slice=args.len_graph_slice, graph_keys=args.graph_keys, hier_type=curr_type)

    train_loader = DataLoader(train_set, 1, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    valid_loader = DataLoader(valid_set, 1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    emotion_loader = DataLoader(emotion_set, 5, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    multi_perf_loader = DataLoader(multi_perf_set, 1, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=multi_collate)
    # emotion_loader = None
    return train_loader, valid_loader, emotion_loader, multi_perf_loader

def prepare_directories_and_logger(output_dir, log_dir, exp_name, make_log=True):
    out_dir = output_dir / exp_name
    print(out_dir)
    (out_dir / log_dir).mkdir(exist_ok=True, parents=True)
    if make_log:
        logger = Logger(out_dir/ log_dir)
        return logger, out_dir
    else:
        return None, out_dir



def generate_midi_for_validation(model, valid_data, args):
    input = th.Tensor(valid_data['input_data'])




def validate_with_midi_generation(model, total_perform_z, valid_piece_list, out_dir, iteration, device):
    abs_mean_by_emotion, norm_mean_by_emotion = sty.get_emotion_representative_vectors(total_perform_z)
    for piece in valid_piece_list:
        xml_path = Path(f'/home/svcapp/userdata/chopin_cleaned/{piece[0]}') / 'musicxml_cleaned.musicxml'
        composer = piece[1]
        save_path = out_dir / f"{'_'.join(piece[0].split('/'))}_zero_iter_{iteration}.mid"
        generate_midi_from_xml(model, xml_path, composer, save_path, device)
        for i, emotion_z in enumerate(abs_mean_by_emotion):
            save_path = out_dir / f"{'_'.join(piece[0].split('/'))}_absE{i+1}_iter_{iteration}.mid"
            generate_midi_from_xml(model, xml_path, composer, save_path, device, initial_z=emotion_z)
        for i, emotion_z in enumerate(norm_mean_by_emotion):
            save_path = out_dir / f"{'_'.join(piece[0].split('/'))}_normE{i+1}_iter_{iteration}.mid"
            generate_midi_from_xml(model, xml_path, composer, save_path, device, initial_z=emotion_z)
    

def train(args,
          model,
          device,
          num_epochs, 
          criterion,
          exp_name,
          ):


    if args.make_log:
      wandb.init(project="VirtuosoNet", entity="dasaem")
      wandb.config = args
      wandb.watch(model)
    train_loader, valid_loader, emotion_loader, multi_perf_loader = prepare_dataloader(args)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger, out_dir = prepare_directories_and_logger(args.checkpoints_dir, args.logs, exp_name, args.make_log)
    shutil.copy(args.yml_path, args.checkpoints_dir/exp_name)
    loss_calculator = LossCalculator(criterion, args, logger)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    best_valid_loss = float("inf")
    # best_trill_loss = float("inf")
    start_epoch = 0
    iteration = 0
    multi_perf_iter = 0 

    if args.resume_training:
        model, optimizer, start_epoch, iteration, best_valid_loss = load_model(model, optimizer, device, args)
    model.stats = train_loader.dataset.stats
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # load data
    print('Loading the training data...')

    for epoch in range(start_epoch, num_epochs):
        print('current training step is ', iteration)
        train_loader.dataset.update_slice_info()
        for _, batch in enumerate(train_loader):
            start =time.perf_counter()
            if args.meas_note:
                batch_x, batch_y, beat_y, meas_y, note_locations, align_matched, pedal_status, edges = utils.batch_to_device(batch, device)
                outputs, perform_mu, perform_var, total_out_list = model(batch_x, batch_y, edges, note_locations)
                total_loss, loss_dict = loss_calculator(outputs, {'note':batch_y, 'beat':beat_y, 'measure':meas_y}, total_out_list, note_locations, align_matched, pedal_status)
            else:
                batch_x, batch_y, note_locations, align_matched, pedal_status, edges = utils.batch_to_device(batch, device)
                outputs, perform_mu, perform_var, total_out_list = model(batch_x, batch_y, edges, note_locations)
                total_loss, loss_dict = loss_calculator(outputs, batch_y, total_out_list, note_locations, align_matched, pedal_status)

            kld_weight = sigmoid((iteration - args.kld_sig) / (args.kld_sig/10)) * args.kld_max
            perform_kld = -0.5 * \
                th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
            total_loss += perform_kld * kld_weight
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            duration = time.perf_counter() - start
            loss_dict["kld"] = perform_kld
            loss_dict["kld_weight"] = kld_weight
            if args.make_log:
                wandb.log(loss_dict)
                logger.log_training(total_loss.item(),loss_dict, grad_norm, optimizer.param_groups[0]['lr'], duration, iteration)
            iteration += 1

            if args.multi_perf_compensation and iteration % args.iters_per_multi_perf == 0:
                batch = next(iter(multi_perf_loader))
                batch_x, batch_y, note_locations, edges  = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if edges is not None:
                    edges = edges.to(device)
                perform_mu, perform_var = model.encode_style_distribution(batch_x, batch_y, edges, note_locations)
                loss, loss_dict = cal_multiple_perf_style_loss(perform_mu, perform_var, args.multi_perf_dist_loss_margin)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                multi_perf_iter += 1
                if multi_perf_iter == len(multi_perf_loader):
                    multi_perf_loader.dataset.update_slice_info()
                    multi_perf_iter = 0
                if args.make_log:
                    logger.log_multi_perf(loss.item(),loss_dict, grad_norm, iteration)

                
            if iteration % args.iters_per_checkpoint == 0:
                valid_loss = []
                valid_loss_dict = []
                model.eval()
                with th.no_grad():
                    for _, batch in enumerate(valid_loader):
                        if args.meas_note:
                            batch_x, batch_y, beat_y, meas_y, note_locations, align_matched, pedal_status, edges = utils.batch_to_device(batch, device)
                            outputs, perform_mu, perform_var, total_out_list = model(batch_x, batch_y, edges, note_locations)
                            total_out_list['iter_out'] = total_out_list['iter_out'][-1:]
                            total_loss, loss_dict = loss_calculator(outputs, {'note':batch_y, 'measure':meas_y, 'beat':beat_y}, total_out_list, note_locations, align_matched, pedal_status)
                        else:
                            batch_x, batch_y, note_locations, align_matched, pedal_status, edges = utils.batch_to_device(batch, device)
                            outputs, perform_mu, perform_var, total_out_list = model(batch_x, batch_y, edges, note_locations)
                            total_loss, loss_dict = loss_calculator(outputs, batch_y, total_out_list[-1:], note_locations, align_matched, pedal_status)

                        perform_kld = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                        loss_dict['kld'] = perform_kld
                        valid_loss.append(total_loss.item())
                        valid_loss_dict.append(loss_dict)
                    valid_loss = sum(valid_loss) / len(valid_loss)
                    print('Valid loss: {}'.format(valid_loss))     

                if args.make_log:
                    logger.log_validation(valid_loss, valid_loss_dict, model, iteration)
                is_best = valid_loss < best_valid_loss
                best_valid_loss = min(best_valid_loss, valid_loss)
                utils.save_checkpoint(args.checkpoints_dir / exp_name, 
                    {'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_valid_loss': best_valid_loss,
                    'optimizer': optimizer.state_dict(),
                    'training_step': iteration,
                    'stats': model.stats,
                    'network_params': model.network_params,
                    'model_code': args.model_code
                }, is_best)
                total_perform_z, abs_confusion, abs_accuracy, norm_confusion, norm_accuracy = validate_style_with_emotion_data(model, emotion_loader, device, out_dir, iteration)
                if args.make_log:
                    logger.log_style_analysis(abs_confusion, abs_accuracy, norm_confusion, norm_accuracy, iteration)
                if not args.is_hier:
                    validate_with_midi_generation(model, total_perform_z, valid_piece_list, out_dir, iteration, device)
                

                model.train()

                


    #end of epoch


# elif args.sessMode in ['test', 'testAll', 'testAllzero', 'encode', 'encodeAll', 'evaluate', 'correlation']:
# ### test session
