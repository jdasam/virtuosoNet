from pathlib import Path
import os
import random
import math
import copy

import numpy as np
import torch as th
import pickle

from .parser import get_parser
from .utils import categorize_value_to_vector
from . import data_process as dp  # maybe confuse with dynamic programming?
from . import graph
from . import utils
from . import model_constants as const
from . import model
# from . import dataset
from . import inference

def sigmoid(x, gain=1):
  # why not np.sigmoid or something?
  return 1 / (1 + math.exp(-gain*x))

class TraningSample():
    def __init__(self, index):
        self.index = index
        self.slice_indexes = None

def train(args,
          model,
          train_data,
          valid_data,
          device,
          optimizer, 
          bins,
          criterion):
    # isn't this redundant?
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of Network Parameters is ', params)

    best_prime_loss = float("inf")
    best_trill_loss = float("inf")
    start_epoch = 0
    NUM_UPDATED = 0

    if args.resumeTraining and not args.trainTrill:
        # Load trained-model to resume the training process
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
    '''
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
    '''
    train_xy = train_data
    test_xy = valid_data
    print('number of train performances: ', len(train_xy), 'number of valid perf: ', len(test_xy))
    print('training sample example', train_xy[0]['input_data'][0])

    train_model = model

    # total_step = len(train_loader)
    for epoch in range(start_epoch, args.num_epochs):
        print('current training step is ', NUM_UPDATED)
        tempo_loss_total = []
        vel_loss_total = []
        dev_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        trill_loss_total = []
        kld_total = []

        num_perf_data = len(train_xy)
        remaining_samples = []
        for i in range(num_perf_data):
            temp_training_sample = TraningSample(i)
            measure_numbers = [x.measure for x in train_xy[i]['note_location']]
            data_size = len(train_xy[i]['input_data'])
            if model.config.hierarchy_level == 'measure':
                temp_training_sample.slice_indexes = dp.make_slice_with_same_measure_number(data_size,
                                                                                       measure_numbers,
                                                                                       measure_steps=args.time_steps)

            else:
                temp_training_sample.slice_indexes = dp.make_slicing_indexes_by_measure(data_size, measure_numbers,
                                                                                   steps=args.time_steps)
            remaining_samples.append(temp_training_sample)
        # print('Total number of training slices: ', sum([len(x.slice_indexes) for x in remaining_samples if x.slice_indexes]))
        while len(remaining_samples) > 0:
            new_index = random.randrange(0, len(remaining_samples))
            selected_sample = remaining_samples[new_index]
            # train_x = train_xy[selected_sample.index][0]
            # train_y = train_xy[selected_sample.index][1]
            train_x = train_xy[selected_sample.index]['input_data']
            train_y = train_xy[selected_sample.index]['output_data']
            # if args.loss == 'CE':
            #     train_y = categorize_value_to_vector(train_y, bins)
            note_locations = train_xy[selected_sample.index]['note_location']
            align_matched = train_xy[selected_sample.index]['align_matched']
            # TODO: which variable would be corresponds to pedal status?
            # pedal_status = train_xy[selected_sample.index][4]
            pedal_status = train_xy[selected_sample.index]['articulation_loss_weight']
            edges = train_xy[selected_sample.index]['graph']

            num_slice = len(selected_sample.slice_indexes)
            selected_idx = random.randrange(0,num_slice)
            slice_idx = selected_sample.slice_indexes[selected_idx]

            if model.config.is_graph:
                graphs = graph.edges_to_matrix_short(edges, slice_idx, model.config)
            else:
                graphs = None

            key_lists = [0]
            key = 0
            for i in range(args.num_key_augmentation):
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
                    utils.batch_train_run(training_data, model=train_model, args=args, optimizer=optimizer)
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
            # print("Remaining samples: ", sum([len(x.slice_indexes) for x in remaining_samples if x.slice_indexes]))
        print('Epoch [{}/{}], Loss - Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}'
              .format(epoch + 1, args.num_epochs, np.mean(tempo_loss_total), np.mean(vel_loss_total),
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
            test_x = xy_tuple['input_data']
            test_y = xy_tuple['output_data']
            note_locations = xy_tuple['note_location']
            align_matched = xy_tuple['align_matched']
            # TODO: need check
            pedal_status = xy_tuple['articulation_loss_weight']
            edges = xy_tuple['graph']
            graphs = graph.edges_to_matrix(edges, len(test_x), model.config)
            # if args.loss == 'CE':
            #     test_y = categorize_value_to_vector(test_y, bins)

            batch_x, batch_y = utils.handle_data_in_tensor(test_x, test_y, model.config, device)
            batch_x = batch_x.view(1, -1, model.config.input_size)
            batch_y = batch_y.view(1, -1, model.config.output_size)
            # input_y = th.Tensor(prev_feature).view((1, -1, TOTAL_OUTPUT)).to(device)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(device)
            pedal_status = th.Tensor(pedal_status).view(1,-1,1).to(device)
            outputs, total_z = utils.run_model_in_steps(batch_x, batch_y, args, graphs, note_locations, model, device)

            # valid_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], batch_y[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], align_matched)
            if model.config.hierarchy_level and not model.config.is_dependent:
                if model.config.hierarchy_level == 'measure':
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif model.config.hierarchy_level == 'beat':
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

                dev_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                trill_loss = th.zeros(1)

                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
            elif model.config.is_trill:
                trill_bool = batch_x[:,:, const.is_trill_index_concated] == 1
                trill_bool = trill_bool.float().view(1,-1,1).to(device)
                trill_loss = criterion(outputs, batch_y, trill_bool)

                tempo_loss = th.zeros(1)
                vel_loss = th.zeros(1)
                dev_loss = th.zeros(1)
                articul_loss = th.zeros(1)
                pedal_loss = th.zeros(1)
                kld_loss = th.zeros(1)
                kld_loss_total.append(kld_loss.item())

            else:
                valid_loss, tempo_loss, vel_loss, dev_loss, articul_loss, pedal_loss = utils.cal_loss_by_output_type(outputs, batch_y, align_matched, pedal_status, args, model.config, note_locations, 0)
                for z in total_z:
                    perform_mu, perform_var = z
                    kld_loss = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                    kld_loss_total.append(kld_loss.item())
                trill_loss = th.zeros(1)

            # valid_loss_total.append(valid_loss.item())
            tempo_loss_total.append(tempo_loss.item())
            vel_loss_total.append(vel_loss.item())
            deviation_loss_total.append(dev_loss.item())
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

        if model.config.is_trill:
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

'''
def test(args,
         model,
         TRILL_model,
         device,
         param):
    # TODO: seperate validation / test / inference.
    if os.path.isfile('prime_' + args.modelCode + args.resume):
        print("=> loading checkpoint '{}'".format(args.modelCode + args.resume))
        # model_codes = ['prime', 'trill']
        filename = 'prime_' + args.modelCode + args.resume
        print('device is ', args.device)
        th.cuda.set_device(args.device)
        if th.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        checkpoint = th.load(filename, map_location=map_location)
        # args.start_epoch = checkpoint['epoch']
        # best_valid_loss = checkpoint['best_valid_loss']
        model.load_state_dict(checkpoint['state_dict'])
        # model.num_graph_iteration = 10
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        # NUM_UPDATED = checkpoint['training_step']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # trill_filename = args.trillCode + args.resume
        trill_filename = args.trillCode + '_best.pth.tar'
        checkpoint = th.load(trill_filename, map_location=map_location)
        TRILL_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(trill_filename, checkpoint['epoch']))

        if args.in_hier:
            HIER_model_PARAM = param.load_parameters(args.hierCode + '_param')
            HIER_model = nnModel.HAN_Integrated(HIER_model_PARAM, device, True).to(device)
            filename = 'prime_' + args.hierCode + args.resume
            checkpoint = th.load(filename, map_location=device)
            HIER_model.load_state_dict(checkpoint['state_dict'])
            print("=> high-level model loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    model.is_teacher_force = False
    
    # Suggestion: move inference-like mode to inference.py
    if args.sessMode == 'test':
        random.seed(0)
        inference.load_file_and_generate_performance(args.testPath, args)
    elif args.sessMode=='testAll':
        path_list = const.emotion_data_path
        emotion_list = const.emotion_key_list
        perform_z_by_list = dataset.encode_all_emotionNet_data(path_list, emotion_list)
        test_list = const.test_piece_list
        for piece in test_list:
            path = './test_pieces/' + piece[0] + '/'
            composer = piece[1]
            if len(piece) == 3:
                start_tempo = piece[2]
            else:
                start_tempo = 0
            for perform_z_pair in perform_z_by_list:
                inference.load_file_and_generate_performance(path, composer, z=perform_z_pair, start_tempo=start_tempo)
            inference.load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)
    elif args.sessMode == 'testAllzero':
        test_list = const.test_piece_list
        for piece in test_list:
            path = './test_pieces/' + piece[0] + '/'
            composer = piece[1]
            if len(piece) == 3:
                start_tempo = piece[2]
            else:
                start_tempo = 0
            random.seed(0)
            inference.load_file_and_generate_performance(path, composer, z=0, start_tempo=start_tempo)

    elif args.sessMode == 'encode':
        perform_z, qpm_primo = dataset.load_file_and_encode_style(args.testPath, args.perfName, args.composer)
        print(perform_z)
        with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
            pickle.dump(perform_z, f, protocol=2)

    elif args.sessMode =='evaluate':
        test_data_name = args.dataName + "_test.dat"
        if not os.path.isfile(test_data_name):
            test_data_name = '/mnt/ssd1/jdasam_data/' + test_data_name
        with open(test_data_name, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            # p = u.load()
            # complete_xy = pickle.load(f)
            complete_xy = u.load()

        tempo_loss_total = []
        vel_loss_total = []
        deviation_loss_total = []
        trill_loss_total = []
        articul_loss_total = []
        pedal_loss_total = []
        kld_total = []

        prev_perf_x = complete_xy[0][0]
        prev_perfs_worm_data = []
        prev_reconstructed_worm_data = []
        prev_zero_predicted_worm_data = []
        piece_wise_loss = []
        human_correlation_total = []
        human_correlation_results = xml_matching.CorrelationResult()
        model_correlation_total = []
        model_correlation_results = xml_matching.CorrelationResult()
        zero_sample_correlation_total = []
        zero_sample_correlation_results= xml_matching.CorrelationResult()



        for xy_tuple in complete_xy:
            current_perf_index = complete_xy.index(xy_tuple)
            test_x = xy_tuple[0]
            test_y = xy_tuple[1]
            note_locations = xy_tuple[2]
            align_matched = xy_tuple[3]
            pedal_status = xy_tuple[4]
            edges = xy_tuple[5]
            graphs = graph.edges_to_matrix(edges, len(test_x), model.config)
            if args.loss == 'CE':
                test_y = categorize_value_to_vector(test_y, bins)

            if xml_matching.check_feature_pair_is_from_same_piece(prev_perf_x, test_x):
                piece_changed = False
                # current_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(test_y, note_locations=note_locations, momentum=0.2)
                # for prev_worm in prev_perfs_worm_data:
                #     tempo_r, _ = xml_matching.cal_correlation(current_perf_worm_data[0], prev_worm[0])
                #     dynamic_r, _ = xml_matching.cal_correlation(current_perf_worm_data[1], prev_worm[1])
                #     human_correlation_results.append_result(tempo_r, dynamic_r)
                # prev_perfs_worm_data.append(current_perf_worm_data)
            else:
                piece_changed = True

            if piece_changed or current_perf_index == len(complete_xy)-1:
                prev_perf_x = test_x
                if piece_wise_loss:
                    piece_wise_loss_mean = np.mean(np.asarray(piece_wise_loss), axis=0)
                    tempo_loss_total.append(piece_wise_loss_mean[0])
                    vel_loss_total.append(piece_wise_loss_mean[1])
                    deviation_loss_total.append(piece_wise_loss_mean[2])
                    articul_loss_total.append(piece_wise_loss_mean[3])
                    pedal_loss_total.append(piece_wise_loss_mean[4])
                    trill_loss_total.append(piece_wise_loss_mean[5])
                    kld_total.append(piece_wise_loss_mean[6])
                piece_wise_loss = []

                # human_correlation_total.append(human_correlation_results)
                # human_correlation_results = xml_matching.CorrelationResult()
                #
                # for predict in prev_reconstructed_worm_data:
                #     for human in prev_perfs_worm_data:
                #         tempo_r, _ = xml_matching.cal_correlation(predict[0], human[0])
                #         dynamic_r, _ = xml_matching.cal_correlation(predict[1], human[1])
                #         model_correlation_results.append_result(tempo_r, dynamic_r)
                #
                # model_correlation_total.append(model_correlation_results)
                # model_correlation_results = xml_matching.CorrelationResult()
                #
                # for zero in prev_zero_predicted_worm_data:
                #     for human in prev_perfs_worm_data:
                #         tempo_r, _ = xml_matching.cal_correlation(zero[0], human[0])
                #         dynamic_r, _ = xml_matching.cal_correlation(zero[1], human[1])
                #         zero_sample_correlation_results.append_result(tempo_r, dynamic_r)
                #
                # zero_sample_correlation_total.append(zero_sample_correlation_results)
                # zero_sample_correlation_results = xml_matching.CorrelationResult()
                #
                # prev_reconstructed_worm_data = []
                # prev_zero_predicted_worm_data = []
                # prev_perfs_worm_data = []
                #
                # print('Human Correlation: ', human_correlation_total[-1])
                # print('Reconst Correlation: ', model_correlation_total[-1])
                # print('Zero Sampled Correlation: ', zero_sample_correlation_total[-1])

            batch_x, batch_y = handle_data_in_tensor(test_x, test_y, hierarchy_test=IN_HIER)
            align_matched = th.Tensor(align_matched).view(1, -1, 1).to(device)
            pedal_status = th.Tensor(pedal_status).view(1, -1, 1).to(device)

            if IN_HIER:
                batch_x = batch_x.view((1, -1, HIER_model.input_size))
                hier_y = batch_y[0].view(1, -1, HIER_model.output_size)
                hier_outputs, _ = run_model_in_steps(batch_x, hier_y, graphs, note_locations, model=HIER_model)
                if HIER_MEAS:
                    hierarchy_numbers = [x.measure for x in note_locations]
                elif HIER_BEAT:
                    hierarchy_numbers = [x.beat for x in note_locations]
                hier_outputs_spanned = HIER_model.span_beat_to_note_num(hier_outputs, hierarchy_numbers, batch_x.shape[1], 0)
                input_concat = th.cat((batch_x, hier_outputs_spanned),2)
                batch_y = batch_y[1].view(1,-1, model.output_size)
                outputs, perform_z = run_model_in_steps(input_concat, batch_y, graphs, note_locations, model=model)

                # make another prediction with random sampled z
                zero_hier_outputs, _ = run_model_in_steps(batch_x, hier_y, graphs, note_locations, model=HIER_model,
                                                        initial_z='zero')
                zero_hier_spanned = HIER_model.span_beat_to_note_num(zero_hier_outputs, hierarchy_numbers, batch_x.shape[1], 0)
                zero_input_concat = th.cat((batch_x, zero_hier_spanned),2)
                zero_prediction, _ = run_model_in_steps(zero_input_concat, batch_y, graphs, note_locations, model=model)

            else:
                batch_x = batch_x.view((1, -1, NUM_INPUT))
                batch_y = batch_y.view((1, -1, NUM_OUTPUT))
                outputs, perform_z = run_model_in_steps(batch_x, batch_y, graphs, note_locations)

                # make another prediction with random sampled z
                zero_prediction, _ = run_model_in_steps(batch_x, batch_y, graphs, note_locations, model=model,
                                                     initial_z='zero')

            output_as_feature = outputs.view(-1, NUM_OUTPUT).cpu().numpy()
            predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(output_as_feature, note_locations,
                                                                                momentum=0.2)
            zero_prediction_as_feature = zero_prediction.view(-1, NUM_OUTPUT).cpu().numpy()
            zero_predicted_perf_worm_data = perf_worm.cal_tempo_and_velocity_by_beat(zero_prediction_as_feature, note_locations,
                                                                                     momentum=0.2)

            prev_reconstructed_worm_data.append(predicted_perf_worm_data)
            prev_zero_predicted_worm_data.append(zero_predicted_perf_worm_data)

            # for prev_worm in prev_perfs_worm_data:
            #     tempo_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[0], prev_worm[0])
            #     dynamic_r, _ = xml_matching.cal_correlation(predicted_perf_worm_data[1], prev_worm[1])
            #     model_correlation_results.append_result(tempo_r, dynamic_r)
            # print('Model Correlation: ', model_correlation_results)

            # valid_loss = criterion(outputs[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], batch_y[:,:,const.NUM_TEMPO_PARAM:-const.num_trill_param], align_matched)
            if model.is_baseline:
                tempo_loss = criterion(outputs[:, :, 0], batch_y[:, :, 0], align_matched)
            else:
                tempo_loss = cal_tempo_loss_in_beat(outputs, batch_y, note_locations, 0)
            if args.loss == 'CE':
                vel_loss = criterion(outputs[:, :, const.NUM_TEMPO_PARAM:const.NUM_TEMPO_PARAM + len(bins[1])],
                                     batch_y[:, :, const.NUM_TEMPO_PARAM:const.NUM_TEMPO_PARAM + len(bins[1])], align_matched)
                deviation_loss = criterion(
                    outputs[:, :, const.NUM_TEMPO_PARAM + len(bins[1]):const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2])],
                    batch_y[:, :, const.NUM_TEMPO_PARAM + len(bins[1]):const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2])])
                pedal_loss = criterion(
                    outputs[:, :, const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2]):-const.num_trill_param],
                    batch_y[:, :, const.NUM_TEMPO_PARAM + len(bins[1]) + len(bins[2]):-const.num_trill_param])
                trill_loss = criterion(outputs[:, :, -const.num_trill_param:], batch_y[:, :, -const.num_trill_param:])
            else:
                vel_loss = criterion(outputs[:, :, const.VEL_PARAM_IDX], batch_y[:, :, const.VEL_PARAM_IDX], align_matched)
                deviation_loss = criterion(outputs[:, :, const.DEV_PARAM_IDX], batch_y[:, :, const.DEV_PARAM_IDX],
                                           align_matched)
                articul_loss = criterion(outputs[:, :, const.PEDAL_PARAM_IDX], batch_y[:, :, const.PEDAL_PARAM_IDX],
                                         pedal_status)
                pedal_loss = criterion(outputs[:, :, const.PEDAL_PARAM_IDX + 1:], batch_y[:, :, const.PEDAL_PARAM_IDX + 1:],
                                       align_matched)
                trill_loss = th.zeros(1)

            piece_kld = []
            for z in perform_z:
                perform_mu, perform_var = z
                kld = -0.5 * th.sum(1 + perform_var - perform_mu.pow(2) - perform_var.exp())
                piece_kld.append(kld)
            piece_kld = th.mean(th.stack(piece_kld))

            piece_wise_loss.append((tempo_loss.item(), vel_loss.item(), deviation_loss.item(), articul_loss.item(), pedal_loss.item(), trill_loss.item(), piece_kld.item()))



        mean_tempo_loss = np.mean(tempo_loss_total)
        mean_vel_loss = np.mean(vel_loss_total)
        mean_deviation_loss = np.mean(deviation_loss_total)
        mean_articul_loss = np.mean(articul_loss_total)
        mean_pedal_loss = np.mean(pedal_loss_total)
        mean_trill_loss = np.mean(trill_loss_total)
        mean_kld_loss = np.mean(kld_total)

        mean_valid_loss = (mean_tempo_loss + mean_vel_loss + mean_deviation_loss / 2 + mean_pedal_loss * 8) / 10.5

        print("Test Loss= {:.4f} , Tempo: {:.4f}, Vel: {:.4f}, Deviation: {:.4f}, Articulation: {:.4f}, Pedal: {:.4f}, Trill: {:.4f}, KLD: {:.4f}"
              .format(mean_valid_loss, mean_tempo_loss, mean_vel_loss,
                      mean_deviation_loss, mean_articul_loss, mean_pedal_loss, mean_trill_loss, mean_kld_loss))
        # num_piece = len(model_correlation_total)
        # for i in range(num_piece):
        #     if len(human_correlation_total) > 0:
        #         print('Human Correlation: ', human_correlation_total[i])
        #         print('Model Correlation: ', model_correlation_total[i])


    elif args.sessMode == 'correlation':
        with open('selected_corr_30.dat', "rb") as f:
            u = pickle._Unpickler(f)
            selected_corr = u.load()
        model_cor = []
        for piece_corr in selected_corr:
            if piece_corr is None or piece_corr==[]:
                continue
            path = piece_corr[0].path_name
            composer_name = copy.copy(path).split('/')[1]
            output_features = load_file_and_generate_performance(path, composer_name, 'zero', return_features=True)
            for slice_corr in piece_corr:
                slc_idx = slice_corr.slice_index
                sliced_features = output_features[slc_idx[0]:slc_idx[1]]
                tempos, dynamics = perf_worm.cal_tempo_and_velocity_by_beat(sliced_features)
                model_correlation_results = xml_matching.CorrelationResult()
                model_correlation_results.path_name = slice_corr.path_name
                model_correlation_results.slice_index = slice_corr.slice_index
                human_tempos = slice_corr.tempo_features
                human_dynamics = slice_corr.dynamic_features
                for i in range(slice_corr.num_performance):
                    tempo_r, _ = xml_matching.cal_correlation(tempos, human_tempos[i])
                    dynamic_r, _ = xml_matching.cal_correlation(dynamics, human_dynamics[i])
                    model_correlation_results._append_result(tempo_r, dynamic_r)
                print(model_correlation_results)
                model_correlation_results.tempo_features = copy.copy(slice_corr.tempo_features)
                model_correlation_results.dynamic_features = copy.copy(slice_corr.dynamic_features)
                model_correlation_results.tempo_features.append(tempos)
                model_correlation_results.dynamic_features.append(dynamics)

                save_name = 'test_plot/' + path.replace('chopin_cleaned/', '').replace('/', '_', 10) + '_note{}-{}_by_{}.png'.format(slc_idx[0], slc_idx[1], args.modelCode)
                perf_worm.plot_human_model_features_compare(model_correlation_results.tempo_features, save_name)
                model_cor.append(model_correlation_results)

        with open(args.modelCode + "_cor.dat", "wb") as f:
            pickle.dump(model_cor, f, protocol=2)
'''

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
