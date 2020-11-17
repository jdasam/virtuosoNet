from pickle import load
import numpy as np
import argparse
import torch
import copy

from .constants import *
from .pyScoreParser.data_class import ScoreData
from .pyScoreParser.feature_extraction import ScoreExtractor
from .pyScoreParser.data_for_training import convert_feature_to_VirtuosoNet_format
from .pyScoreParser.feature_to_performance import apply_tempo_perform_features, xml_notes_to_midi
from .pyScoreParser.performanceWorm import plot_performance_worm
from .pyScoreParser.midi_utils.midi_utils import save_midi_notes_as_piano_midi
from pathlib import Path
from .utils import load_weight
from . import graph


def inference(args, model, stats, input_keys, output_keys, device):
    model = load_weight(model, args.checkpoint)
    model.eval()
    # load score
    score, input, edges, note_locations = get_input_from_xml(args.xml_path, args.composer, input_keys, args.graph_keys, stats, device)
    with torch.no_grad():
        outputs, perform_mu, perform_var, total_out_list = model(input, None, edges, note_locations, initial_z='zero')

    save_output_as_midi(outputs, args.output_path, args.xml_path, args.model_code, score, output_keys, stats, note_locations, args.boolPedal, args.disklavier)



def save_output_as_midi(model_outputs, output_path, xml_path, model_code, score, output_keys, stats, note_locations, bool_pedal=False, disklavier=False):
    outputs = scale_model_prediction_to_original(model_outputs, output_keys, stats)
    output_features = model_prediction_to_feature(outputs, output_keys)

    xml_notes = apply_tempo_perform_features(score, output_features, start_time=0.5, predicted=True)

    save_path = output_path / f"{xml_path.parent.stem}_{xml_path.stem}_by_{model_code}.mid"
    if not output_path.exists():
        output_path.mkdir()
    output_midi, midi_pedals = xml_notes_to_midi(xml_notes)

    plot_performance_worm(output_features, note_locations['beat'], save_path.with_suffix('.png'))
    save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_path,
                                               bool_pedal=bool_pedal, disklavier=disklavier)


def get_input_from_xml(xml_path, composer, input_keys, graph_keys, stats, device='cuda'):
    score = ScoreData(xml_path, None, composer, read_xml_only=True)
    feature_extractor = ScoreExtractor(input_keys)
    input_features = feature_extractor.extract_score_features(score)
    input, _, _ = convert_feature_to_VirtuosoNet_format(input_features, stats, output_keys=[], meas_keys=[])
    input = torch.Tensor(input).unsqueeze(0).to(device)
    edges = graph.edges_to_matrix(score.notes_graph, score.num_notes, graph_keys).to(device)
    note_locations = {
            'beat': torch.Tensor(input_features['note_location']['beat']).type(torch.int32),
            'measure': torch.Tensor(input_features['note_location']['measure']).type(torch.int32),
            'section': torch.Tensor(input_features['note_location']['section']).type(torch.int32),
            'voice': torch.Tensor(input_features['note_location']['voice']).type(torch.int32),
    }
    return score, input, edges, note_locations

def scale_model_prediction_to_original(prediction, output_keys, stats):
    prediction = np.squeeze(prediction.cpu().numpy())
    idx = 0
    for key in output_keys:
        prediction[:,idx]  *= stats[key]['stds']
        prediction[:,idx]  += stats[key]['mean']
        idx += 1 
    return prediction


def model_prediction_to_feature(prediction, output_keys):
    output_features = {}
    idx = 0
    for key in output_keys:
        output_features[key] = prediction[:,idx]
        idx += 1
    
    return output_features



def load_file_and_generate_performance(model, path_name, args, trill_model, hier_model=None, return_features=False):
    composer=args.composer
    z=args.latent
    start_tempo=args.startTempo
    path_name = Path(path_name)
    if path_name.suffix not in ('xml', 'musicxml'):
        path_name = path_name / 'xml.xml'
        if not path_name.exists():
            path_name = path_name.parent / 'musicxml_cleaned.musicxml'

    score_features = model.score_feature_keys
    score_data = ScoreData(path_name)
    feature_extractor = ScoreExtractor()
    test_x = feature_extractor.extract_score_features(score_data)
    test_x, xml_notes, xml_doc, edges, note_locations = xml_matching.read_xml_to_array(path_name, means, stds,
                                                                                       start_tempo, composer,
                                                                                       vel_pair)
    batch_x = torch.Tensor(test_x)
    num_notes = len(test_x)
    input_y = torch.zeros(1, num_notes, NUM_OUTPUT).to(DEVICE)

    if type(z) is dict:
        initial_z = z['z']
        qpm_change = z['qpm']
        z = z['key']
        batch_x[:, QPM_PRIMO_IDX] = batch_x[:, QPM_PRIMO_IDX] + qpm_change
    else:
        initial_z = 'zero'

    if IN_HIER:
        batch_x = batch_x.to(DEVICE).view(1, -1, hier_model.input_size)
        graph = edges_to_matrix(edges, batch_x.shape[1])
        MODEL.is_teacher_force = False
        if type(initial_z) is list:
            hier_z = initial_z[0]
            final_z = initial_z[1]
        else:
            # hier_z = [z] * HIER_MODEL_PARAM.encoder.size
            hier_z = 'zero'
            final_z = initial_z
        hier_input_y = torch.zeros(1, num_notes, hier_model.output_size)
        hier_output, _ = run_model_in_steps(
            batch_x, hier_input_y, graph, note_locations, initial_z=hier_z, model=hier_model)
        if 'measure' in args.hierCode:
            hierarchy_numbers = [x.measure for x in note_locations]
        else:
            hierarchy_numbers = [x.section for x in note_locations]
        hier_output_spanned = hier_model.span_beat_to_note_num(
            hier_output, hierarchy_numbers, len(test_x), 0)
        combined_x = torch.cat((batch_x, hier_output_spanned), 2)
        prediction, _ = run_model_in_steps(
            combined_x, input_y, graph, note_locations, initial_z=final_z, model=model)
    else:
        if type(initial_z) is list:
            initial_z = initial_z[0]
        batch_x = batch_x.to(DEVICE).view(1, -1, NUM_INPUT)
        graph = edges_to_matrix(edges, batch_x.shape[1])
        prediction, _ = run_model_in_steps(
            batch_x, input_y, graph, note_locations, initial_z=initial_z, model=model)

    trill_batch_x = torch.cat((batch_x, prediction), 2)
    trill_prediction, _ = run_model_in_steps(trill_batch_x, torch.zeros(
        1, num_notes, cons.num_trill_param), graph, note_locations, model=trill_model)

    prediction = torch.cat((prediction, trill_prediction), 2)
    prediction = scale_model_prediction_to_original(prediction, means, stds)

    output_features = xml_matching.model_prediction_to_feature(prediction)
    output_features = xml_matching.add_note_location_to_features(
        output_features, note_locations)
    if return_features:
        return output_features

    output_xml = xml_matching.apply_tempo_perform_features(xml_doc, xml_notes, output_features, start_time=1,
                                                           predicted=True)
    output_midi, midi_pedals = xml_matching.xml_notes_to_midi(output_xml)
    piece_name = path_name.split('/')
    save_name = 'test_result/' + \
        piece_name[-2] + '_by_' + args.modelCode + '_z' + str(z)

    perf_worm.plot_performance_worm(output_features, save_name + '.png')
    xml_matching.save_midi_notes_as_piano_midi(output_midi, midi_pedals, save_name + '.mid',
                                               bool_pedal=args.boolPedal, disklavier=args.disklavier)




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
            HIER_model = model.HAN_Integrated(HIER_model_PARAM, device, True).to(device)
            filename = 'prime_' + args.hierCode + args.resume
            checkpoint = th.load(filename, map_location=device)
            HIER_model.load_state_dict(checkpoint['state_dict'])
            print("=> high-level model loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    model.is_teacher_force = False
    

def inference_old(model, args):
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
            graphs = graph.edges_to_matrix(edges, len(test_x))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--sessMode", type=str,
                        default='train', help="train or test or testAll")
    parser.add_argument("-path", "--test_path", type=str,
                        default="./test_pieces/bps_5_1/", help="folder path of test mat")
    parser.add_argument("-tempo", "--startTempo", type=int,
                        default=0, help="start tempo. zero to use xml first tempo")

    model = load_model
    load_file_and_generate_performance(args.test_path, args)