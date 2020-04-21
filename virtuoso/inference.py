import torch as th
import argparse
from . import graph
from . import utils
from . import constants as cons
from . import dataset

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str,
                    default='train', help="train or test or testAll")
parser.add_argument("-path", "--testPath", type=str,
                    default="./test_pieces/bps_5_1/", help="folder path of test mat")
parser.add_argument("-tempo", "--startTempo", type=int,
                    default=0, help="start tempo. zero to use xml first tempo")


def load_file_and_generate_performance(model, path_name, args, return_features=False):
    '''
    :param model: A single NN model or tuple of a measure-level model and note-level model
    :param path_name:
    :param args:
    :param return_features:
    :return:
    '''
    if isinstance(model, tuple):
        # if input variable 'model' consists of hier_model and note_model
        hier_model = model[0]
        model = model[1]

    composer = args.composer
    z = args.latent
    start_tempo = args.startTempo
    vel_pair = (int(args.velocity.split(',')[0]), int(
        args.velocity.split(',')[1]))
    test_x, xml_notes, xml_doc, edges, note_location = dataset.read_xml_to_array(path_name, feat_stats,
                                                                                       start_tempo, composer,
                                                                                       vel_pair)

    batch_x = th.Tensor(test_x)
    num_notes = len(test_x)
    input_y = th.zeros(1, num_notes, model.config.output_size).to(model.device)

    if type(z) is dict:
        initial_z = z['z']
        qpm_change = z['qpm']
        z = z['key']
        batch_x[:, QPM_PRIMO_IDX] = batch_x[:, QPM_PRIMO_IDX] + qpm_change
    else:
        initial_z = 'zero'

    if model.config.is_dependent:
        batch_x = batch_x.to(model.device).view(1, -1, hier_model.config.input_size)
        edges = graph.edges_to_matrix(edges, batch_x.shape[1], model.config)
        model.config.is_teacher_force = False
        if type(initial_z) is list:
            hier_z = initial_z[0]
            final_z = initial_z[1]
        else:
            # hier_z = [z] * HIER_MODEL_PARAM.encoder.size
            hier_z = 'zero'
            final_z = initial_z
        hier_input_y = th.zeros(1, num_notes, hier_model.config.output_size)
        hier_output, _ = utils.run_model_in_steps(
            batch_x, hier_input_y, args, edges, note_locations, initial_z=hier_z, model=hier_model)
        if 'measure' in args.hierCode:
            hierarchy_numbers = [x.measure for x in note_locations]
        else:
            hierarchy_numbers = [x.section for x in note_locations]
        hier_output_spanned = hier_model.span_beat_to_note_num(
            hier_output, hierarchy_numbers, len(test_x), 0)
        combined_x = th.cat((batch_x, hier_output_spanned), 2)
        prediction, _ = utils.run_model_in_steps(
            combined_x, input_y, args, edges, note_locations, initial_z=final_z, model=model)
    else:
        if type(initial_z) is list:
            initial_z = initial_z[0]
        batch_x = batch_x.to(model.device).view(1, -1, model.config.input_size)
        edges = graph.edges_to_matrix(edges, batch_x.shape[1], model.config)
        prediction, _ = utils.run_model_in_steps(
            batch_x, input_y, args, edges, note_locations, initial_z=initial_z, model=model)

    trill_batch_x = th.cat((batch_x, prediction), 2)
    trill_prediction, _ = utils.run_model_in_steps(trill_batch_x, th.zeros(
        1, num_notes, cons.num_trill_param), graph, note_locations, model=TRILL_MODEL)

    prediction = torch.cat((prediction, trill_prediction), 2)
    prediction = utils.scale_model_prediction_to_original(prediction, feat_stats)

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
