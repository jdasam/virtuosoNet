import numpy as np
import torch as th

def load_file_and_encode_style(path, perf_name, composer_name):
    test_x, test_y, edges, note_locations = xml_matching.read_score_perform_pair(
        path, perf_name, composer_name, MEANS, STDS)
    qpm_primo = test_x[0][4]

    test_x, test_y = handle_data_in_tensor(
        test_x, test_y, hierarchy_test=IN_HIER)
    edges = edges_to_matrix(edges, test_x.shape[0])

    if IN_HIER:
        test_x = test_x.view((1, -1, HIER_MODEL.input_size))
        hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
        perform_z_high = encode_performance_style_vector(
            test_x, hier_y, edges, note_locations, model=HIER_MODEL)
        hier_outputs, _ = run_model_in_steps(
            test_x, hier_y, edges, note_locations, model=HIER_MODEL)
        if HIER_MEAS:
            hierarchy_numbers = [x.measure for x in note_locations]
        elif HIER_BEAT:
            hierarchy_numbers = [x.beat for x in note_locations]
        hier_outputs_spanned = HIER_MODEL.span_beat_to_note_num(
            hier_outputs, hierarchy_numbers, test_x.shape[1], 0)
        input_concat = torch.cat((test_x, hier_outputs_spanned), 2)
        batch_y = test_y[1].view(1, -1, MODEL.output_size)
        perform_z_note = encode_performance_style_vector(
            input_concat, batch_y, edges, note_locations, model=MODEL)
        perform_z = [perform_z_high, perform_z_note]

    else:
        batch_x = test_x.view((1, -1, NUM_INPUT))
        batch_y = test_y.view((1, -1, NUM_OUTPUT))
        perform_z = encode_performance_style_vector(
            batch_x, batch_y, edges, note_locations)
        perform_z = [perform_z]

    return perform_z, qpm_primo


#>>>>>>>>>>>>>> maybe to be removed
def load_all_file_and_encode_style(parsed_data, measure_only=False, emotion_data=False):
    total_z = []
    perf_name_list = []
    num_piece = len(parsed_data[0])
    for i in range(num_piece):
        piece_test_x = parsed_data[0][i]
        piece_test_y = parsed_data[1][i]
        piece_edges = parsed_data[2][i]
        piece_note_locations = parsed_data[3][i]
        piece_perf_name = parsed_data[4][i]
        num_perf = len(piece_test_x)
        if num_perf == 0:
            continue
        piece_z = []
        for j in range(num_perf):
            # test_x, test_y, edges, note_locations, perf_name = perf
            if measure_only:
                test_x, test_y = handle_data_in_tensor(
                    piece_test_x[j], piece_test_y[j], hierarchy_test=IN_HIER)
                edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
                test_x = test_x.view((1, -1, HIER_MODEL.input_size))
                hier_y = test_y[0].view(1, -1, HIER_MODEL.output_size)
                perform_z_high = encode_performance_style_vector(
                    test_x, hier_y, edges, piece_note_locations[j], model=HIER_MODEL)
            else:
                test_x, test_y = handle_data_in_tensor(
                    piece_test_x[j], piece_test_y[j], hierarchy_test=False)
                edges = edges_to_matrix(piece_edges[j], test_x.shape[0])
                test_x = test_x.view((1, -1, MODEL.input_size))
                test_y = test_y.view(1, -1, MODEL.output_size)
                perform_z_high = encode_performance_style_vector(test_x, test_y, edges, piece_note_locations[j],
                                                                 model=MODEL)
            # perform_z_high = perform_z_high.reshape(-1).cpu().numpy()
            # piece_z.append(perform_z_high)
            # perf_name_list.append(piece_perf_name[j])

            perform_z_high = [z.reshape(-1).cpu().numpy()
                              for z in perform_z_high]
            piece_z += perform_z_high
            perf_name_list += [piece_perf_name[j]] * len(perform_z_high)
        if emotion_data:
            for i, name in enumerate(piece_perf_name):
                if name[-2:] == 'E1':
                    or_idx = i
                    break
            or_z = piece_z.pop(or_idx)
            piece_z = np.asarray(piece_z)
            piece_z -= or_z
            perf_name_list.pop(-(5-or_idx))
        else:
            piece_z = np.asarray(piece_z)
            average_piece_z = np.average(piece_z, axis=0)
            piece_z -= average_piece_z
        total_z.append(piece_z)
    total_z = np.concatenate(total_z, axis=0)
    return total_z, perf_name_list
#<<<<<<<<<<<<<< 


def encode_all_emotionNet_data(path_list, style_keywords):
    perform_z_by_emotion = []
    perform_z_list_by_subject = []
    qpm_list_by_subject = []
    num_style = len(style_keywords)
    if IN_HIER:
        num_model = 2
    else:
        num_model = 1
    for pair in path_list:
        subject_num = pair[2]
        for sub_idx in range(subject_num):
            indiv_perform_z = []
            indiv_qpm = []
            path = cons.emotion_folder_path + pair[0] + '/'
            composer_name = pair[1]
            for key in style_keywords:
                perf_name = key + '_sub' + str(sub_idx+1)
                perform_z_li, qpm_primo = load_file_and_encode_style(
                    path, perf_name, composer_name)
                perform_z_li = [torch.mean(torch.stack(z), 0, True)
                                for z in perform_z_li]
                indiv_perform_z.append(perform_z_li)
                indiv_qpm.append(qpm_primo)
            for i in range(1, num_style):
                for j in range(num_model):
                    indiv_perform_z[i][j] = indiv_perform_z[i][j] - \
                        indiv_perform_z[0][j]
                indiv_qpm[i] = indiv_qpm[i] - indiv_qpm[0]
            perform_z_list_by_subject.append(indiv_perform_z)
            qpm_list_by_subject.append(indiv_qpm)
    for i in range(num_style):
        z_by_models = []
        for j in range(num_model):
            emotion_mean_z = []
            for z_list in perform_z_list_by_subject:
                emotion_mean_z.append(z_list[i][j])
            mean_perform_z = torch.mean(torch.stack(emotion_mean_z), 0, True)
            z_by_models.append(mean_perform_z)
        if i is not 0:
            emotion_qpm = []
            for qpm_change in qpm_list_by_subject:
                emotion_qpm.append(qpm_change[i])
            mean_qpm_change = np.mean(emotion_qpm)
        else:
            mean_qpm_change = 0
        print(style_keywords[i], z_by_models, mean_qpm_change)
        perform_z_by_emotion.append(
            {'z': z_by_models, 'key': style_keywords[i], 'qpm': mean_qpm_change})

    return perform_z_by_emotion
    # with open(args.testPath + args.perfName + '_style' + '.dat', 'wb') as f:
    #     pickle.dump(mean_perform_z, f, protocol=2)
