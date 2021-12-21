import torch as th
from .utils import batch_to_device
from . import style_analysis as sty


def get_style_from_emotion_data(model, emotion_loader, device):
    total_perform_z = []
    with th.no_grad():
        for i, batch in enumerate(emotion_loader):
            origin_data = emotion_loader.dataset.data[i*5]
            perform_z_set = {'score_path':origin_data['score_path'], 'perform_path':origin_data['perform_path']}
            for j, perform in enumerate(batch):
                batch_x, batch_y, note_locations, _, _, edges = batch_to_device(perform, device)
                perform_z_list = model.encode_style(batch_x, batch_y, edges, note_locations)
                perform_z_set[f'E{j+1}'] = [x.detach().cpu().numpy()[0] for x in perform_z_list]
            total_perform_z.append(perform_z_set)
    return total_perform_z


def validate_style_with_emotion_data(model, emotion_loader, device, out_dir, iteration):
    total_perform_z = get_style_from_emotion_data(model, emotion_loader, device)
    abs_confusion, abs_accuracy, norm_confusion, norm_accuracy = sty.get_classification_error_with_svm(total_perform_z, emotion_loader.dataset.cross_valid_split)
    
    tsne_z, tsne_normalized_z = sty.embedd_tsne_of_emotion_dataset(total_perform_z)

    save_name = out_dir / f"emotion_tsne_it{iteration}.png"
    sty.draw_tsne_for_emotion_data(tsne_z, save_name)
    save_name = out_dir / f"emotion_tsne_norm_it{iteration}.png"
    sty.draw_tsne_for_emotion_data(tsne_normalized_z, save_name)

    return total_perform_z, abs_confusion, abs_accuracy, norm_confusion, norm_accuracy