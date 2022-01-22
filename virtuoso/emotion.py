import torch as th
from .utils import batch_to_device
from . import style_analysis as sty
import wandb
import numpy as np
import pandas as pd
import plotly.express as px

def get_style_from_emotion_data(model, emotion_loader, device):
  total_perform_z = []
  with th.no_grad():
    for i, batch in enumerate(emotion_loader):
      origin_data = emotion_loader.dataset.data[i*5]
      perform_z_set = {'score_path':origin_data['score_path'], 'perform_path':origin_data['perform_path']}
      batch_x, batch_y, _, _, note_locations, _, _, edges = batch_to_device(batch, device)
      perform_z_tensor = model.encode_style(batch_x, batch_y, edges, note_locations)
      perform_z_np_array = perform_z_tensor.detach().cpu().numpy()
      for j in range(5):
        perform_z_set[f'E{j+1}'] = perform_z_np_array[j]
      total_perform_z.append(perform_z_set)
      # for j, perform in enumerate(batch):
      #     batch_x, batch_y, note_locations, _, _, edges = batch_to_device(perform, device)
      #     perform_z_list = model.encode_style(batch_x, batch_y, edges, note_locations)
      #     perform_z_set[f'E{j+1}'] = [x.detach().cpu().numpy()[0] for x in perform_z_list]
      # total_perform_z.append(perform_z_set)
  return total_perform_z


def validate_style_with_emotion_data(model, emotion_loader, device, out_dir, iteration, send_wandb_log=True):
    total_perform_z = get_style_from_emotion_data(model, emotion_loader, device)
    abs_confusion, abs_accuracy, norm_confusion, norm_accuracy = sty.get_classification_error_with_svm(total_perform_z, emotion_loader.dataset.cross_valid_split)
    # for dim_reduc_type in ("pca", "umap"):
    for dim_reduc_type in ["pca"]:
      embedded_z, embedded_normalized_z = sty.embedd_dim_reduction_of_emotion_dataset(total_perform_z, dim_reduction_type=dim_reduc_type)
      save_name = out_dir / f"emotion_{dim_reduc_type}_it{iteration}.png"
      sty.draw_tsne_for_emotion_data(embedded_z, save_name)
      save_name = out_dir / f"emotion_{dim_reduc_type}_norm_it{iteration}.png"
      sty.draw_tsne_for_emotion_data(embedded_normalized_z, save_name)
      if send_wandb_log:
        type_names = ["abs", "norm"]
        for i, selected_embedding in enumerate([embedded_z, embedded_normalized_z]):
          z_for_df = selected_embedding.transpose(0,2,1,3).reshape(embedded_z.shape[0]*embedded_z.shape[2]*5,2)
          df = pd.DataFrame(z_for_df)
          emotion_index = np.tile(np.asarray([1,2,3,4,5]), embedded_z.shape[0]*embedded_z.shape[2])
          df['emotion_id'] = emotion_index
          fig = px.scatter(df, x=0, y=1, color="emotion_id")
          wandb.log({f"emotion_embedding_{dim_reduc_type}_{type_names[i]}":fig}, step=iteration)

    return total_perform_z, abs_confusion, abs_accuracy, norm_confusion, norm_accuracy