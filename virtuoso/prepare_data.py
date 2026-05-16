from email.policy import default
from .pyScoreParser import data_for_training as dft, data_class
from .utils import load_dat
from pathlib import Path
import argparse



def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_path", 
                      help="dir path to score/perform dataset", 
                      type=str,
                      default='/home/teo/userdata/datasets/chopin_cleaned/')
  parser.add_argument("--emotionset_path", 
                      help="dir path to emotion-cued dataset", 
                      type=str,
                      default='/home/teo/userdata/datasets/EmotionData/'
                      )
  parser.add_argument("-s", "--save_from_beginning", 
                      help='Load MXL and MID to make and save data',
                      type=lambda x: (str(x).lower() == 'true'), 
                      default=False)
  parser.add_argument("--load_feature_only", 
                      help='Load pre-extracted features instead of loading MXL and MID',
                      action='store_true')
  parser.add_argument("--output_dir_path",
                      help="dir path to score/perform dataset", 
                      type=str,
                      default='datasets/main_dataset')
  parser.add_argument("--emotion_output_dir_path",
                      help="dir path to emotion dataset", 
                      type=str,
                      default='datasets/emotion_dataset')
  parser.add_argument("--exclude_long_graces", 
                    help='exclude long graces notes (cadenza-like) during the feature saving',
                    action='store_true')
  parser.add_argument("--exclude_slur",
                    help='exclude slur and beam vectors from input features',
                  action='store_true')
  return parser

def get_vnet_input_keys(args):
  default_keys = dft.VNET_INPUT_KEYS
  if args.exclude_slur:
    dummy = list(default_keys)
    dummy.pop(dummy.index('slur_beam_vec'))
    default_keys = tuple(dummy)
  return default_keys

def clean_emotion_set(emotion_data_dir):
  '''
  Emotion data has to include All 5 emotions for each piece.
  However, some of the performance can be excluded during the alignment process.
  Therefore, this code delete 
  
  '''
  dir_path = Path(emotion_data_dir)
  data_list = list(dir_path.rglob('*.mid.pkl'))
  for pth in data_list:
    if 'E1' in pth.stem:
      for i in range(2,6):
        other_emotion_path = pth.parent / (pth.stem.replace('E1', f'E{i}') +'.pkl')
        if not other_emotion_path.exists():
          try:
            pth.unlink()
          except:
            continue
  data_list = list(dir_path.rglob('*.mid.pkl'))
  data_list.sort()
  for pth in data_list:
    if 'E1' in pth.stem:
      continue
    for i in range(2,6):
      other_emotion_path = pth.parent / (pth.stem.replace(f'E{i}', 'E1') +'.pkl')
      if not other_emotion_path.exists():
        try:
          pth.unlink()
        except:
          continue


if __name__ == "__main__":
  parser = make_parser()
  args = parser.parse_args()
  
  dataset = data_class.YamahaDataset(args.dataset_path, save=args.save_from_beginning, features_only=args.load_feature_only)
  emotion_dataset = data_class.EmotionDataset(args.emotionset_path, save=args.save_from_beginning, features_only=args.load_feature_only)

  if not args.load_feature_only:
    dataset.extract_all_features(save=True)
    emotion_dataset.extract_all_features(save=True)
  
  vnet_input_keys = get_vnet_input_keys(args)

  pair_set = dft.PairDataset(dataset, args.exclude_long_graces)
  pair_set.save_features_for_virtuosoNet(args.output_dir_path, input_key_list=vnet_input_keys)
  emotion_pair_set = dft.PairDataset(emotion_dataset, args.exclude_long_graces)
  emotion_pair_set.feature_stats = load_dat(Path(args.output_dir_path)/'stat.pkl')['stats']
  emotion_pair_set.save_features_for_virtuosoNet(args.emotion_output_dir_path, update_stats=False, input_key_list=vnet_input_keys)
  
  clean_emotion_set(args.emotion_output_dir_path)
  len_emotion_dat = len(list(Path(args.emotion_output_dir_path).rglob('*.mid.pkl')))
  assert len_emotion_dat % 5 == 0