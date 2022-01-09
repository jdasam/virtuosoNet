from virtuoso.pyScoreParser import feature_extraction
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
                      type=lambda x: (str(x).lower() == 'true'), 
                      default=False)
  parser.add_argument("--output_dir_path",
                      help="dir path to score/perform dataset", 
                      type=str,
                      default='datasets/main_dataset')
  parser.add_argument("--emotion_output_dir_path",
                      help="dir path to score/perform dataset", 
                      type=str,
                      default='datasets/emotion_dataset')
  return parser


if __name__ == "__main__":
  parser = make_parser
  args = parser.parse_args()

  dataset = data_class.YamahaDataset(args.dataset_path, save=args.save_from_beginning, features_only=args.load_feature_only)
  emotion_dataset = data_class.EmotionDataset(args.emotionset_path, save=args.save_from_beginning, features_only=args.load_feature_only)

  if not args.load_feature_only:
    dataset.extract_all_features(save=True)
    emotion_dataset.extract_all_features(save=True)

  pair_set = dft.PairDataset(dataset)
  pair_set.save_features_for_virtuosoNet(args.output_dir_path)
  emotion_pair_set = dft.PairDataset(emotion_dataset)
  emotion_pair_set.feature_stats = load_dat(Path(args.output_dir_path)/'stat.dat')['stats']
  emotion_pair_set.save_features_for_virtuosoNet(args.emotion_output_dir_path, update_stats=False)