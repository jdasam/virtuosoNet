
import argparse
from .data_class import YamahaDataset, EmotionDataset



parser = argparse.ArgumentParser()
parser.add_argument('--yamaha_path', type=str)
parser.add_argument('--emotion_path', type=str)
args = parser.parse_args()

if args.yamaha_path:
    dataset = YamahaDataset(args.yamaha_path)
    for piece in dataset.pieces:
        print(piece)  
if args.emotion_path:
    dataset = EmotionDataset(args.emotion_path)
    for piece in dataset.pieces:
        print(piece)