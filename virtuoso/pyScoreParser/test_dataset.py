
import argparse
from .data_class import YamahaDataset, EmotionDataset, AsapDataset



parser = argparse.ArgumentParser()
parser.add_argument('--yamaha_path', type=str)
parser.add_argument('--emotion_path', type=str)
parser.add_argument('--asap_path', type=str)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

if args.yamaha_path:
    dataset = YamahaDataset(args.yamaha_path, args.save)
    for piece in dataset.pieces:
        print(piece)  
if args.emotion_path:
    dataset = EmotionDataset(args.emotion_path, args.save)
    for piece in dataset.pieces:
        print(piece)
if args.asap_path:
    dataset = AsapDataset(args.asap_path, args.save)
    for piece in dataset.pieces:
        print(piece)
    