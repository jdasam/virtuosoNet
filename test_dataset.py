
import argparse
from .data_class import YamahaDataset 



parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str)
args = parser.parse_args()

dataset = YamahaDataset(args.dataset_path)
for piece in dataset.pieces:
    print(piece)