
import argparse
from .data_class import DataSet



parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str)
args = parser.parse_args()

dataset = DataSet(args.dataset_path)

