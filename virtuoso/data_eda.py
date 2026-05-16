from .dataset import ScorePerformDataset


if __name__ == '__main__':
    dataset_path = 'dataset_beat/'
    dataset = ScorePerformDataset(dataset_path, 'entire', len_slice=1000, len_graph_slice=400, graph_keys=['forward', 'onset'])
    dataset