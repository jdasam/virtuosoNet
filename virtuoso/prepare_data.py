from .pyScoreParser import data_for_training as dft, data_class


dataset = data_class.YamahaDataset('/home/svcapp/userdata/chopin_cleaned/', save=True)
dataset.extract_all_features(save=True)

# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Liszt/Sonata', save=True)
# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', save=False)
# dataset.extract_all_features(save=True)
# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', save=False, features_only=True)
# pair_set = dft.PairDataset(dataset)
# pair_set.save_features_for_virtuosoNet('dataset_test/')