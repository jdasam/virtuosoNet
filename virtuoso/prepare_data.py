from virtuoso.pyScoreParser import feature_extraction
from .pyScoreParser import data_for_training as dft, data_class
from .utils import load_dat

# dataset = data_class.YamahaDataset('/home/svcapp/userdata/chopin_cleaned/', save=False, features_only=True)
dataset = data_class.EmotionDataset('/home/svcapp/userdata/emotion_data/emotion/', save=False, features_only=True)
# dataset.extract_all_features(save=True)
# dataset = data_class.YamahaDataset('/home/svcapp/userdata/chopin_cleaned/Chopin/Etudes_op_25/1/', save=True)
# dataset.extract_all_features(save=True)

pair_set = dft.PairDataset(dataset)
# pair_set.save_features_for_virtuosoNet('dataset_beat/')

pair_set.feature_stats = load_dat('dataset_beat/stat.dat')['stats']
pair_set.save_features_for_virtuosoNet('dataset_emotion_beat/', update_stats=False)

# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Liszt/Sonata', save=True)
# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', save=False)
# dataset.extract_all_features(save=True)
# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', save=False, features_only=True)
# pair_set = dft.PairDataset(dataset)
# pair_set.save_features_for_virtuosoNet('dataset_test/')