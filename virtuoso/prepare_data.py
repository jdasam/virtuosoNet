from virtuoso.pyScoreParser import feature_extraction
from .pyScoreParser import data_for_training as dft, data_class
from .utils import load_dat


# dataset = data_class.YamahaDataset('/home/svcapp/userdata/chopin_cleaned/', save=False)
emotion_dataset = data_class.EmotionDataset('/Users/jeongdasaem/Documents/GitHub/EmotionData', save=True)

# dataset = data_class.YamahaDataset('/home/svcapp/userdata/chopin_cleaned/', save=False, features_only=True)
# emotion_dataset = data_class.EmotionDataset('/home/svcapp/userdata/emotion_data/emotion/', save=False, features_only=True)
# dataset.extract_all_features(save=True)
# dataset = data_class.YamahaDataset('/home/svcapp/userdata/chopin_cleaned/Chopin/Etudes_op_25/1/', save=True)
emotion_dataset.extract_all_features(save=True)

# pair_set = dft.PairDataset(dataset)
# pair_set.save_features_for_virtuosoNet('dataset_section_tempo/')

emotion_pair_set = dft.PairDataset(emotion_dataset)
emotion_pair_set.feature_stats = load_dat('dataset_section_tempo/stat.dat')['stats']
emotion_pair_set.save_features_for_virtuosoNet('dataset_emotion_section_tempo/', update_stats=False)

# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Liszt/Sonata', save=True)
# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', save=False)
# dataset.extract_all_features(save=True)
# dataset = data_class.AsapDataset('/home/svcapp/userdata/asap-dataset/Haydn', save=False, features_only=True)
# pair_set = dft.PairDataset(dataset)
# pair_set.save_features_for_virtuosoNet('dataset_test/')