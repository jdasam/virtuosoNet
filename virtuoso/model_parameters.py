import pickle
from . import model_constants as cons

class NetParams:
    class Param:
        def __init__(self):
            self.size = 0
            self.layers = 1
            self.input = 0
            self.margin = 0

    def __init__(self):
        self.note = self.Param()
        self.onset = self.Param()
        self.beat = self.Param()
        self.measure = self.Param()
        self.section = self.Param()
        self.performance = self.Param()
        self.final = self.Param()
        self.voice = self.Param()
        self.sum = self.Param()
        self.encoder = self.Param()
        self.time_reg = self.Param()
        self.margin = self.Param()
        self.encoded_vector_size = 16
        self.num_attention_head = 8



class ModelConfig(NetParams):
    def __init__(self):
        super().__init__()
        self.input_size = 0
        self.output_size = 0
        self.graph_iteration = 5
        self.sequence_iteration = 5
        self.is_graph = False
        self.is_teacher_force = False
        self.is_baseline = False
        self.hierarchy_level = None
        self.is_dependent = False
        self.is_simplified = False
        self.is_test_version = False
        self.is_trill = False
        self.training_args = None
        self.graph_keys = ['onset', 'forward', 'melisma', 'rest']
        self.num_edge_types = len(self.graph_keys) * 2
        self.loss_type = 'MSE'



def save_parameters(param, save_name):
    with open(save_name + ".dat", "wb") as f:
        pickle.dump(param, f, protocol=2)


def load_parameters(file_name):
    with open(file_name + ".dat", "rb") as f:
        u = pickle._Unpickler(f)
        net_params = u.load()
        return net_params


def initialize_model_parameters_by_code(args):
    model_config = ModelConfig()
    model_config.input_size = cons.SCORE_INPUT
    model_config.output_size = cons.NUM_PRIME_PARAM

    if args.slurEdge:
        model_config.graph_keys.append('slur')
    if args.voiceEdge:
        model_config.graph_keys.append('voice')
    model_config.num_edge_types = len(model_config.graph_keys) * 2

    if 'isgn' in args.modelCode:
        model_config.note.layers = 2
        model_config.note.size = 128
        model_config.measure.layers = 2
        model_config.measure.size = 64
        model_config.final.margin = 32

        model_config.encoded_vector_size = 16
        model_config.encoder.size = 128
        model_config.encoder.layers = 2

        model_config.time_reg.layers = 2
        model_config.time_reg.size = 32
        model_config.graph_iteration = 4
        model_config.sequence_iteration = 3

        model_config.final.input = (model_config.note.size + model_config.measure.size * 2) * 2
        model_config.encoder.input = (model_config.note.size + model_config.measure.size * 2) * 2 \
                                  + cons.NUM_PRIME_PARAM
        if 'sggnn_note' in args.modelCode:
            model_config.final.input += model_config.note.size
            model_config.encoder.input += model_config.note.size

        if 'baseline' in args.modelCode:
            model_config.is_baseline = True

    elif 'han' in args.modelCode:
        model_config.note.layers = 2
        model_config.note.size = 128
        model_config.beat.layers = 2
        model_config.beat.size = 128
        model_config.measure.layers = 1
        model_config.measure.size = 128
        model_config.final.layers = 1
        model_config.final.size = 64
        model_config.voice.layers = 2
        model_config.voice.size = 128
        model_config.performance.size = 128

        # net_param.num_attention_head = 1
        model_config.encoded_vector_size = 16
        model_config.encoder.size = 64
        model_config.encoder.layers = 2
        model_config.encoder.input = (model_config.note.size + model_config.beat.size +
                                   model_config.measure.size + model_config.voice.size) * 2 \
                                  + model_config.performance.size
        num_tempo_info = 3  # qpm primo, tempo primo
        num_dynamic_info = 0
        model_config.final.input = (model_config.note.size + model_config.voice.size + model_config.beat.size +
                                 model_config.measure.size) * 2 + model_config.encoder.size + \
                                num_tempo_info + num_dynamic_info
        if 'graph' in args.modelCode:
            model_config.is_graph = True
            model_config.graph_iteration = 3
            model_config.encoder.input = (model_config.note.size + model_config.beat.size +
                                       model_config.measure.size) * 2 \
                                      + cons.NUM_PRIME_PARAM
            model_config.final.input = (model_config.note.size +  model_config.beat.size +
                                     model_config.measure.size) * 2 + model_config.encoder.size + \
                                    num_tempo_info + num_dynamic_info
        if 'ar' in args.modelCode:
            model_config.final.input += model_config.output_size

        if 'teacher' in args.modelCode:
            model_config.is_teacher_force = True
        if 'baseline' in args.modelCode:
            model_config.is_baseline = True
            model_config.encoder.input = model_config.note.size * 2 + cons.NUM_PRIME_PARAM
            model_config.final.input = model_config.note.size * 2 + model_config.encoder.size + num_tempo_info + num_dynamic_info + model_config.output_size

    elif 'trill' in args.modelCode:
        model_config.input_size = cons.SCORE_INPUT + cons.NUM_PRIME_PARAM
        model_config.output_size = cons.NUM_TRILL_PARAM
        model_config.note.layers = 2
        model_config.note.size = 32
        model_config.is_trill = True

    else:
        print('Unclassified model code')

    if 'measure' in args.modelCode:
        model_config.hierarchy_level = 'measure'
        model_config.output_size = 2
        # net_param.encoder.input += 2 - cons.NUM_PRIME_PARAM
    elif 'beat' in args.modelCode:
        model_config.hierarchy_level = 'beat'
        model_config.output_size = 2
        # net_param.encoder.input += 2 - cons.NUM_PRIME_PARAM
    elif 'note' in args.modelCode:
        model_config.input_size += 2
        model_config.is_dependent = True
        if 'measure' in args.hierCode:
            model_config.hierarchy_level = 'measure'
        elif 'beat' in args.hierCode:
            model_config.hierarchy_level = 'beat'

    if 'altv' in args.modelCode:
        model_config.is_test_version = True

    return model_config

