import sys
from pathlib import Path

from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch as th

from .parser import get_parser, get_name
from . import model as modelzoo
from . import model_parameters as param

def main():
    parser = get_parser()
    # random.seed(0)
    args = parser.parse_args()
    name = get_name(parser, args)    
    print(f"Experiment {name}")

    eval_folder = args.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.logs.mkdir(exist_ok=True)
    metrics_path = args.logs / f"{name}.txt"
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.checkpoints.mkdir(exist_ok=True, parents=True)
    args.models.mkdir(exist_ok=True, parents=True)

    if args.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    th.manual_seed(args.seed)
    # Prevents too many threads to be started when running `museval` as it can be quite
    # inefficient on NUMA architectures.
    # os.environ["OMP_NUM_THREADS"] = "1"

    
    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(args.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)


    # Perhaps it can be handle in graph.py? 
    # GRAPH_KEYS = ['onset', 'forward', 'melisma', 'rest']
    # if args.slurEdge:
    #     GRAPH_KEYS.append('slur')
    # if args.voiceEdge:
    #     GRAPH_KEYS.append('voice')
    # N_EDGE_TYPE = len(GRAPH_KEYS) * 2
    # Moved to model_parameters


    # Suggestion: 
    # load parameter directly.
    # save model param in checkpoint?
    if args.sessMode == 'train' and not args.resumeTraining:
        NET_PARAM = param.initialize_model_parameters_by_code(args)
        NET_PARAM.num_edge_types = N_EDGE_TYPE
        NET_PARAM.training_args = args
        param.save_parameters(NET_PARAM, args.modelCode + '_param')
    elif args.resumeTraining:
        NET_PARAM = param.load_parameters(args.modelCode + '_param')
    else:
        NET_PARAM = param.load_parameters(args.modelCode + '_param')
        TrillNET_Param = param.load_parameters(args.trillCode + '_param')
        # if not hasattr(NET_PARAM, 'num_edge_types')
        #     NET_PARAM.num_edge_types = 10
        # if not hasattr(TrillNET_Param, 'num_edge_types'):
        #     TrillNET_Param.num_edge_types = 10
        TRILL_MODEL = modelzoo.TrillRNN(TrillNET_Param, device).to(device)

    if 'isgn' in args.modelCode:
        MODEL = modelzoo.ISGN(NET_PARAM, device).to(device)
    elif 'han' in args.modelCode:
        if 'ar' in args.modelCode:
            step_by_step = True
        else:
            step_by_step = False
        MODEL = modelzoo.HAN_Integrated(NET_PARAM, device, step_by_step).to(device)
    elif 'trill' in args.modelCode:
        MODEL = modelzoo.TrillRNN(NET_PARAM, device).to(device)
    else:
        print('Error: Unclassified model code')
        # Model = modelzoo.HAN_VAE(NET_PARAM, device, False).to(device)

    optimizer = th.optim.Adam(
        MODEL.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    checkpoint = args.checkpoints / f"{name}.th"
    checkpoint_tmp = args.checkpoints / f"{name}.th.tmp"
    if args.restart and checkpoint.exists():
        checkpoint.unlink()


    # TODO: to single function
    # load dataset
    

