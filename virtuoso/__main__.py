import sys
from pathlib import Path

from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch

from .parser import get_parser, get_name
from . import model as modelzoo
from . import model_parameters as param
from . import utils
from . train import train

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
    # args.models.mkdir(exist_ok=True, parents=True)

    if args.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    config = utils.read_model_setting(args.yml_path)
    torch.manual_seed(args.seed)
    # Prevents too many threads to be started when running `museval` as it can be quite
    # inefficient on NUMA architectures.
    # os.environ["OMP_NUM_THREADS"] = "1"

    
    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
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
    
    net_param = config.nn_params

    # Suggestion: 
    # load parameter directly.
    # save model param in checkpoint?
    # if args.sessMode == 'train' and not args.resumeTraining:
    #     NET_PARAM = param.initialize_model_parameters_by_code(args.modelCode)
    #     NET_PARAM.num_edge_types = N_EDGE_TYPE
    #     NET_PARAM.training_args = args
    #     param.save_parameters(NET_PARAM, args.modelCode + '_param')
    # elif args.resumeTraining:
    #     NET_PARAM = param.load_parameters(args.modelCode + '_param')
    # else:
    #     NET_PARAM = param.load_parameters(args.modelCode + '_param')
    #     TrillNET_Param = param.load_parameters(args.trillCode + '_param')
    #     # if not hasattr(NET_PARAM, 'num_edge_types')
    #     #     NET_PARAM.num_edge_types = 10
    #     # if not hasattr(TrillNET_Param, 'num_edge_types'):
    #     #     TrillNET_Param.num_edge_types = 10
    #     TRILL_MODEL = modelzoo.TrillRNN(TrillNET_Param, device).to(device)

    if 'isgn' in args.model_code:
        model = modelzoo.ISGN(net_param, device).to(device)
    elif 'han' in args.model_code:
        if 'ar' in args.model_code:
            step_by_step = True
        else:
            step_by_step = False
        model = modelzoo.HAN_Integrated(net_param, device, step_by_step).to(device)
    elif 'trill' in args.model_code:
        model = modelzoo.TrillRNN(net_param, device).to(device)
    else:
        print('Error: Unclassified model code')

    
    checkpoint = args.checkpoints / f"{name}.pt"
    checkpoint_tmp = args.checkpoints / f"{name}.pt.tmp"
    if args.resume_training and checkpoint.exists():
        checkpoint.unlink()

    criterion = utils.make_criterion_func(config.train_params.loss_type, device)

    #TODO add bins information
    bins = []


    if args.session_mode == "train":
        train(args,
            model,
            device,
            optimizer, 
            args.num_epochs, 
            bins, 
            args.time_steps,
            criterion, 
            NUM_INPUT, 
            NUM_OUTPUT)

    

if __name__ == '__main__':
    main()