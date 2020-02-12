import sys
from pathlib import Path

from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch as th
import _pickle as cPickle

from .parser import get_parser, get_name
from . import model as modelzoo
from . import model_parameters as param
from . import train
from . import utils


def main():
    parser = get_parser()
    # random.seed(0)
    args = parser.parse_args()
    # name = get_name(parser, args)  # TODO: make model code via get_name
    name = args.modelCode
    print(f"Experiment {name}")

    '''
    eval_folder = args.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.logs.mkdir(exist_ok=True)
    metrics_path = args.logs / f"{name}.txt"
    eval_folder.mkdir(exist_ok=True, parents=True)
    args.checkpoints.mkdir(exist_ok=True, parents=True)
    args.models.mkdir(exist_ok=True, parents=True)
    '''

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
        model_config = param.initialize_model_parameters_by_code(args)
        model_config.training_args = args
        param.save_parameters(model_config, args.modelCode + '_param')
    elif args.resumeTraining:
        model_config = param.load_parameters(args.modelCode + '_param')
    else:
        model_config = param.load_parameters(args.modelCode + '_param')
        TrillNET_Param = param.load_parameters(args.trillCode + '_param')
        # if not hasattr(NET_PARAM, 'num_edge_types')
        #     NET_PARAM.num_edge_types = 10
        # if not hasattr(TrillNET_Param, 'num_edge_types'):
        #     TrillNET_Param.num_edge_types = 10
        TRILL_MODEL = modelzoo.TrillRNN(TrillNET_Param, device).to(device)

    if 'isgn' in args.modelCode:
        MODEL = modelzoo.ISGN(model_config, device).to(device)
    elif 'han' in args.modelCode:
        if 'ar' in args.modelCode:
            step_by_step = True
        else:
            step_by_step = False
        MODEL = modelzoo.HAN_Integrated(model_config, device, step_by_step).to(device)
    elif 'trill' in args.modelCode:
        MODEL = modelzoo.TrillRNN(model_config, device).to(device)
    else:
        print('Error: Unclassified model code')
        # Model = modelzoo.HAN_VAE(NET_PARAM, device, False).to(device)

    optimizer = th.optim.Adam(
        MODEL.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # checkpoint = args.checkpoints / f"{name}.th"
    # checkpoint_tmp = args.checkpoints / f"{name}.th.tmp"
    # if args.restart and checkpoint.exists():
    #     checkpoint.unlink()

    # TODO: to single function
    # load dataset

    with open(args.dataName, "rb") as f:
        u = cPickle.Unpickler(f)
        data_set = u.load()
    with open(args.test_dataName, "rb") as f:
        u = cPickle.Unpickler(f)
        test_data_set = u.load()
    train_data = data_set['train']
    valid_data = data_set['valid']
    print(type(train_data))
    test_data = test_data_set

    '''
    if args.loss == 'MSE':
        def criterion(pred, target, aligned_status=1):
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = th.sum(aligned_status).item() * pred.shape[-1]
                if data_size == 0:
                    data_size = 1
            if target.shape != pred.shape:
                print('Error: The shape of the target and prediction for the loss calculation is different')
                print(target.shape, pred.shape)
                return th.zeros(1).to(device)
            return th.sum(((target - pred) ** 2) * aligned_status) / data_size

    elif args.loss == 'CE':
        # criterion = nn.CrossEntropyLoss()
        def criterion(pred, target, aligned_status=1):
            if isinstance(aligned_status, int):
                data_size = pred.shape[-2] * pred.shape[-1]
            else:
                data_size = th.sum(aligned_status).item() * pred.shape[-1]
                if data_size == 0:
                    data_size = 1
                    print('data size for loss calculation is zero')
            return -1 * th.sum((target * th.log(pred) + (1-target) * th.log(1-pred)) * aligned_status) / data_size
    '''
    criterion = utils.criterion

    train.train(args,
                MODEL,
                train_data,
                valid_data,
                device,
                optimizer, 
                args.num_epochs, 
                None,  # TODO: bins: what should be? 
                args.time_steps,
                criterion)


if __name__ == "__main__":
    main()