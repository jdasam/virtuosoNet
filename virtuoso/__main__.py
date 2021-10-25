import sys
from datetime import datetime
from pathlib import Path

from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch

from .parser import get_parser, get_name
from . import model as modelzoo
from . import model_parameters as param
from . import utils
from .train import train
from .inference import inference, inference_with_emotion
from . import encoder_score as encs
from . import encoder_perf as encp
from . import decoder as dec
from . import residual_selector as res

def main():
    parser = get_parser()

    args = parser.parse_args()
    torch.manual_seed(args.th_seed)
    # random.seed(0)

    if "isgn" not in args.model_code:
        args.intermediate_loss = False
    name = get_name(parser, args)  + "_" + datetime.now().strftime('%y%m%d-%H%M%S')
    print(f"Experiment {name}")
    # eval_folder = args.evals / name
    # eval_folder.mkdir(exist_ok=True, parents=True)
    # metrics_path = args.logs / f"{name}.txt"
    # eval_folder.mkdir(exist_ok=True, parents=True)
    # args.models.mkdir(exist_ok=True, parents=True)

    if args.device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    if args.yml_path is not None:
        config = utils.read_model_setting(args.yml_path)
        net_param = config.nn_params
    else:
        net_param = torch.load(str(args.checkpoint), map_location='cpu')['network_params']
        args.yml_path = list(Path(args.checkpoint).parent.glob('*.yml'))[0]
        config = utils.read_model_setting(args.yml_path)
    args.graph_keys = net_param.graph_keys
    args.meas_note = net_param.meas_note
    criterion = utils.make_criterion_func(config.train_params.loss_type, device)


    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    
    model = modelzoo.VirtuosoNet()
    model.score_encoder = getattr(encs, net_param.score_encoder_name)(net_param)
    model.performance_encoder = getattr(encp, net_param.performance_encoder_name)(net_param)
    model.residual_info_selector = getattr(res, net_param.residual_info_selector_name)()
    model.performance_decoder = getattr(dec, net_param.performance_decoder_name)(net_param)
    model.network_params = net_param
    model = model.to(device)
    

    # if not (args.session_mode =="train" and args.resume_training):
    #     checkpoint = torch.load(args.checkpoint)
    # checkpoint = args.checkpoints / f"{name}.pt"
    # checkpoint_tmp = args.checkpoints / f"{name}.pt.tmp"
    # if args.resume_training and checkpoint.exists():
    #     checkpoint.unlink()

    if args.session_mode == "train":
        train(args,
            model,
            device,
            args.num_epochs, 
            criterion,
            name,
            )
    elif args.session_mode == "inference":
        # stats= utils.load_dat(args.data_path / 'stat.dat')
        inference(args, model, device)
    
    elif args.session_mode == "inference_with_emotion":
        inference_with_emotion(args, model, device)

if __name__ == '__main__':
    main()