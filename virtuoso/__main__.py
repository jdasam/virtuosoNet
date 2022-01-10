import sys
from datetime import datetime
from pathlib import Path

from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch

from .parser import get_parser, get_name
from . import utils
from .train import train
from .model import make_model
from .inference import inference, inference_with_emotion

def main():
    parser = get_parser()

    args = parser.parse_args()
    torch.manual_seed(args.th_seed)
    # random.seed(0)

    args, net_param, config = utils.handle_args(args)
    name = get_name(parser, args)  + "_" + datetime.now().strftime('%y%m%d-%H%M%S')
    print(f"Experiment {name}")

    device = utils.get_device(args)
    criterion = utils.make_criterion_func(config.train_params.loss_type)


    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    model = make_model(net_param)
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