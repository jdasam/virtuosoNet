import argparse
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser("virtuosonet")
    parser.add_argument("-data", "--dataName", type=str,
                        default="training_data", help="dat file name")
    parser.add_argument("--resume", type=str,
                        default="_best.pth.tar", help="best model path")

    # model model options
    parser.add_argument("-trill", "--trainTrill", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="train trill")
    parser.add_argument("-slur", "--slurEdge", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="slur edge in graph")
    parser.add_argument("-voice", "--voiceEdge", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="network in voice level")
    
    # training parameters
    parser.add_argument("-vel", "--velocity", type=str,
                        default='50,65', help="mean velocity of piano and forte")
    parser.add_argument("--num_key_augmentation", type=int, default=1)
    
    # dist parallel options
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")

    # save options
    parser.add_argument("--checkopints", 
                        type=Path,
                        default=Path('checkpoints'),
                        help='folder to store checkpoints')    
    parser.add_argument("--evals",
                        type='Path',
                        default=Path('evals'))
    parser.add_argument("--save",
                        action="store_true",)
    parser.add_argument("--logs",
                        type=Path,
                        default=Path("logs")
                        )
    
    # training option
    parser.add_argument("--lr",
                        type=float,
                        default=3e-4
                        )
    parser.add_argument("--time_steps",
                        type=int,
                        default=200
                        )
    parser.add_argument("--valid_steps",
                        type=int,
                        default=10000
                        )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-5
                        )
    parser.add_argument("--delta_weight",
                        type=float,
                        default=1
                        )
    parser.add_argument("--grad_clip",
                        type=float,
                        default=5
                        ) 
    parser.add_argument("--kld_max",
                        type=float,
                        default=0.02
                        ) 
    parser.add_argument("--kld_sig",
                        type=float,
                        default=15e4
                        ) 
    parser.add_argument("-loss", "--trainingLoss", type=str,
                        default='MSE', help='type of training loss')
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    
    # environment options
    parser.add_argument("-dev", "--device", type=int,
                        default=1, help="cuda device number")
    parser.add_argument("-code", "--modelCode", type=str,
                        default='isgn', help="code name for saving the model")
    parser.add_argument("-tCode", "--trillCode", type=str,
                        default='trill_default', help="code name for loading trill model")
    parser.add_argument("-comp", "--composer", type=str,
                        default='Beethoven', help="composer name of the input piece")
    parser.add_argument("--latent", type=float, default=0, help='initial_z value')
    parser.add_argument("-bp", "--boolPedal", default=False, type=lambda x: (
        str(x).lower() == 'true'), help='make pedal value zero under threshold')
    parser.add_argument("-reTrain", "--resumeTraining", default=False, type=lambda x: (
        str(x).lower() == 'true'), help='resume training after loading model')
    parser.add_argument("-perf", "--perfName", default='Anger_sub1',
                        type=str, help='resume training after loading model')
    parser.add_argument("-delta", "--deltaLoss", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="network in voice level")
    parser.add_argument("-hCode", "--hierCode", type=str,
                        default='han_measure', help="code name for loading hierarchy model")
    parser.add_argument("-intermd", "--intermediateLoss", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="intermediate loss in ISGN")
    parser.add_argument("-randtr", "--randomTrain", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="use random train")
    parser.add_argument("-dskl", "--disklavier", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="save midi for disklavier")

    return parser


def get_name(parser, args):
    """
    Return the name of an experiment given the args. Some parameters are ignored,
    for instance --workers, as they do not impact the final result.
    """
    ignore_args = set([
        "checkpoints",
        "deterministic",
        "eval",
        "evals",
        "eval_cpu",
        "eval_workers",
        "logs",
        "master",
        "rank",
        "restart",
        "save",
        "save_model",
        "show",
        "valid",
        "workers",
        "world_size",
    ])
    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in ignore_args:
            continue
        if value != parser.get_default(name):
            if isinstance(value, Path):
                parts.append(f"{name}={value.name}")
            else:
                parts.append(f"{name}={value}")
    if parts:
        name = " ".join(parts)
    else:
        name = "default"
    return name
