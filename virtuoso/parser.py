import argparse
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser("virtuosonet")
    parser.add_argument("-sess", "--session_mode", type=str,
                        default="train", help="train or inference")
    parser.add_argument("-yml", "--yml_path", type=str,
                        #default="isgn_param.yml",
                         help="yml file path")
    parser.add_argument("-data", "--data_path", type=Path,
                        default=Path("dataset_beat/"), help="data dir name")
    parser.add_argument("--emotion_data_path", type=Path,
                    default=Path("dataset_emotion_beat/"), help="data dir name")
    parser.add_argument("--resume", type=str,
                        default="_best.pth.tar", help="best model path")
    parser.add_argument("--xml_path", type=Path,
                        default=Path('/home/svcapp/userdata/dev/virtuosoNet/test_pieces/bps_5_1/musicxml_cleaned.musicxml'))
    parser.add_argument("--output_path", type=Path,
                        default=Path('test_result/'))
    # model model options
    parser.add_argument("-trill", "--is_trill", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="train trill")
    # parser.add_argument("-slur", "--slurEdge", default=False,
    #                     type=lambda x: (str(x).lower() == 'true'), help="slur edge in graph")
    # parser.add_argument("-voice", "--voiceEdge", default=True,
    #                     type=lambda x: (str(x).lower() == 'true'), help="network in voice level")
    # TODO: no redundancy?
    parser.add_argument("--is_hier", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--in_hier", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--hier_beat", default=False, type=lambda x: (str(x).lower() == 'true'))   
    parser.add_argument("--hier_model", default=False, type=lambda x: (str(x).lower() == 'true'))   
    parser.add_argument("--hier_meas", default=False, type=lambda x: (str(x).lower() == 'true'))   
    parser.add_argument("--meas_note", default=False, type=lambda x: (str(x).lower() == 'true'))   
    
    # training parameters
    parser.add_argument("--num_key_augmentation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    
    # dist parallel options
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")

    # save options
    parser.add_argument("--checkpoints_dir", 
                        type=Path,
                        default=Path('checkpoints'),
                        help='folder to store checkpoints')
    parser.add_argument("--checkpoint", 
                    type=Path,
                    default=Path('/home/svcapp/userdata/dev/virtuosoNet/isgn_best.pt'),
                    help='path to load checkpoint')    
    parser.add_argument("--evals",
                        type=Path,
                        default=Path('evals')
                        )
    parser.add_argument("--save",
                        action="store_true",)
    parser.add_argument("--logs",
                        type=Path,
                        default=Path("logs")
                        )
    
    # training option
    parser.add_argument("--num_epochs",
                    type=int,
                    default=100
                    )
    parser.add_argument("--iters_per_checkpoint",
                    type=int,
                    default=5000
                    )
    parser.add_argument("--iters_per_multi_perf",
                    type=int,
                    default=50
                    )
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4
                        )
    parser.add_argument("--len_slice",
                        type=int,
                        default=800
                        )
    parser.add_argument("--len_graph_slice",
                        type=int,
                        default=400
                        )
    parser.add_argument("--len_valid_slice",
                        type=int,
                        default=10000
                        )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-5
                        )
    parser.add_argument("--lr_decay_step",
                        type=float,
                        default=5000
                        )
    parser.add_argument("--lr_decay_rate",
                        type=float,
                        default=0.98
                        )
    parser.add_argument("--delta_weight",
                        type=float,
                        default=1
                        )
    parser.add_argument("--meas_loss_weight",
                        type=float,
                        default=1
                        )
    parser.add_argument("--multi_perf_dist_loss_margin",
                        type=float,
                        default=1
                        )
    parser.add_argument("--grad_clip",
                        type=float,
                        default=2
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
    
    # environment options
    parser.add_argument("-dev", "--device", type=int,
                        default=0, help="cuda device number")
    parser.add_argument("--num_workers", type=int,
                        default=0, help="num workers for dataloader")
    parser.add_argument("--pin_memory", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="pin memory for loader")
    parser.add_argument("--make_log", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="make log for training")
    parser.add_argument("--multi_perf_compensation", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="train style vector to be zero with multiple performances")
    parser.add_argument("-code", "--model_code", type=str,
                        default='isgn', help="code name for saving the model")
    parser.add_argument("-tCode", "--trillCode", type=str,
                        default='trill_default', help="code name for loading trill model")
    parser.add_argument("-comp", "--composer", type=str,
                        default='Beethoven', help="composer name of the input piece")
    parser.add_argument("--qpm_primo", type=int, help="Tempo at the beginning of the input piece in quarter notes per minute")
    parser.add_argument("--latent", type=float, default=0, help='initial_z value')
    parser.add_argument("-bp", "--boolPedal", default=False, type=lambda x: (
        str(x).lower() == 'true'), help='make pedal value zero under threshold')
    parser.add_argument("-reTrain", "--resume_training", default=False, type=lambda x: (
        str(x).lower() == 'true'), help='resume training after loading model')
    parser.add_argument("-perf", "--perfName", default='Anger_sub1',
                        type=str, help='resume training after loading model')
    parser.add_argument("-delta", "--delta_loss", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="apply delta value as loss during training")
    parser.add_argument("--vel_balance_loss", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="apply velocity balance as loss during training")
    parser.add_argument("-hCode", "--hierCode", type=str,
                        default='han_measure', help="code name for loading hierarchy model")
    parser.add_argument("-intermd", "--intermediate_loss", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="intermediate loss in ISGN")
    parser.add_argument("--tempo_loss_in_note", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="calculate tempo loss in note-level instead of beat-level")
    parser.add_argument("-dskl", "--disklavier", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="save midi for disklavier")
    parser.add_argument("--multi_instruments", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="save multi instruments as separate track")

    return parser


def get_name(parser, args):
    """
    Return the name of an experiment given the args. Some parameters are ignored,
    for instance --workers, as they do not impact the final result.
    """
    ignore_args = set([
        "checkpoints_dir",
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
        "device",
        "num_workers",
        "pin_memory",
        "make_log",
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
