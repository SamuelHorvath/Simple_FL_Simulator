import argparse
import time
import torch
from datetime import datetime
import os


def get_args(args):
    parser = initialise_arg_parser(args, 'Simple FL Simulator.')

    # Experimental setup
    parser.add_argument("--task", type=str, help="Task to solve")
    parser.add_argument("-agg", "--aggregation", default="mean", help="How to aggregate updates")
    parser.add_argument("--sampling", type=str, default="uniform", help="Client sampling")
    parser.add_argument("-sw", "--simulated-workers", type=int, help="Number of workers to use for simulation")

    # Utility
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--identifier", type=str, default=str(time.time()), help="Identifier for the current job")

    # Experiment configuration
    parser.add_argument("-cr", "--comm_rounds", type=int, default=10, help="Number of epochs")
    parser.add_argument("--global-opt", default='sgd', type=str, help="Global optimizer")
    parser.add_argument("--global-lr", default=1., type=float, help="Global learning rate")
    parser.add_argument("--local-opt", default='sgd', type=str, help="Local optimizer")
    parser.add_argument("--local-lr", default=0.1, type=float, help="Local learning rate")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--test-batch-size", default=128, type=int, help="Test batch size")

    # SETUP ARGUMENTS
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="../outputs/",
        help="Base root directory for the output."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/",
        help="Base root directory for the dataset."
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Define on which GPU (CPU or m1 chip for Macs) to run the model. If -1, use CPU."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Num workers for dataset loading"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="How often to do validation."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically for reproducibility."
    )
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    os.makedirs("../logs/", exist_ok=True)
    parser.add_argument(
        "--logfile",
        type=str,
        default=f"../logs/log_{now}.txt"
    )

    args = parser.parse_args()
    transform_gpu_args(args)
    return args


def transform_gpu_args(args):
    """
    Transforms the gpu arguments to the expected format.
    """
    if args.gpu == "m1":
        args.device = "mps" if \
            torch.backends.mps.is_available() else 'cpu'
    elif args.gpu == "-1":
        args.device = "cpu"
    else:
        gpu_str_arg = args.gpu.split(',')
        if len(gpu_str_arg) > 1:
            raise NotImplementedError("Multiple GPUs not supported.")
        else:
            args.device = f"cuda:{args.gpu}" if \
                torch.cuda.is_available() else 'cpu'


def initialise_arg_parser(args, description):
    parser = argparse.ArgumentParser(args, description=description)
    return parser
