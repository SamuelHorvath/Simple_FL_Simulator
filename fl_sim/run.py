import os
import sys
import json
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader

from opts import get_args

# Utility functions
from utils.utils import top1_accuracy, \
    create_model_dir, init_metrics_meter, extend_metrics_dict, metric_to_dict
from utils.tasks import get_task_elements, get_agg, get_sampling, get_optimizer_init
from utils.logger import Logger

# Main Modules
from worker import initialize_worker
from server import TorchServer
from simulator import ParallelTrainer, DistributedEvaluator


def main(args):
    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    Logger()

    if torch.cuda.device_count():
        cuda_support = True
    else:
        Logger.get().warning('CUDA unsupported!!')
        cuda_support = False

    if args.deterministic:
        import os
        import random

        if cuda_support:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)

    loader_kwargs = {"persistent_workers": args.num_workers > 0, "num_workers": args.num_workers} \
        if not args.device == 'cpu' else {}
    train_loader_kwargs = {'batch_size': args.batch_size, 'shuffle': True, **loader_kwargs}

    global_model, loss_func, is_rnn, test_batch_size, train_datasets, test_dataset = \
        get_task_elements(args.task, args.test_batch_size, args.data_path)
    global_model, loss_func = global_model.to(args.device), loss_func.to(args.device)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False, **loader_kwargs)

    local_model_s = [deepcopy(global_model) for _ in range(args.simulated_workers)]
    train_loader_s = [DataLoader(dataset, args.batch_size, shuffle=True, **loader_kwargs) for
                      dataset in train_datasets]

    local_opt_init = get_optimizer_init(args.local_opt, args.local_lr)
    server_opt = get_optimizer_init(args.global_opt, args.global_lr)(global_model.parameters())
    agg = get_agg(args.aggregation)
    client_sampler = get_sampling(
        args.sampling, args.comm_rounds, args.simulated_workers, len(train_loader_s), args.seed)

    # EDIT YOUR METRICS OF INTEREST HERE
    metrics = {'top_1_acc': top1_accuracy}

    server = TorchServer(
        global_model=global_model,
        optimizer=server_opt
    )
    trainer = ParallelTrainer(
        server=server,
        aggregator=agg,
        client_sampler=client_sampler,
        datasets=train_datasets,
        data_loader_kwargs=train_loader_kwargs,
        log_interval=args.log_interval,
        metrics=metrics,
        device=args.device
    )

    test_evaluator = DistributedEvaluator(
        model=global_model,
        is_rnn=is_rnn,
        data_loader=test_loader,
        loss_func=loss_func,
        device=args.device,
        metrics=metrics,
        log_interval=args.log_interval,
        log_identifier_type='Test',
    )

    for worker_id, w_model in enumerate(local_model_s):
        worker = initialize_worker(
            worker_id=worker_id,
            model=w_model,
            is_rnn=is_rnn,
            optimizer_init=local_opt_init,
            loss_func=loss_func,
            device=args.device,
            server=server,
            log_interval=args.log_interval,
        )
        trainer.add_worker(worker)

    full_metrics = init_metrics_meter(metrics)
    model_dir = create_model_dir(args)
    if os.path.exists(os.path.join(
            model_dir, 'full_metrics.json')):
        Logger.get().info(f"{model_dir} already exists.")
        Logger.get().info("Skipping this setup.")
        return
    # create model directory
    os.makedirs(model_dir, exist_ok=True)
    test_metric = test_evaluator.evaluate(0)
    extend_metrics_dict(
        full_metrics, metric_to_dict(test_metric, metrics, 0, 'test'))
    for comm_round in range(1, args.comm_rounds + 1):
        Logger.get().info(f"Communication round {comm_round}/{args.comm_rounds}")
        train_metric = trainer.train(comm_round)
        extend_metrics_dict(
            full_metrics, metric_to_dict(train_metric, metrics, comm_round, 'train'))
        if comm_round % args.eval_every == 0 or comm_round == args.comm_rounds:
            test_metric = test_evaluator.evaluate(comm_round)
            extend_metrics_dict(
                full_metrics, metric_to_dict(test_metric, metrics, comm_round, 'test'))
    #  store the run
    with open(os.path.join(
            model_dir, 'full_metrics.json'), 'w') as f:
        json.dump(full_metrics, f, indent=4)


if __name__ == "__main__":
    args = get_args(sys.argv)
    main(args)
    torch.cuda.empty_cache()
