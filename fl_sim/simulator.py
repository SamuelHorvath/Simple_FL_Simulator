import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from typing import Union, Callable, Any, List

from utils.logger import Logger
from utils.utils import init_metrics_meter, update_metrics

from worker import TorchWorker
from server import TorchServer
from client_sampling.base import _ClientSampler


class DistributedSimulatorBase(object):
    """Simulate distributed programs with low memory usage.
    This base class is used by both trainer and evaluator.
    """

    def __init__(self, metrics: dict, device: str):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            device (bool): dpu, cuda or mps
        """
        self.metrics = metrics
        self.device = device
        self.workers = []

        self.logger = Logger.get()


class ParallelTrainer(DistributedSimulatorBase):
    """Synchronous and parallel training with specified aggregator."""

    def __init__(
            self,
            server: TorchServer,
            aggregator: Callable[[list], torch.Tensor],
            client_sampler: _ClientSampler,
            datasets: List[Dataset],
            data_loader_kwargs: dict,
            log_interval: int,
            metrics: dict,
            device: str,
    ):
        """
        Args:
            server (TorchServer)
            aggregator (callable): A callable which takes a list of tensors and returns
                an aggregated tensor.
            client_sampler (_ClientSampler)
            datasets (List[Dataset]): list of client data sets
            data_loader_kwargs:  params for data_loader
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functionality
            device (str): cuda, cpu or mps
        """
        self.aggregator = aggregator
        self.client_sampler = client_sampler
        self.datasets = datasets
        self.data_loader_kwargs = data_loader_kwargs
        self.server = server
        self.log_interval = log_interval
        self.random_states = {}
        super().__init__(metrics, device)

    def aggregation_and_update(self):
        pseudo_gradients = self.parallel_get(lambda w: w.get_update())
        aggregated_gradients = self.aggregator(pseudo_gradients)

        # Assume that the model and optimizers are shared among workers.
        self.server.set_gradient(aggregated_gradients)
        self.server.apply_gradient()

        # OPTIMIZERS STATES AGGREGATION
        # local_optim_states = self.parallel_get(lambda w: w.get_optim_states())
        # aggregated_optim_state = self.aggregator(local_optim_states)
        # self.server.set_local_optimizer_dict(
        #     self.workers[0].optimizer.state_dict(), aggregated_optim_state)

    def train(self, comm_round):
        global_model_dict = deepcopy(self.server.global_model.state_dict())
        # local_optimizer_state_dict = deepcopy(self.server.local_optimizer_state_dict)
        self.set_data_loaders(comm_round)
        self.parallel_call(lambda w: w.model.load_state_dict(global_model_dict))
        # OPTIMIZERS STATES AGGREGATION
        # # Avoid loading optimizer state dict in the first round
        # if local_optimizer_state_dict is not None:
        #     self.parallel_call(lambda w: w.optimizer.load_state_dict(local_optimizer_state_dict))

        self.parallel_call(lambda w: w.train_epoch_start())

        metrics_meter = init_metrics_meter(self.metrics, comm_round)
        self.parallel_get(lambda w: w.run_local_epochs(metrics_meter))
        self.aggregation_and_update()
        return metrics_meter

    def add_worker(self, worker: TorchWorker):
        worker.add_metrics(self.metrics)
        self.workers.append(worker)
        self.logger.info(f"=> Add worker {str(worker)}")

    def parallel_call(self, f: Callable[[TorchWorker], None]) -> None:
        for w in self.workers:
            f(w)

    def parallel_get(self, f: Callable[[TorchWorker], Any]) -> list:
        results = []
        for w in self.workers:
            results.append(f(w))
        return results

    def set_data_loaders(self, comm_round) -> None:
        for i, w in zip(self.client_sampler.get_sampled_clients(comm_round), self.workers):
            w.assign_data_loader(i, DataLoader(self.datasets[i], **self.data_loader_kwargs))

    def __str__(self):
        return (
            "ParallelTrainer("
            f"aggregator={self.aggregator}, "
            f"log_interval={self.log_interval}, "
            f"metrics={list(self.metrics.keys())}"
            ")"
        )

    def log_train(self, metrics_meter, batch_idx, epoch):

        # Output to console
        self.logger.info(
            f"Epoch: {epoch :2} Batch: {batch_idx}| {len(self.workers[0].data_loader)}|"
            f"  Loss: {metrics_meter['loss'].get_avg():.4f} "
            + " ".join(key + "=" + "{:>8.4f}".format(metrics_meter[key].get_avg()) for key in self.metrics)
        )


class DistributedEvaluator(DistributedSimulatorBase):
    def __init__(
            self,
            model: torch.nn.Module,
            is_rnn: bool,
            data_loader: torch.utils.data.DataLoader,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            metrics: dict,
            log_interval: int,
            log_identifier_type="Validation",
    ):
        super().__init__(metrics, device)
        self.model = model
        self.is_rnn = is_rnn
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
        self.log_identifier_type = log_identifier_type
        self.log_interval = log_interval

    def __str__(self):
        return (
            "DistributedEvaluator("
            f"device={self.device}, "
            ")"
        )

    def evaluate(self, comm_round):
        self.model.eval()
        metrics_meter = init_metrics_meter(self.metrics, comm_round)
        if self.is_rnn:
            hidden = self.model.init_hidden(self.data_loader.batch_size, self.device)

        with torch.no_grad():
            for i, (data, target) in enumerate(self.data_loader):
                batch_size = data.shape[0]
                data, target = data.to(self.device), target.to(self.device)
                if self.is_rnn:
                    output, hidden = self.model(data, hidden)
                    target = target.reshape(-1)
                else:
                    output = self.model(data)
                loss = self.loss_func(output, target).item()
                update_metrics(metrics_meter, 'loss', loss, batch_size)
                for key in self.metrics:
                    update_metrics(metrics_meter, key, self.metrics[key](output, target, self.model), batch_size)
                if i % self.log_interval == 0 or i + 1 == len(self.data_loader):
                    self.logger.info(
                        f"{self.log_identifier_type} | {i+1}/{len(self.data_loader)} |"
                        f" loss = {metrics_meter['loss'].get_avg():.4f}; "
                        + " ".join(key + " = " + "{:.4f}".format(metrics_meter[key].get_avg()) for key in self.metrics)
                    )
        return metrics_meter
