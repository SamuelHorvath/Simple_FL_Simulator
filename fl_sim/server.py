import torch
from copy import deepcopy


class TorchServer(object):
    def __init__(self, global_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.global_model = global_model
        self.optimizer = optimizer
        self.local_optimizer_state_dict = None

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.global_model.parameters():
            # gradient and weights (p.data) have the same shape
            end = beg + len(p.data.view(-1))
            x = gradient[beg:end].reshape_as(p.data)
            p.grad = x.clone().detach()
            beg = end

    def set_local_optimizer_dict(self, dict_example, local_optimizer_state):
        beg = 0
        self.local_optimizer_state_dict = deepcopy(dict_example)
        optim_states = self.local_optimizer_state_dict['state']
        for param_id in optim_states.keys():
            for param_type in optim_states[param_id].keys():
                param = optim_states[param_id][param_type]
                end = beg + len(param.data.view(-1))
                p = local_optimizer_state[beg:end].reshape_as(param.data)
                if p.device != param.device:
                    p = p.to(param.device)
                optim_states[param_id][param_type] = p.clone().detach()
