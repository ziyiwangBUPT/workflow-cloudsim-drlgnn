import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class VmActorCriticAgent(nn.Module):
    def __init__(self, vm_count: int):
        super().__init__()
        self.vm_count = vm_count
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array((vm_count * 3,)).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array((vm_count * 3,)).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, vm_count), std=0.01),
        )

    def get_value(self, x: torch.Tensor):
        x = x[:, self.vm_count :]
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mask = x[:, : self.vm_count]
        x = x[:, self.vm_count :]

        logits = self.actor(x) - (1 - mask) * 1e8
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
