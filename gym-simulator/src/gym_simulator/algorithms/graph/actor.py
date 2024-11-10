from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from icecream import ic

from gym_simulator.algorithms.graph.decode import decode_observation
from gym_simulator.algorithms.graph.network import Network


class Actor(nn.Module):
    def __init__(self, max_jobs: int, max_machines: int, hidden_dim: int = 64):
        super().__init__()

        self.max_jobs = max_jobs
        self.max_machines = max_machines
        self.actor = Network(max_jobs, max_machines, hidden_dim, max_jobs * max_machines)
        self.critic = Network(max_jobs, max_machines, hidden_dim, 1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, N)
        :return values: (batch_size,)
        """
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            features = decode_observation(x[batch_index])
            values.append(self.critic(*features))

        return torch.stack(values)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (batch_size, N)
        :param action: (batch_size,)
        :return chosen_actions: (batch_size,)
        :return log_probs: (batch_size,)
        :return entropies: (batch_size,)
        :return values: (batch_size,)
        """
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            features = decode_observation(x[batch_index])
            action_scores: torch.Tensor = self.actor(*features)
            action_scores = action_scores.squeeze(-1)
            ic(action_scores)

            mask = torch.zeros((self.max_jobs, self.max_machines))
            task_state_ready = features[1]
            task_vm_compatibility = features[4]
            mask[task_state_ready == 0, :] = 1
            mask = mask.masked_fill(~task_vm_compatibility.bool(), 1)

            action_scores = action_scores.reshape(self.max_jobs, self.max_machines)
            ic(action_scores)
            action_scores = action_scores.masked_fill(mask.bool(), -1e8)
            action_scores = action_scores.flatten()

            action_probabilities = F.softmax(action_scores, dim=0)
            action_probabilities = action_probabilities.reshape(self.max_jobs, self.max_machines)
            action_probabilities = action_probabilities.masked_fill(mask.bool(), -1e8)
            action_probabilities = action_probabilities.flatten()

            probs = Categorical(logits=action_probabilities)
            chosen_action = action[batch_index] if action is not None else probs.sample()

            log_prob = torch.log(action_probabilities[chosen_action])
            entropy = -(action_probabilities * torch.log(action_probabilities + 1e-10)).sum()
            value = self.critic(*features)

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions)
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        values = torch.stack(all_values)

        return chosen_actions, log_probs, entropies, values
