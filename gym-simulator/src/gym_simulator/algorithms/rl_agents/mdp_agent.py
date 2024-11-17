from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GIN, global_mean_pool

from icecream import ic

from gym_simulator.algorithms.rl_agents.input_decoder import decode_observation

# Job Actor
# -----------------------------------------------------------------------------


class JobActor(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines

        self.graph_network = GIN(
            in_channels=3,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=hidden_dim,
        )
        self.machine_embedder = nn.Sequential(
            nn.Linear(n_machines, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_decoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        task_state_scheduled: torch.Tensor,
        task_state_ready: torch.Tensor,
        task_completion_time: torch.Tensor,
        vm_completion_time: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = torch.zeros(self.n_jobs, dtype=torch.long)
        edge_index, _ = dense_to_sparse(adj)
        node_values = torch.cat(
            [
                task_state_scheduled.unsqueeze(-1),
                task_state_ready.unsqueeze(-1),
                task_completion_time.unsqueeze(-1),
            ],
            dim=-1,
        )

        node_embeddings = self.graph_network(node_values, edge_index=edge_index)  # h_v^L
        graph_embedding = global_mean_pool(node_embeddings, batch=batch)  # h_G
        machine_embedding = self.machine_embedder(vm_completion_time.unsqueeze(0))  # u

        combined_embeddings = torch.cat(
            [
                node_embeddings,
                graph_embedding.expand(self.n_jobs, -1),
                machine_embedding.expand(self.n_jobs, -1),
            ],
            dim=-1,
        )
        action_scores = self.mlp_decoder(combined_embeddings)
        action_scores = action_scores.reshape(self.n_jobs)
        action_scores[task_state_ready == 0] = -float("inf")

        action_probabilities = F.softmax(action_scores, dim=0)
        return action_probabilities, graph_embedding, machine_embedding


# Machine Actor
# -----------------------------------------------------------------------------


class MachineActor(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines

        self.machine_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_decoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        vm_compatibility: torch.Tensor,
        vm_time_cost: torch.Tensor,
        vm_power_cost: torch.Tensor,
        graph_embedding: torch.Tensor,
        machine_embedding: torch.Tensor,
    ) -> torch.Tensor:
        node_values = torch.cat(
            [
                vm_compatibility.unsqueeze(-1),
                vm_time_cost.unsqueeze(-1),
                vm_power_cost.unsqueeze(-1),
            ],
            dim=-1,
        )

        node_embeddings = self.machine_encoder(node_values)

        combined_embeddings = torch.cat(
            [
                node_embeddings,
                graph_embedding.expand(self.n_machines, -1),
                machine_embedding.expand(self.n_machines, -1),
            ],
            dim=-1,
        )
        action_scores = self.mlp_decoder(combined_embeddings)
        action_scores = action_scores.reshape(self.n_machines)
        action_scores[vm_compatibility == 0] = -float("inf")

        action_probabilities = F.softmax(action_scores, dim=0)
        return action_probabilities


# Agent
# -----------------------------------------------------------------------------


class MdpAgent(nn.Module):
    def __init__(self, max_jobs: int, max_machines: int, hidden_dim: int = 64):
        super().__init__()

        self.max_jobs = max_jobs
        self.max_machines = max_machines

        self.job_actor = JobActor(max_jobs, max_machines, hidden_dim)
        self.machine_actor = MachineActor(max_jobs, max_machines, hidden_dim)
        self.critic_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            (
                task_state_scheduled,
                task_state_ready,
                task_completion_time,
                vm_completion_time,
                task_vm_compatibility,
                task_vm_time_cost,
                task_vm_power_cost,
                task_graph_edges,
            ) = decode_observation(x[batch_index])
            job_action_probabilities, graph_embedding, machine_embedding = self.job_actor(
                task_state_scheduled=task_state_scheduled,
                task_state_ready=task_state_ready,
                task_completion_time=task_completion_time,
                vm_completion_time=vm_completion_time,
                adj=task_graph_edges,
            )
            values.append(self.critic_network(graph_embedding))

        return torch.stack(values)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            (
                task_state_scheduled,
                task_state_ready,
                task_completion_time,
                vm_completion_time,
                task_vm_compatibility,
                task_vm_time_cost,
                task_vm_power_cost,
                task_graph_edges,
            ) = decode_observation(x[batch_index])

            job_action_probabilities, graph_embedding, machine_embedding = self.job_actor(
                task_state_scheduled=task_state_scheduled,
                task_state_ready=task_state_ready,
                task_completion_time=task_completion_time,
                vm_completion_time=vm_completion_time,
                adj=task_graph_edges,
            )
            job_probs = Categorical(job_action_probabilities)
            chosen_job_action = action[batch_index] // self.max_machines if action is not None else job_probs.sample()
            job_log_prob = job_probs.log_prob(chosen_job_action)
            job_entropy = job_probs.entropy()

            machine_action_probabilities = self.machine_actor(
                vm_compatibility=task_vm_compatibility[chosen_job_action],
                vm_time_cost=task_vm_time_cost[chosen_job_action],
                vm_power_cost=task_vm_power_cost[chosen_job_action],
                graph_embedding=graph_embedding,
                machine_embedding=machine_embedding,
            )
            mach_probs = Categorical(machine_action_probabilities)
            chosen_mch_action = action[batch_index] % self.max_machines if action is not None else mach_probs.sample()
            machine_log_prob = mach_probs.log_prob(chosen_mch_action)
            machine_entropy = mach_probs.entropy()

            chosen_action = (chosen_job_action * self.max_machines) + chosen_mch_action
            log_prob = job_log_prob + machine_log_prob
            entropy = job_entropy + machine_entropy

            value = self.critic_network(graph_embedding.flatten())

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions)
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        values = torch.stack(all_values)

        return chosen_actions, log_probs, entropies, values
