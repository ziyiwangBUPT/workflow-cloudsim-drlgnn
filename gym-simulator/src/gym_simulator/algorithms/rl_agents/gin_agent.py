from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GIN, global_mean_pool

from icecream import ic

from gym_simulator.algorithms.rl_agents.input_decoder import decode_observation


# Base Gin Network
# -----------------------------------------------------------------------------


class BaseGinNetwork(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.embedding_dim = embedding_dim
        self.device = device

        self.job_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.machine_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.job_machine_edge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.graph_network = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        )

    def forward(
        self,
        task_state_scheduled: torch.Tensor,
        task_state_ready: torch.Tensor,
        task_completion_time: torch.Tensor,
        vm_completion_time: torch.Tensor,
        task_vm_compatibility: torch.Tensor,
        task_vm_time_cost: torch.Tensor,
        task_vm_power_cost: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: n_jobs * n_machines vector
        """

        # Job nodes and features
        job_nodes = torch.arange(self.n_jobs).to(self.device).unsqueeze(1).expand(self.n_jobs, self.n_machines)
        job_features = torch.cat(
            [
                task_state_scheduled.unsqueeze(1),
                task_state_ready.unsqueeze(1),
                task_completion_time.unsqueeze(1),
            ],
            dim=1,
        )

        # Machine nodes and features
        machine_nodes = torch.arange(self.n_machines).to(self.device).unsqueeze(0).expand(self.n_jobs, self.n_machines)
        machine_nodes = machine_nodes + self.n_jobs  # Machine node indices are offset in n_jobs
        machine_features = vm_completion_time.unsqueeze(1)

        # Edges (Job-Job and Job-Machine)
        job_job_edge_index, _ = dense_to_sparse(adj)
        job_machine_connectivity = task_vm_compatibility.flatten()
        job_machine_edges = torch.cat([job_nodes.flatten().unsqueeze(1), machine_nodes.flatten().unsqueeze(1)], dim=1)
        job_machine_edges = job_machine_edges[job_machine_connectivity == 1]
        job_machine_edge_index = job_machine_edges.T
        edge_index = torch.cat([job_machine_edge_index, job_job_edge_index], dim=1)

        # Job-Machine Edge features
        job_machine_edge_features = task_vm_time_cost.reshape(-1, 1)
        job_machine_edge_features = job_machine_edge_features[job_machine_connectivity == 1]
        job_job_edge_count = edge_index.shape[1] - job_machine_edge_features.shape[0]

        # Encode nodes and edges
        x_job_node: torch.Tensor = self.job_encoder(job_features)
        x_machine_node: torch.Tensor = self.machine_encoder(machine_features)
        x_job_machine_edges: torch.Tensor = self.job_machine_edge_encoder(job_machine_edge_features)
        x_job_job_edges = torch.zeros(job_job_edge_count, self.embedding_dim, device=self.device)

        x = torch.cat([x_job_node, x_machine_node])
        edge_attr = torch.cat([x_job_machine_edges, x_job_job_edges], dim=0)

        # Get embeddings
        batch = torch.zeros(self.n_jobs + self.n_machines, dtype=torch.long, device=self.device)
        node_embeddings = self.graph_network(x, edge_index=edge_index, edge_attr=edge_attr)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        graph_embedding = global_mean_pool(node_embeddings, batch=batch)

        return node_embeddings, edge_embeddings, graph_embedding


# Gin Actor
# -----------------------------------------------------------------------------


class GinActor(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device

        self.network = BaseGinNetwork(
            n_jobs,
            n_machines,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        task_state_scheduled: torch.Tensor,
        task_state_ready: torch.Tensor,
        task_completion_time: torch.Tensor,
        vm_completion_time: torch.Tensor,
        task_vm_compatibility: torch.Tensor,
        task_vm_time_cost: torch.Tensor,
        task_vm_power_cost: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        _, edge_embeddings, _ = self.network(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_completion_time=task_completion_time,
            vm_completion_time=vm_completion_time,
            task_vm_compatibility=task_vm_compatibility,
            task_vm_time_cost=task_vm_time_cost,
            task_vm_power_cost=task_vm_power_cost,
            adj=adj,
        )

        # Get edge embedding scores
        edge_embedding_scores: torch.Tensor = self.edge_scorer(edge_embeddings)

        # Extract the job-machine edges only
        job_machine_connectivity = task_vm_compatibility.flatten()
        job_machine_edge_count = job_machine_connectivity.sum()
        job_machine_edge_embeddings = edge_embedding_scores[:job_machine_edge_count]

        # Actions scores should be the value in edge embedding, but -inf on invalid actions
        action_scores = torch.ones_like(job_machine_connectivity, dtype=torch.float32, device=self.device) * -1e8
        action_scores[job_machine_connectivity == 1] = job_machine_edge_embeddings.flatten()
        action_scores = action_scores.reshape(self.n_jobs, self.n_machines)
        action_scores[task_state_ready == 0, :] = -1e8  # Remove scores of actions with not ready tasks

        return action_scores


# Gin Critic
# -----------------------------------------------------------------------------


class GinCritic(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device

        BaseGinNetwork(
            n_jobs,
            n_machines,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
        )
        self.graph_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        task_state_scheduled: torch.Tensor,
        task_state_ready: torch.Tensor,
        task_completion_time: torch.Tensor,
        vm_completion_time: torch.Tensor,
        task_vm_compatibility: torch.Tensor,
        task_vm_time_cost: torch.Tensor,
        task_vm_power_cost: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        _, _, graph_embedding = self.network(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_completion_time=task_completion_time,
            vm_completion_time=vm_completion_time,
            task_vm_compatibility=task_vm_compatibility,
            task_vm_time_cost=task_vm_time_cost,
            task_vm_power_cost=task_vm_power_cost,
            adj=adj,
        )

        # Critic value is derived from global graph state
        return self.graph_scorer(graph_embedding.unsqueeze(0))


# Gin Agent
# -----------------------------------------------------------------------------


class GinAgent(nn.Module):
    def __init__(self, max_jobs: int, max_machines: int, device: torch.device):
        super().__init__()

        self.max_jobs = max_jobs
        self.max_machines = max_machines
        self.device = device

        self.actor = GinActor(max_jobs, max_machines, hidden_dim=32, embedding_dim=4, device=device)
        self.critic = GinCritic(max_jobs, max_machines, hidden_dim=32, embedding_dim=4, device=device)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, N)
        :return values: (batch_size,)
        """
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = decode_observation(x[batch_index].to(self.device))
            value = self.critic(
                task_state_scheduled=decoded_obs.task_state_scheduled,
                task_state_ready=decoded_obs.task_state_ready,
                task_completion_time=decoded_obs.task_completion_time,
                vm_completion_time=decoded_obs.vm_completion_time,
                task_vm_compatibility=decoded_obs.task_vm_compatibility,
                task_vm_time_cost=decoded_obs.task_vm_time_cost,
                task_vm_power_cost=decoded_obs.task_vm_power_cost,
                adj=decoded_obs.task_graph_edges,
            )
            values.append(value)

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
            decoded_obs = decode_observation(x[batch_index].to(self.device))
            action_scores: torch.Tensor = self.actor(
                task_state_scheduled=decoded_obs.task_state_scheduled,
                task_state_ready=decoded_obs.task_state_ready,
                task_completion_time=decoded_obs.task_completion_time,
                vm_completion_time=decoded_obs.vm_completion_time,
                task_vm_compatibility=decoded_obs.task_vm_compatibility,
                task_vm_time_cost=decoded_obs.task_vm_time_cost,
                task_vm_power_cost=decoded_obs.task_vm_power_cost,
                adj=decoded_obs.task_graph_edges,
            )
            action_scores = action_scores.flatten()
            action_probabilities = F.softmax(action_scores, dim=0)

            probs = Categorical(action_probabilities)
            chosen_action = action[batch_index] if action is not None else probs.sample()
            value = self.critic(
                task_state_scheduled=decoded_obs.task_state_scheduled,
                task_state_ready=decoded_obs.task_state_ready,
                task_completion_time=decoded_obs.task_completion_time,
                vm_completion_time=decoded_obs.vm_completion_time,
                task_vm_compatibility=decoded_obs.task_vm_compatibility,
                task_vm_time_cost=decoded_obs.task_vm_time_cost,
                task_vm_power_cost=decoded_obs.task_vm_power_cost,
                adj=decoded_obs.task_graph_edges,
            )

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(probs.log_prob(chosen_action))
            all_entropies.append(probs.entropy())
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions)
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)
        values = torch.stack(all_values)

        return chosen_actions, log_probs, entropies, values
