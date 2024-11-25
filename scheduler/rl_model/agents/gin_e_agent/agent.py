from typing import Optional, Tuple
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax

from torch_geometric.nn.models import GIN
from torch_geometric.nn.glob import global_mean_pool
from torch_geometric.utils import dense_to_sparse

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.agent import Agent
from scheduler.rl_model.agents.gin_e_agent.mapper import GinEAgentObsTensor, GinEAgentMapper


# Base Gin Network
# ----------------------------------------------------------------------------------------------------------------------


class BaseGinENetwork(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.job_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.machine_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.graph_network = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinEAgentObsTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_jobs = obs.task_vm_compatibility.shape[0]
        n_machines = obs.task_vm_compatibility.shape[1]

        # Job nodes and features
        job_nodes = torch.arange(n_jobs).to(self.device).unsqueeze(1).expand(n_jobs, n_machines)
        job_features = torch.cat(
            [
                obs.task_state_scheduled.unsqueeze(1),
                obs.task_state_ready.unsqueeze(1),
                obs.task_completion_time.unsqueeze(1),
            ],
            dim=1,
        )

        # Machine nodes and features
        machine_nodes = torch.arange(n_machines).to(self.device).unsqueeze(0).expand(n_jobs, n_machines)
        machine_nodes = machine_nodes + n_jobs  # Machine node indices are offset in n_jobs
        machine_features = obs.vm_completion_time.unsqueeze(1)

        # Edges (Job-Job and Job-Machine)
        job_job_edge_index, _ = dense_to_sparse(obs.adj)
        job_machine_connectivity = obs.task_vm_compatibility.flatten()
        job_machine_edges = torch.cat([job_nodes.flatten().unsqueeze(1), machine_nodes.flatten().unsqueeze(1)], dim=1)
        job_machine_edges = job_machine_edges[job_machine_connectivity == 1]
        job_machine_edge_index = job_machine_edges.T
        edge_index = torch.cat([job_machine_edge_index, job_job_edge_index], dim=1)

        # Job-Machine Edge features
        job_machine_flag = torch.ones(n_jobs * n_machines, 1)
        flat_task_vm_time_cost = obs.task_vm_time_cost.reshape(-1, 1)
        flat_task_vm_energy_cost = obs.task_vm_energy_cost.reshape(-1, 1)
        job_machine_edge_features = torch.cat(
            [job_machine_flag, flat_task_vm_time_cost, flat_task_vm_energy_cost], dim=1
        )
        job_machine_edge_features = job_machine_edge_features[job_machine_connectivity == 1]
        job_job_edge_count = edge_index.shape[1] - job_machine_edge_features.shape[0]
        job_job_edge_features = torch.zeros(job_job_edge_count, 3, device=self.device)
        edge_features = torch.cat([job_machine_edge_features, job_job_edge_features])

        # Encode nodes and edges
        x_job_node: torch.Tensor = self.job_encoder(job_features)
        x_machine_node: torch.Tensor = self.machine_encoder(machine_features)
        x = torch.cat([x_job_node, x_machine_node])
        x_edge_attr: torch.Tensor = self.edge_encoder(edge_features)

        # Get embeddings
        batch = torch.zeros(n_jobs + n_machines, dtype=torch.long, device=self.device)
        node_embeddings = self.graph_network(x, edge_index=edge_index, edge_attr=x_edge_attr)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        graph_embedding = global_mean_pool(node_embeddings, batch=batch)

        return node_embeddings, edge_embeddings, graph_embedding


# Gin Actor
# ----------------------------------------------------------------------------------------------------------------------


class GinEActor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.network = BaseGinENetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinEAgentObsTensor) -> torch.Tensor:
        n_jobs = obs.task_vm_compatibility.shape[0]
        n_machines = obs.task_vm_compatibility.shape[1]

        _, edge_embeddings, graph_embedding = self.network(obs)

        # Get edge embedding scores
        rep_graph_embedding = graph_embedding.expand(edge_embeddings.shape[0], self.embedding_dim)
        edge_embeddings = torch.cat([edge_embeddings, rep_graph_embedding], dim=1)
        edge_embedding_scores: torch.Tensor = self.edge_scorer(edge_embeddings)

        # Extract the job-machine edges only
        job_machine_connectivity = obs.task_vm_compatibility.flatten()
        job_machine_edge_count = job_machine_connectivity.sum()
        job_machine_edge_embeddings = edge_embedding_scores[:job_machine_edge_count]

        # Actions scores should be the value in edge embedding, but -inf on invalid actions
        action_scores = torch.ones_like(job_machine_connectivity, dtype=torch.float32, device=self.device) * -1e8
        action_scores[job_machine_connectivity == 1] = job_machine_edge_embeddings.flatten()
        action_scores = action_scores.reshape(n_jobs, n_machines)
        action_scores[obs.task_state_ready == 0, :] = -1e8  # Remove scores of actions with not ready tasks

        return action_scores


# Gin Critic
# ----------------------------------------------------------------------------------------------------------------------


class GinECritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.network = BaseGinENetwork(
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
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinEAgentObsTensor) -> torch.Tensor:
        # Critic value is derived from global graph state
        _, _, graph_embedding = self.network(obs)
        return self.graph_scorer(graph_embedding.unsqueeze(0))


# Gin Agent
# ----------------------------------------------------------------------------------------------------------------------


class GinEAgent(Agent, nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.mapper = GinEAgentMapper(MAX_OBS_SIZE)
        self.actor = GinEActor(hidden_dim=32, embedding_dim=32, device=device)
        self.critic = GinECritic(hidden_dim=32, embedding_dim=32, device=device)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = self.mapper.unmap(x[batch_index])
            value = self.critic(decoded_obs)
            values.append(value)

        return torch.stack(values).to(self.device)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            decoded_obs = self.mapper.unmap(x[batch_index])
            action_scores = self.actor(decoded_obs)
            action_scores = action_scores.flatten()
            action_probabilities = softmax(action_scores, dim=0)

            probs = Categorical(action_probabilities)
            chosen_action = action[batch_index] if action is not None else probs.sample()
            value = self.critic(decoded_obs)

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(probs.log_prob(chosen_action))
            all_entropies.append(probs.entropy())
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions).to(self.device)
        log_probs = torch.stack(all_log_probs).to(self.device)
        entropies = torch.stack(all_entropies).to(self.device)
        values = torch.stack(all_values).to(self.device)

        return chosen_actions, log_probs, entropies, values
