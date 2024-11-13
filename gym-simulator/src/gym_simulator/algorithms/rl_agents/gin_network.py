import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import dense_to_sparse
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GIN, global_mean_pool

from icecream import ic


class GinNetwork(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int = 64, embedding_dim: int = 8) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.embedding_dim = embedding_dim

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
        self.connection_encoder = nn.Sequential(
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
    ) -> torch.Tensor:
        """
        :return: n_jobs * n_machines vector
        """

        job_nodes = torch.arange(self.n_jobs)
        job_nodes = job_nodes.unsqueeze(1).expand(self.n_jobs, self.n_machines)
        machine_nodes = torch.arange(self.n_machines) + self.n_jobs
        machine_nodes = machine_nodes.unsqueeze(0).expand(self.n_jobs, self.n_machines)

        edges = torch.cat([job_nodes.flatten().unsqueeze(1), machine_nodes.flatten().unsqueeze(1)], dim=1)
        condition = task_vm_compatibility.flatten()
        edges = edges[condition == 1]

        edge_index, _ = dense_to_sparse(adj)
        edge_index = torch.cat([edges.T, edge_index], dim=1)

        job_features = torch.cat(
            [
                task_state_scheduled.unsqueeze(1),
                task_state_ready.unsqueeze(1),
                task_completion_time.unsqueeze(1),
            ],
            dim=1,
        )
        machine_features = vm_completion_time.unsqueeze(1)

        connection_features = task_vm_time_cost.reshape(-1, 1)
        connection_features = connection_features[condition == 1]

        x_job_node = self.job_encoder(job_features)
        x_machine_node = self.machine_encoder(machine_features)
        x = torch.cat([x_job_node, x_machine_node])

        edge_attr = self.connection_encoder(connection_features)
        node_node_edges = edge_index.shape[1] - edge_attr.shape[0]
        edge_attr = torch.cat([edge_attr, torch.zeros(node_node_edges, self.embedding_dim)], dim=0)

        batch = torch.zeros(self.n_jobs + self.n_machines, dtype=torch.long)
        node_embeddings = self.graph_network(x, edge_index=edge_index, edge_attr=edge_attr)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        graph_embedding = global_mean_pool(node_embeddings, batch=batch)

        return node_embeddings, edge_embeddings, graph_embedding


class GinActorNetwork(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int = 64, embedding_dim: int = 8) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines

        self.network = GinNetwork(n_jobs, n_machines, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        self.edge_scorer = nn.Linear(2 * embedding_dim, 1)
        self.mapper = nn.Sequential(
            nn.Linear(n_jobs * n_machines, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_jobs * n_machines),
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
        edge_embedding_scores = self.edge_scorer(edge_embeddings)

        condition = task_vm_compatibility.flatten()
        job_machine_edge_embeddings = edge_embedding_scores[: condition.sum()]

        edge_scores = torch.zeros_like(condition, dtype=torch.float32)
        edge_scores[condition == 1] = job_machine_edge_embeddings.flatten()

        return self.mapper(edge_scores.unsqueeze(0)).flatten()


class GinCriticNetwork(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int = 64, embedding_dim: int = 8) -> None:
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines

        self.network = GinNetwork(n_jobs, n_machines, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
        self.mapper = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
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

        return self.mapper(graph_embedding.unsqueeze(0))
