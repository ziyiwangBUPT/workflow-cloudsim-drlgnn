from typing import Optional
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from gym_simulator.algorithms.graph.graphcnn import GraphCNN
from gym_simulator.algorithms.graph.mlp import MLPActor, MLPCritic


class JobActor(nn.Module):
    """
    Job Actor model for job scheduling tasks in reinforcement learning.
    """

    def __init__(
        self,
        n_jobs: int,
        n_machines: int,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        feature_extract_num_mlp_layers: int,
        critic_num_mlp_layers: int,
        critic_hidden_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.device = device

        # Graph-based encoder and actor-critic networks
        self.encoder = GraphCNN(num_layers, feature_extract_num_mlp_layers, input_dim, hidden_dim, device)
        self.pooled_machine_input = nn.Parameter(torch.Tensor(hidden_dim).uniform_(-1, 1).to(device))
        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        self.critic = MLPCritic(critic_num_mlp_layers, hidden_dim, critic_hidden_dim, 1).to(device)

    def forward(
        self,
        features: torch.Tensor,  # (batch_size * num_nodes, input_dim)
        graph_pool: torch.Tensor,  # (batch_size, batch_size * num_nodes)
        adj: torch.Tensor,  # (batch_size * num_nodes, batch_size * num_nodes)
        candidate: torch.Tensor,  # (batch_size, n_jobs)
        mask: torch.Tensor,  # (batch_size, n_jobs)
    ):
        # Encode graph structure and node features
        global_graph_embedding, node_embeddings = self.encoder(x=features, graph_pool=graph_pool, adj=adj)

        # h^L_v,t: Extract node embeddings for candidate jobs
        candidate_expanded = candidate.unsqueeze(-1).expand(-1, self.n_jobs, node_embeddings.size(-1))
        batch_node = node_embeddings.reshape(candidate_expanded.size(0), -1, candidate_expanded.size(-1))
        job_node_embeddings = torch.gather(batch_node, 1, candidate_expanded)

        # Repeat pooled features and concatenate for actor
        global_graph_embedding_rep = global_graph_embedding.unsqueeze(-2).expand_as(job_node_embeddings)
        machine_pooling_vector_rep = self.pooled_machine_input[None, None, :].expand_as(job_node_embeddings)
        concat_feats = torch.cat(
            #    h^L_v,t         || h^t_G                || u_t
            (job_node_embeddings, global_graph_embedding_rep, machine_pooling_vector_rep),
            dim=-1,
        )
        ic(concat_feats.shape)

        # c^o_t,k: Calculate scores and apply mask
        job_action_scores: torch.Tensor = self.actor(concat_feats) * 10
        job_action_scores = job_action_scores.squeeze(-1)
        job_action_scores = job_action_scores.masked_fill(mask.squeeze(1).bool(), float("-inf"))
        ic(job_action_scores.shape)

        # p_j(a^m_t): Softmax for probability distribution
        job_action_probabilities = F.softmax(job_action_scores, dim=1)
        ic(job_action_probabilities.shape)
        ic(job_action_probabilities[0])

        return job_action_probabilities


def compute_graph_pooling_matrix(
    batch_size: int, num_jobs: int, num_machines: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Compute graph pooling matrix for the job scheduling task.

    @param batch_size: Batch size
    @param num_jobs: Number of jobs
    @param num_machines: Number of machines
    @param device: Device to use for computation

    @return graph_pooling_matrix: Graph pooling matrix (batch_size, batch_size * num_jobs * num_machines)
    """

    n_nodes = num_jobs * num_machines

    pooling_value = 1 / n_nodes
    pooling_values = torch.full((batch_size * n_nodes,), pooling_value, dtype=torch.float32, device=device)

    row_indices = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)
    col_indices = torch.arange(batch_size * n_nodes, device=device)
    indices = torch.stack([row_indices, col_indices])
    graph_pooling_matrix = torch.sparse_coo_tensor(
        indices, pooling_values, (batch_size, batch_size * n_nodes), device=device
    )

    return graph_pooling_matrix


if __name__ == "__main__":
    job_actor = JobActor(
        n_jobs=30,
        n_machines=20,
        num_layers=2,
        input_dim=64,
        hidden_dim=64,
        feature_extract_num_mlp_layers=2,
        critic_num_mlp_layers=2,
        critic_hidden_dim=64,
    )
    result = job_actor(
        features=torch.rand(4 * 30 * 20, 64),
        graph_pool=compute_graph_pooling_matrix(4, 30, 20),
        adj=torch.rand(4 * 30 * 20, 4 * 30 * 20),
        candidate=torch.randint(0, 2, (4, 30)),
        mask=torch.randint(0, 2, (4, 30)),
    )
