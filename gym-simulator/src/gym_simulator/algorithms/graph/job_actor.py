from dataclasses import dataclass
import re
from typing import Optional, Tuple
import icecream
from numpy import long
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from icecream import ic

from gym_simulator.algorithms.graph.graphcnn import GraphCNN
from gym_simulator.algorithms.graph.mlp import MLPActor, MLPCritic


class JobActorCriticAgent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        hidden_dim: int = 128,
        feature_extract_num_mlp_layers: int = 3,
        critic_num_mlp_layers: int = 2,
        critic_hidden_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

        # Define Encoder, Actor, and Critic
        self.encoder = GraphCNN(num_layers, feature_extract_num_mlp_layers, input_dim, hidden_dim, device)
        self.actor = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        self.critic = MLPCritic(critic_num_mlp_layers, hidden_dim, critic_hidden_dim, 1).to(device)

        # A learnable parameter that is used to represent the global state of the machine pool
        self.pooled_machine_input = nn.Parameter(torch.Tensor(hidden_dim).uniform_(-1, 1).to(device))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the value estimation from critic network."""

        values = []
        for i in range(x.shape[0]):
            (n_jobs, _), features, adj, _, _ = extract_features(x[i], self.input_dim)
            graph_pool = torch.full((1, n_jobs), 1 / n_jobs, dtype=torch.float32, device=self.device)
            global_graph_embedding, _ = self.encoder(features[i].unsqueeze(0), graph_pool, adj[i].unsqueeze(0))
            values.append(self.critic(global_graph_embedding))

        return torch.cat(values, dim=0)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action, log-prob, entropy, and value.

        @param x: Input tensor (batch_size, input_dim + 3 * n_jobs + adj_dim)
        @param action: Action tensor (batch_size, n_jobs)

        @return:
            - Chosen action tensor (batch_size, n_jobs)
            - Log-probability tensor (batch_size, n_jobs)
            - Entropy tensor (batch_size, n_jobs)
            - Value tensor (batch_size, 1)
        """
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for i in range(x.shape[0]):
            (n_jobs, _), features, adj, candidate, mask = extract_features(x[i], self.input_dim)
            ic(n_jobs, features.shape, adj.shape, candidate.shape, mask.shape)

            graph_pool = torch.full((1, n_jobs), 1 / n_jobs, dtype=torch.float32, device=self.device)
            global_graph_embedding, node_embeddings = self.encoder(features, graph_pool, adj)
            ic(graph_pool.shape, global_graph_embedding.shape, node_embeddings.shape)

            # Node embeddings for candidate jobsq
            candidate_expanded = candidate.unsqueeze(-1).expand(n_jobs, node_embeddings.size(-1))
            job_node_embeddings = torch.gather(node_embeddings, 1, candidate_expanded)
            ic(candidate_expanded.shape, job_node_embeddings.shape)

            # Combine features
            global_graph_embedding_rep = global_graph_embedding.expand_as(job_node_embeddings)
            machine_pooling_vector_rep = self.pooled_machine_input[None, :].expand_as(job_node_embeddings)
            concat_feats = torch.cat(
                (job_node_embeddings, global_graph_embedding_rep, machine_pooling_vector_rep), dim=-1
            )
            ic(concat_feats.shape)

            # Compute action scores and apply mask
            job_action_scores = self.actor(concat_feats).squeeze(-1) * 10
            job_action_scores = job_action_scores.masked_fill(mask.bool(), float("-inf"))
            ic(job_action_scores.shape)

            # Get probabilities and distribution
            job_action_probabilities = F.softmax(job_action_scores)
            probs = Categorical(logits=job_action_probabilities)
            chosen_action = action[i] if action is not None else probs.sample()
            ic(job_action_probabilities, probs, chosen_action)

            # Get log-prob and entropy
            log_prob = probs.log_prob(chosen_action)
            entropy = probs.entropy()
            value = self.critic(global_graph_embedding)

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
            all_values.append(value)

        return (
            torch.stack(all_chosen_actions),
            torch.stack(all_log_probs),
            torch.stack(all_entropies),
            torch.stack(all_values),
        )


def extract_features(
    x: torch.Tensor, input_dim: int
) -> Tuple[Tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features from input tensor.

    @param x: Input tensor
    @param input_dim: Input dimension

    @return:
        - Tuple of (n_jobs, n_machines)
        - Features tensor (batch_size, n_jobs * n_machines, input_dim)
        - Adjacency matrix tensor (batch_size, n_jobs * n_machines, n_jobs * n_machines)
        - Candidate tensor (batch_size, n_jobs)
        - Mask tensor (batch_size, n_jobs)
    """

    n_jobs = x[0].long()
    n_machines = x[1].long()
    x = x[2:]

    features_size = n_jobs * input_dim
    adj_size = n_jobs * n_jobs
    candidate_size = n_jobs
    mask_size = n_jobs

    ic(n_jobs, n_machines, x.shape)
    ic(features_size, adj_size, candidate_size, mask_size)
    assert x.shape[0] >= features_size + adj_size + candidate_size + mask_size, "Invalid input tensor shape"
    return (
        (n_jobs, n_machines),
        x[:features_size].reshape(n_jobs, input_dim),
        x[features_size : features_size + adj_size].reshape(n_jobs, n_jobs),
        x[features_size + adj_size : features_size + adj_size + candidate_size].long(),
        x[features_size + adj_size + candidate_size : features_size + adj_size + candidate_size + mask_size],
    )
