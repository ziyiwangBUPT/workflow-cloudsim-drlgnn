import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class JobActorCriticAgent(nn.Module):
    def __init__(self, n_jobs, n_machines, input_dim, hidden_dim, device=torch.device("cpu")):
        super().__init__()
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Graph Encoder, Actor and Critic Networks
        self.encoder = nn.ModuleList(
            [
                GCNConv(input_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim),
            ]
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.pooled_machine_input = nn.Parameter(torch.Tensor(hidden_dim).uniform_(-1, 1).to(device))

    def encode_graph(self, x, edge_index, batch):
        for layer in self.encoder:
            x = F.relu(layer(x, edge_index))
        return global_mean_pool(x, batch)

    def get_value(self, x):
        features, edge_index, batch, _ = x
        global_graph_embedding = self.encode_graph(features, edge_index, batch)
        return self.critic(global_graph_embedding)

    def get_action_and_value(self, x, action=None):
        features, edge_index, batch, mask = x

        global_graph_embedding = self.encode_graph(features, edge_index, batch)

        # Pool and compute job action scores
        machine_pooling_vector_rep = self.pooled_machine_input[None, :].expand_as(global_graph_embedding)
        concat_feats = torch.cat((global_graph_embedding, machine_pooling_vector_rep), dim=-1)
        job_action_scores = self.actor(concat_feats).squeeze(-1) * 10
        job_action_scores = job_action_scores.masked_fill(mask.bool(), float("-inf"))

        # Action sampling
        job_action_probabilities = F.softmax(job_action_scores, dim=1)
        probs = Categorical(logits=job_action_probabilities)
        chosen_action = action if action is not None else probs.sample()

        # Calculate log-prob and entropy
        log_prob = probs.log_prob(chosen_action)
        entropy = probs.entropy()
        value = self.critic(global_graph_embedding)

        return chosen_action, log_prob, entropy, value


if __name__ == "__main__":
    # Example usage (data loading and preparation would be required)
    # Initialize agent
    agent = JobActorCriticAgent(n_jobs=3, n_machines=5, input_dim=7, hidden_dim=64)

    # Create synthetic graph data
    features = torch.rand((30, 7))  # Assume 30 nodes
    edge_index = torch.randint(0, 30, (2, 100))  # Random adjacency
    batch = torch.zeros(30, dtype=torch.long)  # Dummy batch indexing
    mask = torch.zeros(30, 3, dtype=torch.bool)  # Mask for incompatible machines

    # Forward pass
    action, log_prob, entropy, value = agent.get_action_and_value((features, edge_index, batch, mask))
