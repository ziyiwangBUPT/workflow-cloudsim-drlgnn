import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from icecream import ic

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse


class GraphNetwork(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        self.fc2 = nn.Linear(hidden_dim * 3, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Unbatched forward.

        :param x: (num_nodes, input_dim)
        :param adj: (num_nodes, num_nodes)
        :return: (1, out_dim)
        """

        edge_index, _ = dense_to_sparse(adj)
        batch = torch.zeros(adj.shape[0], dtype=torch.long)

        h1: torch.Tensor = self.conv1(x, edge_index)
        h2: torch.Tensor = self.conv2(h1, edge_index)
        h3: torch.Tensor = self.conv3(h2, edge_index)

        h1_pool = global_mean_pool(h1, batch)
        h2_pool = global_mean_pool(h2, batch)
        h3_pool = global_mean_pool(h3, batch)

        h_pool = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)
        h_pool = self.fc1(h_pool)
        h_pool = F.dropout(h_pool, p=0.5, training=self.training)
        h_pool = self.fc2(h_pool)

        return h_pool
