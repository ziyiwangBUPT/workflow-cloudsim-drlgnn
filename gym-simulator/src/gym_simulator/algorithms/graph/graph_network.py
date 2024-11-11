import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from icecream import ic

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse


class GraphNetwork(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 32):
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
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Unbatched forward.

        :param x: (num_nodes, input_dim)
        :param adj: (num_nodes, num_nodes)
        :return: (1, out_dim)
        """

        edge_index, _ = dense_to_sparse(adj)
        batch = torch.zeros(adj.shape[0], dtype=torch.long)

        h: torch.Tensor = self.conv1(x, edge_index)
        h_pool = global_mean_pool(h, batch)
        h_pool = self.fc(h_pool)

        return h_pool
