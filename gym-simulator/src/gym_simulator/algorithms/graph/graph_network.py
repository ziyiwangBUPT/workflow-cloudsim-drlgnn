import torch
import torch.nn as nn
from torch_geometric.nn import GIN

from icecream import ic

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse


class GraphNetwork(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.gin = GIN(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=out_dim,
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Unbatched forward.

        :param x: (num_nodes, input_dim)
        :param adj: (num_nodes, num_nodes)
        :return: (1, out_dim)
        """

        edge_index, _ = dense_to_sparse(adj)
        batch = torch.zeros(adj.shape[0], dtype=torch.long)

        h: torch.Tensor = self.gin(x, edge_index)
        h_pool = global_mean_pool(h, batch)
        return h_pool
