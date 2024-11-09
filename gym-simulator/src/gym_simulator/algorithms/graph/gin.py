# https://mlabonne.github.io/blog/posts/2022-04-25-Graph_Isomorphism_Network.html
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ModuleList, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse


class GINActor(torch.nn.Module):
    """GIN"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        )
        self.model = Sequential(
            Linear(hidden_dim * 3, hidden_dim * 3),
            ReLU(),
            Dropout(p=0.5),
            Linear(hidden_dim * 3, 1),
        )

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        :param features: (n_jobs, input_dim)
        :param adj: (n_jobs, n_jobs)

        :return graph_embedding: (hidden_dim,)
        :return node_embedding: (n_jobs, hidden_dim)
        """
        edge_index, _ = dense_to_sparse(adj)
        batch = torch.zeros(adj.shape[0], dtype=torch.long)

        # Node embeddings
        h1: torch.Tensor = self.conv1(features, edge_index)  # h1: (n_jobs, hidden_dim)
        h2: torch.Tensor = self.conv2(h1, edge_index)  # h2: (n_jobs, hidden_dim)
        h3: torch.Tensor = self.conv3(h2, edge_index)  # h3: (n_jobs, hidden_dim)

        # Graph-level readout
        h1_pool = global_mean_pool(h1, batch)  # h1_pool: (1, hidden_dim)
        h2_pool = global_mean_pool(h2, batch)  # h2_pool: (1, hidden_dim)
        h3_pool = global_mean_pool(h3, batch)  # h3_pool: (1, hidden_dim)

        # Concatenate graph embeddings
        h = torch.cat((h1_pool, h2_pool, h3_pool), dim=1)  # h: (1, 3*hidden_dim)
        h_out: torch.Tensor = self.model(h)  # h_out: (1, 1)

        return F.log_softmax(h_out, dim=1).squeeze(0)
