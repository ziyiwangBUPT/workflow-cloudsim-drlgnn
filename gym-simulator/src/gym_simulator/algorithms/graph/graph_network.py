import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features: int, num_edge_features: int, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [
            GATConv(self.num_features, self.hidden_size, edge_dim=num_edge_features),
            GATConv(self.hidden_size, self.hidden_size, edge_dim=num_edge_features),
        ]
        self.edge_scorer = nn.Linear(2 * self.hidden_size, self.target_size)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        src_nodes = edge_index[0]
        dest_nodes = edge_index[1]
        edge_embeddings = torch.cat([x[src_nodes], x[dest_nodes]], dim=1)

        return self.edge_scorer(edge_embeddings)
