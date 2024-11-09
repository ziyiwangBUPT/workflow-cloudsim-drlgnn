import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

from gym_simulator.algorithms.graph.mlp import MLP


class GraphCNN(nn.Module):
    """
    Graph Convolutional Neural Network (GCNN) with MLP-based aggregation.
    """

    def __init__(
        self,
        num_layers: int,
        num_mlp_layers: int,
        input_dim: int,
        hidden_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.device = device
        self.num_layers = num_layers

        # Initialize MLP layers and batch norms for each graph layer
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers - 1):
            # If it's the first layer, input dimension is `input_dim`; otherwise, it's `hidden_dim`
            self.mlps.append(MLP(num_mlp_layers, input_dim if layer == 0 else hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(
        self, x: torch.Tensor, graph_pool: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for GraphCNN.

        @param x: Node features (num_nodes, input_dim)
        @param graph_pool: Graph-level pooling tensor (batch_size, num_nodes)
        @param adj: Adjacency matrix (num_nodes, num_nodes)

        @return h_pooled: Pooled features of graph actions (batch_size, hidden_dim)
        @return h_nodes: Node features after all layers (num_nodes, hidden_dim)
        """

        h = x.to(self.device)
        for layer in range(self.num_layers - 1):
            # Graph convolution operation
            pooled = torch.mm(adj, h)
            degree = torch.mm(adj, torch.ones((adj.shape[0], 1), device=self.device))
            pooled = pooled / (degree + 1e-8)

            # MLP-based aggregation for node representations
            pooled_rep = self.mlps[layer](pooled)
            h = F.relu(self.batch_norms[layer](pooled_rep))

        ic(graph_pool.shape)
        ic(h.shape)
        h_nodes = h.clone()
        h_pooled = torch.sparse.mm(graph_pool, h)

        return h_pooled, h_nodes


if __name__ == "__main__":
    torch.manual_seed(42)
    graphcnn = GraphCNN(num_layers=3, num_mlp_layers=2, input_dim=512, hidden_dim=512)
    result = graphcnn(
        x=torch.rand(64, 512),
        graph_pool=torch.rand(32, 64),
        adj=torch.rand(64, 64),
    )
    assert result[0].shape == (32, 512)
    assert result[1].shape == (64, 512)
    ic(result[0][0].sum())
