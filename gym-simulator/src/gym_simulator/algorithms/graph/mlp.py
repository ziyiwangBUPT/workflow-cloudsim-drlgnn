import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with an optional linear output layer.
    """

    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        assert num_layers >= 1, "Number of layers should be positive!"
        self.linears = create_linears(num_layers, input_dim, hidden_dim, output_dim)
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in range(len(self.linears) - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[-1](h)


class MLPActor(nn.Module):
    """
    Actor model using MLP for reinforcement learning.
    """

    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        assert num_layers >= 1, "Number of layers should be positive!"
        self.linears = create_linears(num_layers, input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in range(len(self.linears) - 1):
            h = torch.tanh(self.linears[layer](h))
        return self.linears[-1](h)


class MLPCritic(nn.Module):
    """
    Critic model using MLP for reinforcement learning.
    """

    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        assert num_layers >= 1, "Number of layers should be positive!"
        self.linears = create_linears(num_layers, input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in range(len(self.linears) - 1):
            h = torch.tanh(self.linears[layer](h))
        return self.linears[-1](h)


# ------------------------------ Helper Functions ------------------------------


def create_linears(num_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> nn.ModuleList:
    linears = nn.ModuleList()

    # input_dim -> output_dim
    if num_layers == 1:
        linears.append(nn.Linear(input_dim, output_dim))
        return linears

    # input_dim -> hidden_dim -> hidden_dim -> ... -> hidden_dim -> output_dim
    linears.append(nn.Linear(input_dim, hidden_dim))
    for _ in range(num_layers - 2):
        linears.append(nn.Linear(hidden_dim, hidden_dim))
    linears.append(nn.Linear(hidden_dim, output_dim))
    return linears
