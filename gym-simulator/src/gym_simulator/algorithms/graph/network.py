import torch
import torch.nn as nn

from icecream import ic


from gym_simulator.algorithms.graph.cnn_network import CnnNetwork
from gym_simulator.algorithms.graph.graph_network import GraphNetwork


class Network(nn.Module):
    def __init__(self, n_jobs: int, n_machines: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.cnn_network = CnnNetwork(n_jobs, n_machines, 3, hidden_dim)
        self.graph_network = GraphNetwork(3, hidden_dim)
        self.linear_network = nn.Sequential(
            nn.Linear(n_machines, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

        self.cat_network = nn.Sequential(
            nn.Linear(3 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        task_state_scheduled: torch.Tensor,
        task_state_ready: torch.Tensor,
        task_completion_time: torch.Tensor,
        vm_completion_time: torch.Tensor,
        task_vm_compatibility: torch.Tensor,
        task_vm_time_cost: torch.Tensor,
        task_vm_power_cost: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unbatched.

        :return: (1, 1)
        """

        # cnn_in: (batch_size, n_tasks, n_machines, 3)
        cnn_in = torch.cat(
            (
                task_vm_compatibility.unsqueeze(-1).unsqueeze(0),
                task_vm_time_cost.unsqueeze(-1).unsqueeze(0),
                task_vm_power_cost.unsqueeze(-1).unsqueeze(0),
            ),
            dim=-1,
        )
        ic(cnn_in)
        # graph_in: (batch_size, n_tasks, 3)
        graph_in = torch.cat(
            (
                task_state_scheduled.unsqueeze(-1),
                task_state_ready.unsqueeze(-1),
                task_completion_time.unsqueeze(-1),
            ),
            dim=-1,
        )
        ic(graph_in)
        # lin_in: (batch_size, n_machines)
        lin_in = vm_completion_time.unsqueeze(0)
        ic(lin_in)

        cnn_out = self.cnn_network(cnn_in)
        graph_out = self.graph_network(graph_in, adj)
        lin_out = self.linear_network(lin_in)

        # cocat: (batch_size, hidden_dim*3)
        concat = torch.cat((cnn_out, graph_out, lin_out), dim=-1)
        ic(cnn_out, graph_out, lin_out)
        return self.cat_network(concat)
