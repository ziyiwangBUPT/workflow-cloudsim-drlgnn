import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from icecream import ic

from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import dense_to_sparse


class CnnNetwork(nn.Module):
    def __init__(self, height: int, width: int, channels: int, out_dim: int):
        super().__init__()

        self.height = height
        self.width = width
        self.channels = channels

        conv_out_size = 16 * ((self.height - 2) // 2 - 2) // 2 * ((self.width - 2) // 2 - 2) // 2

        self.conv1 = nn.Conv2d(channels, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched forward.

        :param x: (batch_size, height, width, channels)
        :return: (batch_size, out_dim)
        """

        assert x.shape[1] == self.height
        assert x.shape[2] == self.width
        assert x.shape[3] == self.channels
        x = torch.moveaxis(x, 3, 1)

        # Rule: out = (in - K + 1) = (in - 2)
        c1 = F.relu(self.conv1(x))  # c1: (B, 8, h - 2, w - 2)
        s1 = F.max_pool2d(c1, (2, 2))  # s1: (B, 8, (h - 2)//2, (w - 2)//2)
        c2 = F.relu(self.conv2(s1))  # c2: (B, 16, (h - 2)//2 - 2, (w - 2)//2 - 2)
        s2 = F.max_pool2d(c2, (2, 2))  # s2: (B, 16, ((h - 2)//2 - 2)//2, ((w - 2)//2 - 2)//2)
        s3 = torch.flatten(s2, start_dim=1)  # s3: (B, 16 * ((h - 2)//2 - 2)//2 * ((w - 2)//2 - 2)//2)
        f4 = F.relu(self.fc1(s3))  # f4: (B, 128)
        f5 = F.relu(self.fc2(f4))  # f4: (B, 64)
        return F.relu(self.fc3(f5))  # f4: (B, out_dim)


class GraphNetwork(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        )
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
        )
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, out_dim)

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
        h_pool = F.relu(self.fc1(h_pool))
        h_pool = F.dropout(h_pool, p=0.5, training=self.training)
        h_pool = F.relu(self.fc2(h_pool))

        return h_pool


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


if __name__ == "__main__":
    torch.manual_seed(0)
    net = CnnNetwork(10, 15, 3, 64)
    x = torch.rand(1, 10, 15, 3)
    y = net(x)
    ic(x.shape, y.shape)

    g_net = GraphNetwork(3, 64)
    x = torch.rand(7, 3)
    adj = torch.randint(0, 2, (1, 7, 3))
    y = g_net(x, adj)
    ic(x.shape, y.shape)
