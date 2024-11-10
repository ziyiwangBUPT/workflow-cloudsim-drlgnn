import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic


class CnnNetwork(nn.Module):
    def __init__(self, height: int, width: int, channels: int, out_dim: int):
        super().__init__()

        self.height = height
        self.width = width
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(1024, 128)
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
        p3 = self.pool(s2)  # p3: (B, 16, 8, 8)
        s3 = torch.flatten(p3, start_dim=1)  # s3: (B, 1024)
        f4 = F.relu(self.fc1(s3))  # f4: (B, 128)
        f5 = F.relu(self.fc2(f4))  # f4: (B, 64)
        return self.fc3(f5)  # f4: (B, out_dim)
