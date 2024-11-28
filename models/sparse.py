import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bridge import Bridge


class SparseNet(nn.Module):
    def __init__(self, dropout_rate: float, in_channels: int = 3):
        super(SparseNet, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, 32, 1, 1),
            nn.GroupNorm(32, 32),
            nn.ReLU(),
            spconv.SubMConv2d(32, 64, 1, 1),
            nn.ReLU(),
            spconv.SparseMaxPool2d(2, 2),
            spconv.SubMConv2d(64, 64, 1, 1),
            nn.GroupNorm(64, 64),
            nn.ReLU(),
            spconv.SparseMaxPool2d(2, 2),
            spconv.ToDense(),
        )
        self.fc1 = nn.Linear(16384, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dp1 = nn.Dropout(p=dropout_rate)
        self.dp2 = nn.Dropout(p=dropout_rate)
        self.out_layer = Bridge(128, [1, 1], dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        # x: [N, 28, 28, 1], must be NHWC tensor
        x_sp = spconv.SparseConvTensor.from_dense(x)
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.net(x_sp)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp2(x)
        wc, sel = self.out_layer(x)
        return wc, sel
