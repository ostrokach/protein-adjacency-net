import torch
import torch.nn as nn


class PermutePad(nn.Module):
    def __init__(self, padding=1):
        super().__init__()
        self.padding = 1

    def forward(self, x):
        x = torch.cat([x[:, :, -self.padding :], x, x[:, :, : self.padding]], 2)
        return x.contiguous()


class PermutePadTranspose(nn.Module):
    def __init__(self, padding=2):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        top = x[:, :, : self.padding].clone()
        x[:, :, : self.padding] += x[:, :, -self.padding :]
        x[:, :, -self.padding :] += top
        return x
