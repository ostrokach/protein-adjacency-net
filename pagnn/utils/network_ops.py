from contextlib import contextmanager
from typing import Generator

import torch.nn as nn


@contextmanager
def eval_net(net: nn.Module) -> Generator:
    training = net.training
    try:
        net.train(False)
        yield
    finally:
        net.train(training)


def freeze_net(net_d):
    for p in net_d.parameters():
        p.requires_grad = False


def unfreeze_net(net_d):
    for p in net_d.parameters():
        p.requires_grad = True


def freeze_adj_conv(net_d):
    for m in net_d.modules():
        if isinstance(m, nn.Conv1d) and m.kernel_size == (2,):
            for p in m.parameters():
                p.requires_grad = False


def unfreeze_adj_conv(net_d):
    for m in net_d.modules():
        if isinstance(m, nn.Conv1d) and m.kernel_size == (2,):
            for p in m.parameters():
                p.requires_grad = True
