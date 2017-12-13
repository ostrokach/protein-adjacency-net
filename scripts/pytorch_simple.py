# flake8: noqa
import os
import os.path as op
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numba import boolean, float32, int32, int64, jit
from scipy import signal, stats
from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm, tqdm_notebook

# %matplotlib inline

# %%

filter_ = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)

_cutoff = 0.7

num_pos = 0
for i in range(1_000):
    data = generate_data(1_000, 5)
    if label_data(data, filter_, _cutoff):
        num_pos += 1
print(num_pos / (i + 1))

# %% Unit tests

test_array_1 = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=np.float32)

assert count_matches(test_array_1, filter_) == 1

test_array_2 = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.float32)

assert count_matches(test_array_2, filter_) == 1

test_array_3 = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=np.float32)

assert count_matches(test_array_3, filter_) == 0

test_array_4 = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=np.float32)

assert count_matches(test_array_4, filter_) == 1

# %% Parameters

FRAC_NEGATIVE = 0.7
BATCH_SIZE = 64
NUM_BATCHES = 200
NUM_FEATURES = 5

MIN_DOMAIN_LENGTH = 40
MAX_DOMAIN_LENGTH = 600

# %% Define network


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        n_filters = 2
        self.conv = nn.Conv1d(5, n_filters, 2, stride=2, bias=False)
        self.combine_filters = nn.Linear(n_filters, 1, bias=False)

    def forward(self, input_, multiplier):
        x = self.conv(input_)
        # x = F.relu(x)

        # x = F.avg_pool1d(x, x.size()[2]).sum(dim=1)
        # x = x.sum(dim=2) / input_.size()[0]

        x = x.sum(dim=2).sum(dim=1) / x.size()[2]

        # x = x.sum(dim=2) / x.size()[2]
        # x = self.combine_filters(x).squeeze()

        x = F.sigmoid(x)
        return x


# %% Test data

test_data_ = np.stack([generate_data(64, 5).T for _ in range(512)])

test_targets_ = np.array(
    [
        label_data(test_data_[i, :, :].T, filter_, FRAC_NEGATIVE)
        for i in range(test_data_.shape[0])
    ],
    dtype=np.float32)

# %% Train network

net = Net()

# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()

# create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)

test_data = Variable(torch.Tensor(test_data_))
test_targets = Variable(torch.Tensor(test_targets_.astype(np.float32)), requires_grad=False)

# === Run ===
train_performance = []
validation_performance = []
for i in tqdm(range(NUM_BATCHES)):

    training_data = Variable(
        torch.Tensor(np.stack([generate_data(64, 5).T for _ in range(BATCH_SIZE)])))

    training_targets = Variable(
        torch.Tensor(
            np.array(
                [
                    label_data(training_data.data.numpy()[i, :, :].T, filter_, FRAC_NEGATIVE)
                    for i in range(training_data.size()[0])
                ],
                dtype=np.float32)),
        requires_grad=False)

    if not training_targets.data.numpy().any().any():
        print(f"Skipping {i}...")
        continue

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(training_data)
    loss = criterion(output, training_targets)
    loss.backward()
    optimizer.step()  # Does the update

    if i % 10 == 0:
        score = metrics.roc_auc_score(training_targets.data.numpy().astype(int),
                                      output.data.numpy())
        train_performance.append(score)

        predictions = net(test_data)
        score = metrics.roc_auc_score(test_targets.data.numpy().astype(int),
                                      predictions.data.numpy())
        validation_performance.append(score)

plt.plot(range(0, len(train_performance) * 10, 10), train_performance, label='train')
plt.plot(range(0, len(validation_performance) * 10, 10), validation_performance, label='test')
plt.legend()

# %%

plt.plot(test_targets_.astype(int), predictions.data.numpy(), 'rx')

plt.plot(test_targets_.astype(int), F.sigmoid(predictions).data.numpy(), 'rx')
test_targets_[:5]
F.sigmoid(predictions).data.numpy()[:5]

# %%

list(net.parameters())[0]

p = list(net.parameters())[0]

p.data.numpy().squeeze().shape

plt.imshow(filter_, cmap=plt.cm.Greys)
plt.colorbar()
plt.show()

plt.imshow(p[0, :, :].data.numpy().T.squeeze(), cmap=plt.cm.Greys)
plt.colorbar()
plt.show()

plt.imshow(p[1, :, :].data.numpy().T.squeeze(), cmap=plt.cm.Greys)
plt.colorbar()
plt.show()

plt.imshow(p[2, :, :].data.numpy().T.squeeze(), cmap=plt.cm.Greys)
plt.colorbar()
plt.show()
