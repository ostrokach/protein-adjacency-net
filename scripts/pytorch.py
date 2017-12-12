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

%matplotlib inline


# %% Parameters

FRAC_NEGATIVE = 0.7
BATCH_SIZE = 64
NUM_BATCHES = 200
NUM_FEATURES = 5

MIN_DOMAIN_LENGTH = 40
MAX_DOMAIN_LENGTH = 600


# %% Training data

random.seed(42)
np.random.seed(42)

data_list = []
targets_list = []

n_samples = int(MAX_DOMAIN_LENGTH**2)
data_template = generate_data(n_samples, NUM_FEATURES)

for _ in tqdm(range(NUM_BATCHES)):
    _n_samples = int(random.choice(range(MIN_DOMAIN_LENGTH, MAX_DOMAIN_LENGTH))**2)
    _data_list = []
    _targets_list = []

    for j in range(BATCH_SIZE):
        permutation = np.random.permutation(data_template.shape[0])[:_n_samples]

        _data = data_template[permutation, :]
        _data_list.append(_data.T)

        _target = label_data(_data, filter_, FRAC_NEGATIVE)
        _targets_list.append(_target)

    data = np.stack(_data_list)
    data_list.append(data)

    targets = np.array(_targets_list)
    targets_list.append(targets)


# %% Test data

test_data_ = np.stack([generate_data(64, 5).T for _ in range(512)])

test_targets_ = np.array(
    [label_data(test_data_[i, :, :].T, filter_, FRAC_NEGATIVE) for i in range(test_data_.shape[0])],
    dtype=np.float32)


# %% Define network


# %%

torch.cat([test_data, torch.ones(test_data.size()[0], 1, 1) * 42], 0)

test_data.size()
(torch.ones(test_data.size()[0], 1) * 42).size()

# %% Train network

net = Net()

# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()

# create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0.01)

test_data = Variable(torch.Tensor(test_data_))
test_targets = Variable(torch.Tensor(test_targets_.astype(np.float32)), requires_grad=False)

# === Run ===
train_performance = []
validation_performance = []
for i in tqdm(range(len(data_list))):

    training_data = Variable(torch.Tensor(data_list[i]))
    training_targets = Variable(
        torch.Tensor(targets_list[i].astype(np.float32)), requires_grad=False)

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
        score = metrics.roc_auc_score(
            training_targets.data.numpy().astype(int), output.data.numpy())
        train_performance.append(score)

        predictions = net(test_data)
        score = metrics.roc_auc_score(
            test_targets.data.numpy().astype(int), predictions.data.numpy())
        validation_performance.append(score)
        print(score)

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

# %%

i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
torch.sparse.FloatTensor(i.t(), v, torch.Size([2,3])).to_dense()
