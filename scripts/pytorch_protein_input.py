import concurrent.futures
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
from numba import boolean, float32, float64, int32, int64, jit
from scipy import signal, stats
from sklearn import metrics
from torch.autograd import Variable

from tqdm import tqdm, tqdm_notebook

# %matplotlib inline
# %run pytorch_simple.py

# %%

i = torch.LongTensor([[0, 1, 2], [0, 1, 2]])
v = torch.ones(i.size()[1])
adjacency = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
adjacency.to_dense()

t1 = torch.FloatTensor([
    #
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
# adjacency @ t1

# torch.mm(adjacency, t1)

# torch.mm(Variable(adjacency), Variable(t1))


DOMAIN_SIZE = 600

aa = generate_data(DOMAIN_SIZE, NUM_FEATURES)
adj = (np.random.rand(aa.shape[0], aa.shape[0]) < PROB_CONTACT).astype(np.int32)
adj_expanded = expand_adjacency(adj)
adj_expanded.shape

# %%




# %% Generate training and test data

NUM_FEATURES = 5
NUM_BATCHES = 200
BATCH_SIZE = 64
PROB_CONTACT = 0.1
PROB_REAL = 0.3
MIN_DOMAIN_SIZE = 40
MAX_DOMAIN_SIZE = 600

FILTER = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)

training_aa = []
training_adj = []
training_targets = []


def generate_training_data(domain_size):
    adj = (np.random.rand(domain_size, domain_size) < PROB_CONTACT).astype(np.int32)
    training_adj.append(adj)

    adj_expanded = expand_adjacency(adj)

    batch_aa = []
    batch_targets = []

    for j in range(BATCH_SIZE):
        aa = generate_data(domain_size, NUM_FEATURES)
        batch_aa.append(aa.T)

        data = (aa.T @ adj_expanded.T).T.astype(np.float32)
        target = label_data(data, FILTER, PROB_REAL)
        batch_targets.append(target)

    aa = np.stack(batch_aa)
    targets = np.array(batch_targets)

    return aa, adj, targets


with concurrent.futures.ProcessPoolExecutor(12) as p:
    domain_sizes = np.random.randint(MIN_DOMAIN_SIZE, MAX_DOMAIN_SIZE, NUM_BATCHES)
    training_data = list(p.map(generate_training_data, domain_sizes))

# %%

test_data = generate_training_data(128)

# %% Train network

net = Net()

# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()

# create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)

# === Run ===
train_performance = []
validation_performance = []
for i, (aa, adj, targets) in enumerate(tqdm(training_data)):

    aa = Variable(torch.Tensor(aa))
    adj_expanded = Variable(torch.FloatTensor(expand_adjacency(adj).astype(np.float32)))
    targets = Variable(torch.Tensor(targets.astype(np.float32)), requires_grad=False)

    if not targets.data.numpy().any().any():
        print(f"Skipping {i}...")
        continue

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(aa, adj_expanded)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()  # Does the update

    if i % 10 == 0:
        score = metrics.roc_auc_score(targets.data.numpy().astype(int), output.data.numpy())
        train_performance.append(score)

        test_aa = Variable(torch.Tensor(test_data[0]), requires_grad=False)
        test_adj_expanded = Variable(
            torch.FloatTensor(expand_adjacency(test_data[1]).astype(np.float32)),
            requires_grad=False)
        test_targets = Variable(torch.Tensor(test_data[2].astype(np.float32)), requires_grad=False)

        predictions = net(test_aa, test_adj_expanded)
        score = metrics.roc_auc_score(test_targets.data.numpy(), predictions.data.numpy())
        validation_performance.append(score)
        print(score)

plt.plot(range(0, len(train_performance) * 10, 10), train_performance, label='train')
plt.plot(range(0, len(validation_performance) * 10, 10), validation_performance, label='test')
plt.legend()

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
