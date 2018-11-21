import numpy as np
import torch
from scipy import sparse

from pagnn.dataset import permute_adjacency, permute_sequence


def test_permute_sequence():
    seq = torch.sparse_coo_tensor(
        torch.stack(
            [torch.tensor([0] * 3 + [3] * 7, dtype=torch.long), torch.arange(10, dtype=torch.long)]
        ),
        torch.ones(10),
        size=(4, 10),
    )

    seq_ref = torch.sparse_coo_tensor(
        torch.stack(
            [torch.tensor([3] * 7 + [0] * 3, dtype=torch.long), torch.arange(10, dtype=torch.long)]
        ),
        torch.ones(10),
        size=(4, 10),
    )

    seq_ = permute_sequence(seq, 3)
    assert (seq_.to_dense() == seq_ref.to_dense()).all()


def test_permute_adjacency():
    a = np.triu(np.arange(1, 11))
    adj = sparse.coo_matrix(a, dtype=np.float)

    a_ul = a[3:, 3:]
    a_lr = a[:3, :3]
    a_ur = a[3:, :3]
    a_ll = a[:3, 3:]
    a_u = np.hstack([a_ul, a_ur])
    a_l = np.hstack([a_ll, a_lr])
    a_ref = np.vstack([a_u, a_l])
    adj_ref = sparse.coo_matrix(a_ref, dtype=np.float)

    adj_ = permute_adjacency(adj, 3)
    assert (adj_.todense() == adj_ref.todense()).all()
