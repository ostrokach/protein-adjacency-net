from typing import Tuple

import numpy as np
import torch
from kmbio.PDB import Chain, Model, Structure
from scipy import sparse


def permute_structure(structure, offset):
    assert len(list(structure.models)) == 1 and len(list(structure.chains)) == 1
    model = next(structure.models)
    chain = next(structure.chains)

    structure_p = Structure(structure.id)
    model_p = Model(model.id)
    structure_p.add(model_p)
    chain_p = Chain(chain.id)
    model_p.add(chain_p)

    for residue in list(chain.residues)[offset:] + list(chain.residues)[:offset]:
        chain_p.add(residue.copy())

    return structure_p


def permute_sequence(
    indices: torch.Tensor, values: torch.Tensor, offset: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: This should take `SparseMat` as input.
    # New indices
    row, col = indices
    assert (col == torch.arange(len(col))).all()
    row_new = torch.cat([row[offset:], row[:offset]])
    new_indices = torch.stack([row_new, col], 0)
    # New values
    # NB: Can't simply say `assert (values == 1).all()` because we assign 0 to unknown residues.
    new_values = torch.cat([values[offset:], values[:offset]])
    # Result
    return new_indices, new_values


def permute_adjacency(adj: sparse.spmatrix, offset: int):
    # TODO: This should take `SparseMat` as input.
    row = adj.row - offset
    row = np.where(row < 0, adj.shape[0] + row, row)
    col = adj.col - offset
    col = np.where(col < 0, adj.shape[1] + col, col)
    adj_permuted = sparse.coo_matrix((adj.data, (row, col)), dtype=adj.dtype, shape=adj.shape)
    return adj_permuted


def permute_adjacency_dense(indices, num_atoms, num_offset_atoms):
    indices = np.where(
        indices >= num_offset_atoms,
        indices - num_offset_atoms,
        indices - num_offset_atoms + num_atoms,
    )
    return indices
