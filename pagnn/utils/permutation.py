import numpy as np
import torch
from kmbio.PDB import Chain, Model, Structure
from MDAnalysis.lib.distances import self_capped_distance
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


def get_distances(structure):
    df = structure.to_dataframe()
    pairs, distances = self_capped_distance(
        df[["atom_x", "atom_y", "atom_z"]].values, max_cutoff=5, min_cutoff=1
    )
    pairs.sort(axis=1)
    assert pairs.max() < len(df)
    return df, pairs, distances


def permute_sequence(seq: torch.sparse.FloatTensor, offset: int):
    # New indices
    row, col = seq._indices()
    assert (col == torch.arange(len(col))).all()
    row_new = torch.cat([row[offset:], row[:offset]])
    new_indices = torch.stack([row_new, col], 0)
    # New values
    values = seq._values()
    new_values = torch.cat([values[offset:], values[:offset]])
    # Not necessarily true because we assing 0 to unknown residues:
    # if not (values == 1).all()
    # Result
    seq_new = torch.sparse_coo_tensor(
        new_indices, new_values, size=seq.size(), dtype=seq.dtype, device=seq.device
    )
    return seq_new


def permute_adjacency(adj: sparse.spmatrix, offset: int):
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
