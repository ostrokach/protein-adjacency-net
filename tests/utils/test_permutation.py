import kmbio.PDB
import numpy as np
import pytest
import torch
from kmtools import structure_tools
from scipy import sparse

from pagnn.utils import (
    array_to_seq,
    get_distances,
    permute_adjacency,
    permute_adjacency_dense,
    permute_sequence,
    permute_structure,
    seq_to_array,
)


@pytest.mark.parametrize("pdb_id, offset", [("4dkl", 20), ("5w2o", 11), ("6i0t", 33)])
def test_permute_structure(pdb_id, offset):
    # Reference structure (first chain from provided PDB)
    structure = kmbio.PDB.load(f"rcsb://{pdb_id}.cif")
    while len(list(structure.chains)) > 1:
        structure[0]._children.popitem()
    df, pairs, distances = get_distances(structure)
    num_atoms = len(df)
    num_offset_atoms = len(df[df["residue_idx"] < offset])

    # Permutted structure
    structure_p = permute_structure(structure, offset)
    _, pairs_p, distances_p = get_distances(structure_p)

    # === Test that `permute_structure` works ===

    # Permutted adjacency from reference structure
    pairs_p_ref = permute_adjacency_dense(pairs, num_atoms, num_offset_atoms)
    pairs_p_ref.sort(axis=1)
    pairs_distances_p_ref = np.c_[pairs_p_ref, distances]
    pairs_distances_p_ref.sort(axis=0)

    # Permutted adjacency from permutted structure
    pairs_distances_p = np.c_[pairs_p, distances_p]
    pairs_distances_p.sort(axis=0)

    assert (pairs_distances_p_ref == pairs_distances_p).all()

    # === Test that `permute_sequence` works ===

    # Test permute_sequence
    seq = structure_tools.get_chain_sequence(next(structure.chains))
    seq_array = seq_to_array(seq.encode("ascii"))
    seq_array_p = torch.sparse_coo_tensor(
        *permute_sequence(seq_array._indices(), seq_array._values(), offset), size=seq_array.size()
    )
    seq_permutted = array_to_seq(seq_array_p.to_dense())
    seq_permutted_ = structure_tools.get_chain_sequence(next(structure_p.chains))
    assert seq_permutted == seq_permutted_

    # === Test that `permute_adjacency` works ===

    # Permutted adjacency from permutted structure
    pairs_p = np.r_[pairs_p, pairs_p[:, ::-1]]
    adj_permutted_ = sparse.coo_matrix((np.ones(pairs_p.shape[0]), (pairs_p[:, 0], pairs_p[:, 1])))
    assert np.allclose(adj_permutted_.todense(), adj_permutted_.todense().T)
    assert adj_permutted_.max() == 1

    # Permutted adjacency from reference structure
    pairs_sym = np.r_[pairs, pairs[:, ::-1]]
    adj = sparse.coo_matrix((np.ones(pairs_sym.shape[0]), (pairs_sym[:, 0], pairs_sym[:, 1])))
    adj_permutted = permute_adjacency(adj, num_offset_atoms)
    assert np.allclose(adj_permutted.todense(), adj_permutted.todense().T)
    assert adj_permutted.max() == 1
    assert (adj_permutted_.todense() == adj_permutted.todense()).all()

    # Permutted adjacency from reference structure (dense)
    pairs_sym_p_ref = permute_adjacency_dense(pairs_sym, num_atoms, num_offset_atoms)
    adj_permutted_2 = sparse.coo_matrix(
        (np.ones(pairs_sym_p_ref.shape[0]), (pairs_sym_p_ref[:, 0], pairs_sym_p_ref[:, 1]))
    )
    assert np.allclose(adj_permutted_2.todense(), adj_permutted_2.todense().T)
    assert adj_permutted_2.max() == 1
    assert (adj_permutted_.todense() == adj_permutted_2.todense()).all()


@pytest.mark.parametrize("offset", list(range(3, 8)))
def test_permute_sequence(offset):
    row = [0] * 3 + [3] * 7
    n_aa = len(row)

    seq = torch.sparse_coo_tensor(
        torch.stack([torch.tensor(row, dtype=torch.long), torch.arange(n_aa, dtype=torch.long)]),
        torch.ones(n_aa),
        size=(4, 10),
    )

    seq_permutted = torch.sparse_coo_tensor(
        torch.stack(
            [
                torch.tensor(row[offset:] + row[:offset], dtype=torch.long),
                torch.arange(n_aa, dtype=torch.long),
            ]
        ),
        torch.ones(n_aa),
        size=(4, n_aa),
    )

    seq_permutted_ = torch.sparse_coo_tensor(
        *permute_sequence(seq._indices(), seq._values(), offset)
    )
    assert (seq_permutted.to_dense() == seq_permutted_.to_dense()).all()


@pytest.mark.parametrize("offset", list(range(3, 8)))
def test_permute_adjacency(offset):
    a = np.triu(np.arange(1, 11))
    adj = sparse.coo_matrix(a, dtype=np.float)

    a_ul = a[offset:, offset:]
    a_lr = a[:offset, :offset]
    a_ur = a[offset:, :offset]
    a_ll = a[:offset, offset:]
    a_u = np.hstack([a_ul, a_ur])
    a_l = np.hstack([a_ll, a_lr])
    a_ref = np.vstack([a_u, a_l])
    adj_permutted = sparse.coo_matrix(a_ref, dtype=np.float)

    adj_permutted_ = permute_adjacency(adj, offset)

    assert (adj_permutted.todense() == adj_permutted_.todense()).all()
