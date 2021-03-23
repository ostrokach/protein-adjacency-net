from kmbio.PDB import Structure
from MDAnalysis.lib.distances import self_capped_distance


def get_distances(structure: Structure):
    df = structure.to_dataframe()
    pairs, distances = self_capped_distance(
        df[["atom_x", "atom_y", "atom_z"]].values, max_cutoff=5, min_cutoff=1
    )
    pairs.sort(axis=1)
    assert pairs.max() < len(df)
    return df, pairs, distances
