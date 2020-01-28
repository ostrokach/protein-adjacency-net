import tempfile

import mdtraj
import numpy as np
import pandas as pd
import pyarrow as pa
from kmtools import structure_tools
from scipy.spatial.transform import Rotation as R


def construct_residue_df(traj):
    residue_features = {}
    residue_features["residue_idx"] = np.arange(traj.top.n_residues)
    residue_features["dssp"] = mdtraj.compute_dssp(traj)[0].tolist()
    # (
    #     residue_features["sasa"],
    #     residue_features["sasa_relative"],
    # ) = structure_tools.protein_structure_analysis.calculate_sasa(traj).T
    residue_features["phi"] = structure_tools.protein_structure_analysis.calculate_phi(traj)
    residue_features["psi"] = structure_tools.protein_structure_analysis.calculate_psi(traj)
    (
        residue_features["omega_prev"],
        residue_features["omega_next"],
    ) = structure_tools.protein_structure_analysis.calculate_omega(traj).T
    residue_features[
        "ca_angles"
    ] = structure_tools.protein_structure_analysis.calculate_backbone_angles(traj)
    (
        residue_features["ca_dihedral_prev"],
        residue_features["ca_dihedral_next"],
    ) = structure_tools.protein_structure_analysis.calculate_backbone_dihedrals(traj).T
    return pd.DataFrame(residue_features)


def validate_residue_df(residue_df):
    return


def mdtraj_protonate(traj):
    with tempfile.NamedTemporaryFile(suffix=".pdb") as input_file, tempfile.NamedTemporaryFile(
        suffix=".pdb"
    ) as output_file:
        structure_tools.mdtraj_to_pdb(traj, input_file.name)
        structure_tools.fixes.protonate(input_file.name, output_file.name, method="reduce")
        traj_protonated = mdtraj.load(output_file.name)
    return traj_protonated


def complete_hydrogen_bonds(hydrogen_bonds_df):
    hydrogen_bonds_completed_df = (
        pd.concat(
            [
                hydrogen_bonds_df,
                hydrogen_bonds_df.rename(
                    columns={"residue_idx_1": "residue_idx_2", "residue_idx_2": "residue_idx_1"}
                ),
            ],
            sort=False,
        )
        .sort_values(["residue_idx_1", "residue_idx_2"])
        .drop_duplicates(subset=["residue_idx_1", "residue_idx_2"])
    )
    return hydrogen_bonds_completed_df


def construct_residue_pairs_df(traj):
    structure_df = structure_tools.mdtraj_to_dataframe(traj)

    distances_all_df = structure_tools.get_distances(structure_df, 12.0, groupby="residue")
    distances_all_df["distance"] = distances_all_df["distance"] * 10
    distances_all_df = structure_tools.complete_distances(distances_all_df)

    distances_backbone_df = structure_tools.get_distances(
        structure_df, 12.0, groupby="residue-backbone"
    )
    distances_backbone_df["distance"] = distances_backbone_df["distance"] * 10
    distances_backbone_df = structure_tools.complete_distances(distances_backbone_df)

    distances_ca_df = structure_tools.get_distances(structure_df, 12.0, groupby="residue-ca")
    distances_ca_df["distance"] = distances_ca_df["distance"] * 10
    distances_ca_df = structure_tools.complete_distances(distances_ca_df)

    hydrogen_bonds_df = structure_tools.protein_structure_analysis.calculate_hydrogen_bonds(
        mdtraj_protonate(traj)
    )
    hydrogen_bonds_df = complete_hydrogen_bonds(hydrogen_bonds_df)

    internal_coords = structure_tools.protein_structure_analysis.get_internal_coords(structure_df)

    translations = structure_tools.protein_structure_analysis.get_translations(structure_df)
    translations_internal = structure_tools.protein_structure_analysis.map_translations_to_internal_coords(
        translations, internal_coords
    )

    rotations = structure_tools.protein_structure_analysis.get_rotations(internal_coords)

    # Bring everything together
    residue_pairs_df = (
        distances_all_df.merge(
            distances_backbone_df.rename(columns={"distance": "distance_backbone"}),
            on=["residue_idx_1", "residue_idx_2"],
            how="outer",
            validate="1:1",
        )
        .merge(
            distances_ca_df.rename(columns={"distance": "distance_ca"}),
            on=["residue_idx_1", "residue_idx_2"],
            how="outer",
            validate="1:1",
        )
        .merge(
            hydrogen_bonds_df.assign(hbond=True),
            on=["residue_idx_1", "residue_idx_2"],
            how="outer",
            validate="1:1",
        )
    )
    residue_pairs_df["hbond"] = residue_pairs_df["hbond"].fillna(False)
    assert len(residue_pairs_df) == len(distances_all_df)

    residue_pairs_df["translation_1"] = residue_pairs_df[["residue_idx_1", "residue_idx_2"]].apply(
        lambda s: translations_internal[s.residue_idx_1, s.residue_idx_2].tolist(), axis=1
    )
    residue_pairs_df["translation_2"] = residue_pairs_df[["residue_idx_1", "residue_idx_2"]].apply(
        lambda s: translations_internal[s.residue_idx_2, s.residue_idx_1].tolist(), axis=1
    )

    residue_pairs_df["rotation_1"] = residue_pairs_df[["residue_idx_1", "residue_idx_2"]].apply(
        lambda s: R.from_dcm(rotations[s.residue_idx_1, s.residue_idx_2]).as_quat().tolist(), axis=1
    )
    residue_pairs_df["rotation_2"] = residue_pairs_df[["residue_idx_1", "residue_idx_2"]].apply(
        lambda s: R.from_dcm(rotations[s.residue_idx_2, s.residue_idx_1]).as_quat().tolist(), axis=1
    )

    return residue_pairs_df


def validate_residue_pairs_df(residue_pairs_df):
    masks = [
        (
            residue_pairs_df["rotation_1"].apply(lambda l: l[i])
            == residue_pairs_df["rotation_2"].apply(lambda l: l[i])
        )
        for i in range(4)
    ]
    assert (masks[0] == masks[1]).all()
    assert (masks[0] == masks[2]).all()
    assert (masks[0] | masks[3]).all()


def residue_df_to_row(residue_df):
    row_data = {}
    for c in residue_df.columns:
        values = residue_df[c].values
        if isinstance(values.dtype, pd.Int64Dtype):
            ar = values._data.astype(object)
            ar[values._mask] = None
            row_data[c] = [pa.array(ar.tolist())]
        else:
            row_data[c] = [pa.array(values.tolist())]
    return row_data


def residue_pairs_df_to_row(residue_pairs_df):
    row_data = {}
    for c in residue_pairs_df.columns:
        values = residue_pairs_df[c].values
        if isinstance(values[0], list):
            row_data[c] = [pa.array([l for lst in values for l in lst])]
        elif isinstance(values.dtype, pd.Int64Dtype):
            ar = values._data.astype(object)
            ar[values._mask] = None
            row_data[c] = [pa.array(ar.tolist())]
        else:
            row_data[c] = [pa.array(values.tolist())]
    return row_data
