import pandas as pd

from kmbio import PDB
from kmtools import sequence_tools, structure_tools


def get_interaction_dataset(structure, bioassembly_id=False, r_cutoff=5):
    """Copied from "datapkg/pdb-analysis/notebooks/extract_pdb_interactions.ipynb"
    """
    interactions = structure_tools.get_interactions(structure, r_cutoff=r_cutoff, interchain=False)
    interactions_core, interactions_interface = structure_tools.process_interactions(interactions)
    interactions_core_aggbychain = structure_tools.process_interactions_core(
        structure, interactions_core
    )
    # Not neccessary to drop duplicates in our cases
    # interactions_core, interactions_core_aggbychain = structure_tools.drop_duplicates_core(
    #     interactions_core, interactions_core_aggbychain
    # )
    return interactions_core, interactions_core_aggbychain


def get_interaction_dataset_wdistances(structure_file, model_id, chain_id, r_cutoff=12):
    structure = PDB.load(structure_file)
    chain = structure[0][chain_id]
    num_residues = len(list(chain.residues))
    dd = structure_tools.DomainDef(model_id, chain_id, 1, num_residues)
    domain = structure_tools.extract_domain(structure, [dd])
    distances_core = structure_tools.get_distances(domain, r_cutoff, 0, groupby="residue")
    assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
    return distances_core


GET_ADJACENCY_WITH_DISTANCES_ROW_ATTRIBUTES = [
    "structure_id",
    "model_id",
    "chain_id",
    "sequence",
    "s_start",
    "s_end",
    "q_start",
    "q_end",
    "sseq",
    "a2b",
    "b2a",
    "residue_idx_1_corrected",
    "residue_idx_2_corrected",
]


def get_adjacency_with_distances(
    row, max_cutoff=12, min_cutoff=None, structure_url_prefix="rcsb://"
):
    missing_attributes = [
        attr for attr in GET_ADJACENCY_WITH_DISTANCES_ROW_ATTRIBUTES if not hasattr(row, attr)
    ]
    assert not missing_attributes, missing_attributes
    # Load structure
    url = f"{structure_url_prefix}{row.structure_id.lower()}.cif.gz"
    structure = PDB.load(url)
    # Template sequence
    chain_sequence = structure_tools.get_chain_sequence(structure[row.model_id][row.chain_id])
    template_sequence = chain_sequence[int(row.s_start - 1) : int(row.s_end)]
    assert len(template_sequence) == len(row.a2b)
    # Target sequence
    target_sequence = row.sequence[int(row.q_start - 1) : int(row.q_end)]
    assert len(target_sequence) == len(row.b2a)
    # Extract domain
    dd = structure_tools.DomainDef(row.model_id, row.chain_id, int(row.s_start), int(row.s_end))
    domain = structure_tools.extract_domain(structure, [dd])
    assert template_sequence == structure_tools.get_chain_sequence(domain)
    assert template_sequence == row.sseq.replace("-", "")
    # Get interactions
    distances_core = structure_tools.get_distances(
        domain, max_cutoff, min_cutoff, groupby="residue"
    )
    assert (distances_core["residue_idx_1"] <= distances_core["residue_idx_2"]).all()
    # Map interactions to target
    for i in [1, 2]:
        distances_core[f"residue_idx_{i}_corrected"] = distances_core[f"residue_idx_{i}"].apply(
            lambda idx: sequence_tools.convert_residue_index_a2b(idx, row.b2a)
        )
    # Remove missing values
    distances_core = distances_core[
        distances_core["residue_idx_1_corrected"].notnull()
        & distances_core["residue_idx_2_corrected"].notnull()
    ]
    # Convert to integers
    distances_core[["residue_idx_1_corrected", "residue_idx_2_corrected"]] = distances_core[
        ["residue_idx_1_corrected", "residue_idx_2_corrected"]
    ].astype(int)
    # Sanity check
    assert (
        distances_core["residue_idx_1_corrected"] < distances_core["residue_idx_2_corrected"]
    ).all()
    # Get the set of interactions
    interactions_1 = set(
        distances_core[(distances_core["distance"] <= 5)][
            ["residue_idx_1_corrected", "residue_idx_2_corrected"]
        ].apply(tuple, axis=1)
    )
    # Get the reference set of interactions
    interactions_2 = {
        (int(r1), int(r2)) if r1 <= r2 else (int(r2), int(r1))
        for r1, r2 in zip(row.residue_idx_1_corrected, row.residue_idx_2_corrected)
        if pd.notnull(r1) and pd.notnull(r2)
    }
    assert not interactions_1 ^ interactions_2
    return (
        distances_core["residue_idx_1_corrected"].values,
        distances_core["residue_idx_2_corrected"].values,
        distances_core["distance"].values,
    )
