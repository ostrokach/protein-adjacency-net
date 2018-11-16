from kmtools import structure_tools


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
