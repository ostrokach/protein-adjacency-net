import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import sqlalchemy as sa


def get_default_network_data(network_name: str) -> dict:
    select_best_step_query = dedent(
        """\
        SELECT step
        FROM stats
        WHERE model_location IS NOT NULL
        ORDER BY `validation_gan_permute_80_1000-auc` DESC, `validation_gan_exact_80_1000-auc` DESC
        LIMIT 1
    """
    )

    network_path = Path(os.environ["DATAPKG_OUTPUT_DIR"]).joinpath(
        "adjacency-net-v2", network_name, "train_network"
    )

    # Stats database
    stats_db = network_path.joinpath("stats.db")

    # Select best step
    engine = sa.create_engine(f"sqlite:///{stats_db}")
    best_step_df = pd.read_sql_query(select_best_step_query, engine)
    best_step = int(best_step_df.values)

    # Other options
    network_state = network_path.joinpath("models", f"model_{best_step:012}.state")
    network_file = network_path.joinpath("model.py")
    network_info = {"network_name": f"DCN_{network_name}", "network_settings": {}}

    return dict(
        network_state=network_state,
        network_file=network_file,
        network_info=network_info,
        stats_db=stats_db,
    )


def predict_with_network(df: pd.DataFrame, network_info: dict, network_state: Path) -> np.ndarray:
    df = df.copy()
    if network_info["network_name"] == "Classifier":
        from pagnn.prediction.dcn_old import Args, main

        for adj_col in ["adjacency_idx_1", "adjacency_idx_2"]:
            df[adj_col] = df.apply(
                lambda row: np.r_[row[adj_col], 0 : len(row["sequence"])], axis=1
            ).values
    else:
        from pagnn.prediction.dcn import Args, main
    args = Args(network_info=network_info, network_state=network_state)
    output_df = main(args, df)
    return output_df.values
