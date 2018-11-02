import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_default_network_data(network_name: str) -> dict:
    network_path = (
        Path(os.environ['DATAPKG_OUTPUT_DIR'])
        .joinpath("adjacency-net-v2", "master", "train_network", network_name)
    )
    return dict(
        network_state = sorted(network_path.joinpath("models").glob("*.state"))[-1],
        network_file = network_path.joinpath("model.py"),
        network_info={
            "network_name": f"DCN_{network_name}",
            "network_settings": {},
        },
        stats_db=network_path.joinpath("stats.db"),
    )


def predict_with_network(df: pd.DataFrame, network_info: dict, network_state: Path) -> np.ndarray:
    df = df.copy()
    if network_info['network_name'] == "Classifier":
        from pagnn.prediction.dcn_old import Args, main
        for adj_col in ['adjacency_idx_1', 'adjacency_idx_2']:
            df[adj_col] = df.apply(lambda row: np.r_[row[adj_col], 0:len(row['sequence'])], axis=1).values
    else:
        from pagnn.prediction.dcn import Args, main
    args = Args(network_info=network_info, network_state=network_state)
    output_df = main(args, df)
    return output_df.values
