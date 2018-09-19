import logging
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from pagnn.io import gen_datarows_shuffled, iter_datarows, iter_datarows_shuffled
from pagnn.types import DataRow

logger = logging.getLogger(__name__)


def get_rowgen(
    data_path: Path,
    filters: List[Tuple[Callable, str, Any]] = [],
    filterable: bool = False,
    extra_columns: List[str] = [],
    random_state: Optional[np.random.RandomState] = None,
) -> Iterator[DataRow]:
    """Return an iterator of `DataRow` objects from the positive training dataset.

    Thin wrapper around `pagnn.io.iter_datarows_shuffled` providing default parameters.
    """
    parquet_files = sorted(data_path.glob("database_id=*/*.parquet"))
    columns = {
        "qseq": "sequence",
        "residue_idx_1_corrected": "adjacency_idx_1",
        "residue_idx_2_corrected": "adjacency_idx_2",
        **{k: None for k in extra_columns},
    }

    if filterable:
        rowgen_pos = gen_datarows_shuffled(
            parquet_files, columns=columns, filters=filters, random_state=random_state
        )
        next(rowgen_pos)
    else:
        rowgen_pos = iter_datarows_shuffled(
            parquet_files, columns=columns, filters=filters, random_state=random_state
        )

    return rowgen_pos


def get_rowgen_mut(mutation_class: str, data_path: Path):
    assert mutation_class in ["protherm", "humsavar"]

    if mutation_class == "protherm":
        score_column = "ddg_exp"
        parquet_file = data_path.joinpath("protherm_dataset").joinpath(
            "protherm_validaton_dataset.parquet"
        )
    elif mutation_class == "humsavar":
        score_column = "score_exp"
        parquet_file = data_path.joinpath("mutation_datasets").joinpath(
            "humsavar_validaton_dataset.parquet"
        )

    rowgen_mut = iter_datarows(
        parquet_file,
        columns={
            "qseq": "sequence",
            "residue_idx_1_corrected": "adjacency_idx_1",
            "residue_idx_2_corrected": "adjacency_idx_2",
            "qseq_mutation": "mutation",
            score_column: "score",
        },
    )

    return rowgen_mut
