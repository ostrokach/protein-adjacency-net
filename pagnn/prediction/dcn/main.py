import logging
from typing import Callable, Iterator, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import pagnn
from pagnn.dataset import dataset_to_gan, row_to_dataset
from pagnn.types import DataRow, DataSet

from .args import Args

logger = logging.getLogger(__name__)


def make_predictions(args: Args, datagen: Callable[[], Iterator[DataSet]]) -> np.ndarray:
    Net = getattr(pagnn.models.dcn, args.network_info["network_name"])
    net = Net(**args.network_info["network_settings"])
    net.load_state_dict(torch.load(args.network_state.as_posix()))

    outputs_list: List[np.ndarray] = []
    for dataset in datagen():
        datavar = net.dataset_to_datavar(dataset)
        outputs = net(datavar.seqs, [datavar.adjs])
        outputs_list.append(outputs.sigmoid().mean().data.numpy())
    outputs = np.vstack(outputs_list).squeeze()
    return outputs


def main(args: Optional[Args] = None) -> pd.DataFrame:

    if args is None:
        args = Args.from_cli()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    pagnn.settings.device = torch.device("cpu")

    def datagen():
        df = pq.read_table(args.input_file, columns=DataRow._fields).to_pandas()
        for row in df.itertuples():
            dataset = dataset_to_gan(row_to_dataset(row, 0))
            yield dataset

    outputs = make_predictions(args, datagen)
    outputs_df = pd.DataFrame({"predictions": outputs}, index=range(len(outputs)))

    if args.output_file is not None:
        table = pa.Table.from_pandas(outputs_df, preserve_index=False)
        pq.write_table(table, args.output_file, version="2.0", flavor="spark")

    return outputs_df
