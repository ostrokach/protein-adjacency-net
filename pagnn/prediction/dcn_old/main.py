import logging
import sys
from typing import Callable, Iterator, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm

import pagnn
from pagnn.dataset import row_to_dataset
from pagnn.datavardcn import dataset_to_datavar
from pagnn.types import DataRow, DataSet, DataVarCollection
from pagnn.utils import to_numpy

from .args import Args

logger = logging.getLogger(__name__)


def make_predictions(args: Args, datagen: Callable[[], Iterator[DataSet]]) -> np.ndarray:
    Net = getattr(pagnn.models.dcn, args.network_info["network_name"])
    net = Net(**args.network_info["network_settings"])
    net.load_state_dict(torch.load(args.network_state.as_posix()))

    outputs_list: List[np.ndarray] = []
    for dataset in tqdm.tqdm(datagen()):
        datavar = dataset_to_datavar(dataset)
        datavarcol: DataVarCollection = ([datavar], [])
        outputs = net(datavarcol)
        outputs_list.append(to_numpy(outputs))
    outputs = np.vstack(outputs_list).squeeze()
    return outputs


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    args = Args.from_cli()

    pagnn.settings.CUDA = False

    def datagen():
        df = pq.read_table(args.input_file, columns=DataRow._fields).to_pandas()
        for row in df.itertuples():
            dataset = row_to_dataset(row, 0)
            yield dataset

    outputs = make_predictions(args, datagen)
    outputs_df = pd.DataFrame({"predictions": outputs}, index=range(len(outputs)))

    table = pa.Table.from_pandas(outputs_df, preserve_index=False)
    pq.write_table(table, args.output_file, version="2.0", flavor="spark")


if __name__ == "__main__":
    sys.exit(main())
