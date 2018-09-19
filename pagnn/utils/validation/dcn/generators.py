from pathlib import Path

from pagnn import dataset
from pagnn.training.common import get_rowgen_mut
from pagnn.types import DataGen, DataSet


def get_mutation_datagen(mutation_class: str, data_path: Path) -> DataGen:

    mutation_datarows = get_rowgen_mut(mutation_class, data_path)
    mutation_datasets = (dataset.row_to_dataset(row, target=1) for row in mutation_datarows)

    mutation_dsc = []
    for pos_ds in mutation_datasets:
        neg_seq = bytearray(pos_ds.seq)
        mutation = pos_ds.meta["mutation"]  # type: ignore
        mutation_idx = int(mutation[1:-1]) - 1
        assert neg_seq[mutation_idx] == ord(mutation[0]), (chr(neg_seq[mutation_idx]), mutation[0])
        neg_seq[mutation_idx] = ord(mutation[-1])
        neg_ds = DataSet(neg_seq, pos_ds.adj, pos_ds.meta["score"])  # type: ignore
        mutation_dsc.append(([pos_ds], [neg_ds]))

    def datagen():
        for dvc in mutation_dsc:
            yield dvc

    return datagen
