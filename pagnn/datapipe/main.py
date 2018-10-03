from pathlib import Path


def datapipe(index_file: Path, data_file: Path):
    """Generate a positive and a negative dataset batch."""
    ds_list = []
    while len(ds_list) < args.batch_size:
        if buffer is None:
            pos_row = next(positive_rowgen)
            pos_ds = dataset_to_gan(row_to_dataset(pos_row, 1))
            if not dataset_matches_spec(pos_ds, args):
                continue
            ds = negative_ds_gen.send(pos_ds)
        else:
            ds = next(buffer)
        ds_list.append(ds)
    return ds_list


def serialize_sparse_tensor(tensor) -> Dict[str, np.ndarray]:
    ...


def deserialize_sparse_tensor(Dict[str, np.ndarray]) -> tensor:
