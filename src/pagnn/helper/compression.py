import io
from typing import Any, List, Mapping, Union

import brotli
import pyarrow as pa


def downcast_and_compress(
    data: Mapping[str, List[Any]]
) -> Mapping[str, List[Union[int, str, float, bytes]]]:
    data_new = {}
    for key, value in data.items():
        assert len(value) == 1
        value_new: Union[int, str, float, bytes]
        if isinstance(value[0], (int, str, float)):
            value_new = value[0]
        else:
            if isinstance(value[0], pa.Array):
                array = value[0]
            elif isinstance(value[0], (list, tuple)):
                array = pa.array(value[0])
            else:
                raise Exception(f"Unsupported data type: '{type(value[0])}'.")
            array = _downcast_array(array)
            value_new = compress(array)
        data_new[key] = [value_new]
    return data_new


def _downcast_array(array: pa.Array) -> pa.Array:
    if array.type in (pa.float64(),):
        array = array.cast(pa.float32())
    elif array.type in (pa.int64(),):
        array = array.cast(pa.uint16())
    elif array.type in (pa.string(), pa.bool_()):
        pass
    else:
        raise Exception(f"Did not downcast array with type '{array.type}'.")
    return array


def compress(array: pa.Array) -> bytes:
    rb = pa.RecordBatch.from_arrays([array], ["array"])
    buf = io.BytesIO()
    writer = pa.RecordBatchFileWriter(buf, rb.schema)
    writer.write_batch(rb)
    writer.close()
    buf.seek(0)
    return brotli.compress(buf.read())


def decompress(pybytes: bytes) -> pa.Array:
    buf = io.BytesIO()
    buf.write(brotli.decompress(pybytes))
    buf.seek(0)
    reader = pa.RecordBatchFileReader(buf)
    rb = reader.get_batch(0)
    return rb.column(0)
