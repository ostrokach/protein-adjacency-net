
import pytest

from pagnn.utils.converters import str_to_seconds


@pytest.mark.parametrize("time_str, time_seconds", [("1d,2h,21m,1s", 94861), ("1-02:21:01", 94861)])
def test_str_to_seconds(time_str: str, time_seconds: int):
    assert str_to_seconds(time_str) == time_seconds
