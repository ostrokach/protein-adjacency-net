
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def str_to_path(file: str) -> Path:
    return Path(file).resolve()


def str_to_path_opt(file: Optional[str]) -> Optional[Path]:
    return str_to_path(file) if file is not None else None


def str_to_timedelta(time_val: str) -> timedelta:
    """
    Given a *time_val* (string) such as '5d', returns a timedelta object
    representing the given value (e.g. timedelta(days=5)). Accepts the
    following '<num><char>' formats:

    =========   ======= ===================
    Character   Meaning Example
    =========   ======= ===================
    s           Seconds '60s' -> 60 Seconds
    m           Minutes '5m'  -> 5 Minutes
    h           Hours   '24h' -> 24 Hours
    d           Days    '7d'  -> 7 Days
    =========   ======= ===================

    Source: https://bit.ly/2qRSICD.

    Examples:
        >>> str_to_timedelta('7d')
        datetime.timedelta(7)
        >>> str_to_timedelta('24h')
        datetime.timedelta(1)
        >>> str_to_timedelta('60m')
        datetime.timedelta(0, 3600)
        >>> str_to_timedelta('120s')
        datetime.timedelta(0, 120)
    """
    num = float(time_val[:-1])
    if time_val.endswith("s"):
        return timedelta(seconds=num)
    elif time_val.endswith("m"):
        return timedelta(minutes=num)
    elif time_val.endswith("h"):
        return timedelta(hours=num)
    elif time_val.endswith("d"):
        return timedelta(days=num)
    else:
        raise Exception(f"Unsuported duration: {time_val}")


def str_to_seconds(duration: Union[str, float]) -> float:
    if not isinstance(duration, str):
        return duration
    seconds = 0
    if any(s in duration for s in ["s", "m", "h", "d"]):
        for d in duration.split(","):
            seconds += str_to_timedelta(d).total_seconds()
    else:
        duration = duration.replace("-", ":")
        for time, multiplier in zip(reversed(duration.split(":")), [1, 60, 60 * 60, 24 * 60 * 60]):
            seconds += float(time) * multiplier
    return seconds


def load_yaml(file: Any) -> Dict[str, Any]:
    if not isinstance(file, (str, Path)):
        return file
    with open(file, "rt") as fin:
        data = yaml.load(fin)
    return data


def dump_yaml(data: Dict[str, Any], file: Union[str, Path]) -> Path:
    with open(file, "wt") as fout:
        yaml.dump(data, fout)
    return Path(file)
