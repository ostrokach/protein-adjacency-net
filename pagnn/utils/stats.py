from typing import Any, Dict


class StatsBase:
    step: int
    extended: bool = False
    scores: Dict[str, Any]

    def __init__(self, step: int) -> None:
        self.step = step
