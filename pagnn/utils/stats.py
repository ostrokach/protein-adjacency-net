from typing import Any, Callable, Dict, Optional

import pandas as pd
import sqlalchemy as sa


class StatsBase:
    # Class attributes
    columns_to_write: Dict[str, Optional[Callable]] = {}

    # Instance attributes
    step: int
    _engine: sa.engine.Engine
    scores: Dict[str, Any]
    extended: bool

    def __init__(self, step: int, engine: sa.engine.Engine) -> None:
        self.step = step
        self._engine = engine

    def _write_row(self):
        data = {
            k: fn(getattr(self, k)) if fn is not None else getattr(self, k)
            for (k, fn) in self.columns_to_write.items()
        }
        df.to_sql("stats", self._engine, if_exists="append", index=False)
