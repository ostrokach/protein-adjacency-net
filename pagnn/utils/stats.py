from typing import Any, Dict

import sqlalchemy as sa


class StatsBase:
    _engine: sa.engine.Engine
    step: int
    scores: Dict[str, Any]
    metadata: Dict[str, Any]
    extended: bool

    def __init__(self, engine: sa.engine.Engine) -> None:
        self._engine = engine
