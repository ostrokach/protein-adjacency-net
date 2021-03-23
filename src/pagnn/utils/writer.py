import pandas as pd
import sqlalchemy as sa


class Writer:
    def __init__(self, engine: sa.engine.Engine) -> None:
        self._engine = engine

    def get_last_step(self) -> int:
        if self._engine.has_table("stats_extended"):
            sql_query = "select max(step) step from stats_extended"
            step = pd.read_sql_query(sql_query, self._engine).at[0, "step"]
        else:
            step = 0
        return step
