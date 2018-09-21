class StatsBase:
    step: int
    extended: bool = False

    def __init__(self, step: int) -> None:
        self.step = step
