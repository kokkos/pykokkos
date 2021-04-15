import time


class Timer:
    def __init__(self):
        self.start_time: float = time.perf_counter()

    def seconds(self) -> float:
        current_time: float = time.perf_counter()
        return current_time - self.start_time

    def reset(self) -> None:
        self.start_time = time.perf_counter()
