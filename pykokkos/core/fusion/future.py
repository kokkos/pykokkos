from pykokkos.runtime import runtime_singleton


class Future:
    """
    Delayed reductions and scans return a Future
    """

    def __init__(self) -> None:
        self.value = None

    def assign_value(self, value) -> None:
        self.value = value

    def __add__(self, other):
        self.flush_trace()
        return self.value + other

    def __sub__(self, other):
        self.flush_trace()
        return self.value - other

    def __mul__(self, other):
        self.flush_trace()
        return self.value * other

    def __truediv__(self, other):
        self.flush_trace()
        return self.value / other

    def __floordiv__(self, other):
        self.flush_trace()
        return self.value // other

    def __str__(self):
        self.flush_trace()
        return str(self.value)

    def __repr__(self) -> str:
        return str(f"Future(value={self.value})")

    def flush_trace(self) -> None:
        runtime_singleton.runtime.flush_data(self)
        assert self.value is not None
