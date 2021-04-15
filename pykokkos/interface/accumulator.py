from typing import Generic, TypeVar

class Acc(Generic[TypeVar("T")]):
    def __init__(self, val):
        self.val = val
    
    def __add__(self, other):
        self.val = self.val + other
        return self

    def __radd__(self, other):
        self.val = self.val + other
        return self

    def __sub__(self, other):
        self.val = self.val - other
        return self

    def __rsub__(self, other):
        self.val = other - self.val
        return self

    def __mul__(self, other):
        self.val = self.val * other
        return self

    def __rmul__(self, other):
        self.val = self.val * other
        return self

    def __truediv__(self, other):
        self.val = self.val / other
        return self

    def __rtruediv__(self, other):
        self.val = other / self.val
        return self

    def __floordiv__(self, other):
        self.val = self.val // other
        return self

    def __rfloordiv__(self, other):
        self.val = other // self.val
        return self

    def __mod__(self, other):
        self.val = self.val % other
        return self

    def __rmov__(self, other):
        self.val = other % self.val
        return self

    def __neg__(self):
        self.val = -self.val
        return self

    def __index__(self):
        return int(self.val)

    def not_(self):
        self.val = not self
        return self

    def lt(self, other):
        return self.val < other

    def le(self, other):
        return self.val <= other
    
    def eq(self, other):
        return self.val == other

    def ne(self, other):
        return self.val != other

    def ge(self, other):
        return self.val >= other

    def gt(self, other):
        return self.val > other

