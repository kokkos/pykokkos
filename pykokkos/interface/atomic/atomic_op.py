from functools import reduce
from typing import (
    List, Union
)
from ..views import View 


# Atomic operations from:
# https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3Aatomic_op

def atomic_add(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    pass

def atomic_increment(
        view: View,
        indices: List[int]) -> Union[int, float]:
    pass
