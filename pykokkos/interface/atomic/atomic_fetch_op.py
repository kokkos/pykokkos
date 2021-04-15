from functools import reduce
from typing import (
    List, Union
)
import operator
from ..views import View 


# Atomic operations from:
# https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3Aatomic_fetch_op

def atomic_fetch_add(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] += value
    return inner[indices[-1]] 

def atomic_fetch_and(
        view: View,
        indices: List[int],
        value: int) -> int:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] &= value
    return inner[indices[-1]]

def atomic_fetch_div(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] /= value
    return inner[indices[-1]] 

def atomic_fetch_lshift(
        view: View,
        indices: List[int],
        value: int) -> int:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] <<= value
    return inner[indices[-1]]

def atomic_fetch_max(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] = max(inner[indices[-1]], value)
    return inner[indices[-1]] 

def atomic_fetch_min(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] = min(inner[indices[-1]], value)
    return inner[indices[-1]] 

def atomic_fetch_mod(
        view: View,
        indices: List[int],
        value: int) -> int:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] %= value
    return inner[indices[-1]]

def atomic_fetch_mul(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] *= value
    return inner[indices[-1]] 

def atomic_fetch_or(
        view: View,
        indices: List[int],
        value: int) -> int:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] |= value
    return inner[indices[-1]]

def atomic_fetch_rshift(
        view: View,
        indices: List[int],
        value: int) -> int:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] >>= value
    return inner[indices[-1]]

def atomic_fetch_sub(
        view: View,
        indices: List[int],
        value: Union[int, float]) -> Union[int, float]:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] -= value
    return inner[indices[-1]] 

def atomic_fetch_xor(
        view: View,
        indices: List[int],
        value: int) -> int:
    inner = reduce(operator.getitem, indices[:-1], view)
    inner[indices[-1]] ^= value
    return inner[indices[-1]]

def atomic_compare_exchange(
        view: View,
        indices: List[int],
        comparison_value: int,
        new_value: int) -> int:
    pass