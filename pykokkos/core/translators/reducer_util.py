import re
from typing import Optional

from pykokkos.core import cppast


VALUE_LOC_REDUCERS = {"MaxLoc", "MinLoc"}
MINMAX_REDUCERS = {"MinMax"}
MINMAX_LOC_REDUCERS = {"MinMaxLoc"}
SCALAR_REDUCERS = {"BAnd", "BOr", "LAnd", "LOr", "Max", "Min", "Prod", "Sum"}
NON_SCALAR_REDUCERS = VALUE_LOC_REDUCERS | MINMAX_REDUCERS | MINMAX_LOC_REDUCERS


def is_non_scalar_reducer(reducer: Optional[str]) -> bool:
    return reducer in NON_SCALAR_REDUCERS


def get_cpp_type_name(decltype: cppast.Type) -> str:
    if isinstance(decltype, cppast.PrimitiveType):
        typename = decltype.typename
        return typename.value if hasattr(typename, "value") else typename

    if isinstance(decltype, cppast.ClassType):
        return decltype.typename

    raise TypeError(f"Unsupported reducer value type: {decltype}")


def get_reducer_value_type(reducer: str, value_type: str) -> str:
    if reducer in VALUE_LOC_REDUCERS:
        return f"Kokkos::{reducer}<{value_type},int>::value_type"
    if reducer in MINMAX_REDUCERS:
        return f"Kokkos::{reducer}<{value_type}>::value_type"
    if reducer in MINMAX_LOC_REDUCERS:
        return f"Kokkos::{reducer}<{value_type},int>::value_type"

    raise ValueError(f"Reducer {reducer} does not use a non-scalar value_type")


def get_reducer_scalar_type(acc_type: str) -> str:
    match = re.match(r"Kokkos::(?:MaxLoc|MinLoc|MinMax|MinMaxLoc)<([^,>]+)", acc_type)
    if match is None:
        return acc_type

    return match.group(1)
