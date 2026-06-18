from dataclasses import dataclass


@dataclass(frozen=True)
class Reducer:
    """
    Marker for a Kokkos built-in reducer.

    PyKokkos consumes these markers at dispatch time and emits the matching
    Kokkos reducer in generated C++.
    """

    name: str
    is_scalar: bool = True


BAnd = Reducer("BAnd")
BOr = Reducer("BOr")
LAnd = Reducer("LAnd")
LOr = Reducer("LOr")
Max = Reducer("Max")
Min = Reducer("Min")
Prod = Reducer("Prod")
Sum = Reducer("Sum")

MaxLoc = Reducer("MaxLoc", is_scalar=False)
MinLoc = Reducer("MinLoc", is_scalar=False)
MinMax = Reducer("MinMax", is_scalar=False)
MinMaxLoc = Reducer("MinMaxLoc", is_scalar=False)
