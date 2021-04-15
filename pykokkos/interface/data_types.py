from enum import Enum

from pykokkos.bindings import kokkos


class DataType(Enum):
    int16 = kokkos.int16
    int32 = kokkos.int32
    int64 = kokkos.int64
    uint16 = kokkos.uint16
    uint32 = kokkos.uint32
    uint64 = kokkos.uint64
    float = kokkos.float
    double = kokkos.double
    real = None


class DataTypeClass:
    pass


class int16(DataTypeClass):
    pass


class int32(DataTypeClass):
    pass


class int64(DataTypeClass):
    pass


class uint16(DataTypeClass):
    pass


class uint32(DataTypeClass):
    pass


class uint64(DataTypeClass):
    pass


class float(DataTypeClass):
    pass


class double(DataTypeClass):
    pass


class real(DataTypeClass):
    pass