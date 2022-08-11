from enum import Enum

from pykokkos.bindings import kokkos
import numpy as np


class DataType(Enum):
    int16 = kokkos.int16
    int32 = kokkos.int32
    int64 = kokkos.int64
    uint16 = kokkos.uint16
    uint32 = kokkos.uint32
    uint64 = kokkos.uint64
    float = kokkos.float
    double = kokkos.double
    # https://data-apis.org/array-api/2021.12/API_specification/data_types.html
    # A conforming implementation of the array API standard
    # must provide and support the dtypes listed above; for
    # now, we will use aliases and possibly even incorrect/empty
    # implementations so that we can start testing with the array
    # API standard suite, otherwise we won't even be able to import
    # the tests let alone run them
    float32 = kokkos.float
    float64 = kokkos.double
    real = None
    bool = np.bool_


class DataTypeClass:
    pass


class int16(DataTypeClass):
    value = kokkos.int16

class int32(DataTypeClass):
    value = kokkos.int32

class int64(DataTypeClass):
    value = kokkos.int64

class uint16(DataTypeClass):
    value = kokkos.uint16


class uint32(DataTypeClass):
    value = kokkos.int32


class uint64(DataTypeClass):
    value = kokkos.int64


class float(DataTypeClass):
    value = kokkos.float

class double(DataTypeClass):
    value = kokkos.double


class real(DataTypeClass):
    value = None

class float32(DataTypeClass):
    value = kokkos.float

class float64(DataTypeClass):
    value = kokkos.double

class bool(DataTypeClass):
    value = kokkos.int16
