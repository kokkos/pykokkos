from enum import Enum

from pykokkos.bindings import kokkos
import numpy as np


class DataType(Enum):
    int8 = kokkos.int8
    int16 = kokkos.int16
    int32 = kokkos.int32
    int64 = kokkos.int64
    uint8 = kokkos.uint8
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


class uint8(DataTypeClass):
    value = kokkos.uint8
    np_equiv = np.uint8

class int8(DataTypeClass):
    value = kokkos.int8
    np_equiv = np.int8

class int16(DataTypeClass):
    value = kokkos.int16
    np_equiv = np.int16

class int32(DataTypeClass):
    value = kokkos.int32
    np_equiv = np.int32

class int64(DataTypeClass):
    value = kokkos.int64
    np_equiv = np.int64

class uint16(DataTypeClass):
    value = kokkos.uint16
    np_equiv = np.uint16


class uint32(DataTypeClass):
    value = kokkos.uint32
    np_equiv = np.uint32


class uint64(DataTypeClass):
    value = kokkos.uint64
    np_equiv = np.uint64


class float(DataTypeClass):
    value = kokkos.float
    np_equiv = np.float32

class double(DataTypeClass):
    value = kokkos.double
    np_equiv = np.float64


class real(DataTypeClass):
    value = None
    np_equiv = None

class float32(DataTypeClass):
    value = kokkos.float
    np_equiv = np.float32

class float64(DataTypeClass):
    value = kokkos.double
    np_equiv = np.float64

class bool(DataTypeClass):
    value = kokkos.uint8
    np_equiv = np.bool_
