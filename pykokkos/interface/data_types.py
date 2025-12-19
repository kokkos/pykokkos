from enum import Enum
from builtins import float as builtin_float

from pykokkos.bindings import kokkos
import pykokkos.kokkos_manager as km

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
    complex64 = kokkos.complex_float32_dtype
    complex128 = kokkos.complex_float64_dtype


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


class complex(DataTypeClass):
    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot add '{}' and '{}'".format(type(other), type(self)))

        if isinstance(self, complex64):
            return complex64(self.kokkos_complex + other.kokkos_complex)
        elif isinstance(self, complex128):
            return complex128(self.kokkos_complex + other.kokkos_complex)

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot add '{}' and '{}'".format(type(other), type(self)))

        self.kokkos_complex += other.kokkos_complex
        return self

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot subtract '{}' and '{}'".format(type(other), type(self)))

        if isinstance(self, complex64):
            return complex64(self.kokkos_complex - other.kokkos_complex)
        elif isinstance(self, complex128):
            return complex128(self.kokkos_complex - other.kokkos_complex)

    def __isub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot subtract '{}' and '{}'".format(type(other), type(self)))

        self.kokkos_complex -= other.kokkos_complex
        return self

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot multiply '{}' and '{}'".format(type(other), type(self)))

        if isinstance(self, complex64):
            return complex64(self.kokkos_complex * other.kokkos_complex)
        elif isinstance(self, complex128):
            return complex128(self.kokkos_complex * other.kokkos_complex)

    def __imul__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot multiply '{}' and '{}'".format(type(other), type(self)))

        self.kokkos_complex *= other.kokkos_complex
        return self

    def __truediv__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot divide '{}' and '{}'".format(type(other), type(self)))

        if isinstance(self, complex64):
            return complex64(self.kokkos_complex / other.kokkos_complex)
        elif isinstance(self, complex128):
            return complex128(self.kokkos_complex / other.kokkos_complex)

    def __itruediv__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("cannot divide '{}' and '{}'".format(type(other), type(self)))

        self.kokkos_complex /= other.kokkos_complex
        return self

    def __repr__(self):
        return f"({self.kokkos_complex.real_const()}, {self.kokkos_complex.imag_const()})"

    @property
    def real(self):
        return self.kokkos_complex.real_const()

    @property
    def imag(self):
        return self.kokkos_complex.imag_const()

class complex64(complex):
    value = kokkos.complex_float32_dtype
    np_equiv = np.complex64 # 32 bits from real + 32 from imaginary

    def __init__(self, real: "builtin_float | kokkos.complex_float32", imaginary: builtin_float = 0.0):
        if isinstance(real, kokkos.complex_float32):
            self.kokkos_complex = real
        else:
            self.kokkos_complex = km.get_kokkos_module(is_cpu=True).complex_float32(real, imaginary)


class complex128(complex):
    value = kokkos.complex_float64_dtype
    np_equiv = np.complex128 # 64 bits from real + 64 from imaginary

    def __init__(self, real: "builtin_float | kokkos.complex_float64", imaginary: builtin_float = 0.0):
        if isinstance(real, kokkos.complex_float64):
            self.kokkos_complex = real
        else:
            self.kokkos_complex = km.get_kokkos_module(is_cpu=True).complex_float64(real, imaginary)
