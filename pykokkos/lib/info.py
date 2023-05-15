from dataclasses import dataclass

from pykokkos.bindings import kokkos

# the integer and float type information functions appear
# to be required by the array API standard
# i.e.,
# https://data-apis.org/array-api/2021.12/API_specification/data_type_functions.html#objects-in-api


@dataclass
class info_type_attrs:
    """
    Store machine limits for numeric data types.
    """
    bits: int
    max: int
    min: int

def iinfo(type_or_arr):
    # TODO: more correct implementation
    # this is really just an initial hack
    # so we can run the array API tests,
    # and effectively just copies return
    # values from the NumPy equivalent
    if "uint32" in str(type_or_arr):
        return info_type_attrs(bits=32,
                               min=0,
                               max=4294967295)
    if "int32" in str(type_or_arr):
        return info_type_attrs(bits=32,
                               max=2147483647,
                               min=-2147483648)
    elif "uint16" in str(type_or_arr):
        # iinfo(min=0, max=65535, dtype=uint16)
        return info_type_attrs(bits=16,
                               min=0,
                               max=65535)
    elif "int16" in str(type_or_arr):
        # iinfo(min=-32768, max=32767, dtype=int16)
        return info_type_attrs(bits=16,
                               min=-32768,
                               max=32767)
    elif "uint64" in str(type_or_arr):
        # iinfo(min=0, max=18446744073709551615, dtype=uint64)
        return info_type_attrs(bits=64,
                               min=0,
                               max=18446744073709551615)
    elif "int64" in str(type_or_arr):
        return info_type_attrs(bits=64,
                               min=-9223372036854775808,
                               max=9223372036854775807)
    elif "uint8" in str(type_or_arr):
        return info_type_attrs(bits=8,
                               min=0,
                               max=255)
    elif "int8" in str(type_or_arr):
        return info_type_attrs(bits=8,
                               min=-128,
                               max=127)


def finfo(type_or_arr):
    # TODO: more correct implementation
    # this is really just an initial hack
    # so we can run the array API tests,
    # and effectively just copies return
    # values from the NumPy equivalent
    if "float" in str(type_or_arr) and not "float64" in str(type_or_arr):
        return info_type_attrs(bits=32,
                               min=-3.4028235e+38,
                               max=3.4028235e+38,)
    elif "double" in str(type_or_arr) or "float64" in str(type_or_arr):
        return info_type_attrs(bits=64,
                               min=-1.7976931348623157e+308,
                               max=1.7976931348623157e+308)

