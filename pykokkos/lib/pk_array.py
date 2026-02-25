import numpy as np


class PKArray(np.ndarray):
    def __new__(cls, array):
        return np.asarray(array).view(cls)

    @property
    def dtype(self):
        from pykokkos.interface import data_types as dt

        mapping = {
            np.dtype("bool"): dt.bool,
            np.dtype("int8"): dt.int8,
            np.dtype("int16"): dt.int16,
            np.dtype("int32"): dt.int32,
            np.dtype("int64"): dt.int64,
            np.dtype("uint8"): dt.uint8,
            np.dtype("uint16"): dt.uint16,
            np.dtype("uint32"): dt.uint32,
            np.dtype("uint64"): dt.uint64,
            np.dtype("float32"): dt.float32,
            np.dtype("float64"): dt.float64,
        }
        return mapping.get(super().dtype, super().dtype)
