from __future__ import annotations
import ctypes
import math
from enum import Enum
import sys
from types import ModuleType
from typing import (
    Dict, Generic, Iterator, List, Optional,
    Tuple, TypeVar, Union
)

import numpy as np

import pykokkos as pk
from pykokkos.bindings import kokkos
import pykokkos.kokkos_manager as km

from .data_types import (
    DataType, DataTypeClass,
    real,
    int8,
    int16, int32, int64,
    uint8,
    uint16, uint32, uint64,
    double, float64,
)
from .layout import get_default_layout, Layout
from .memory_space import get_default_memory_space, MemorySpace
from .hierarchical import TeamMember

class Trait(Enum):
    Atomic = kokkos.Atomic
    TraitDefault = None
    RandomAccess = kokkos.RandomAccess
    Restrict = kokkos.Restrict
    Unmanaged = kokkos.Unmanaged

class ViewTypeInfo:
    """
    Contains type information for a view that is used by a functor
    """

    def __init__(self, *, space: Optional[MemorySpace] = None, layout: Optional[Layout] = None, trait: Optional[Trait] = None):
        """
        ViewTypeInfo constructor

        :param space: the memory space of the view
        :param layout: the layout of the view
        :param trait: the memory trait of the view
        """

        self.space: Optional[MemorySpace] = space
        self.layout: Optional[Layout] = layout
        self.trait: Optional[Trait] = trait


class ViewType:
    """
    Base class of all view types. Implements methods needed for container objects and some Kokkos specific methods.
    """

    data: np.ndarray
    shape: Tuple[int]
    dtype: DataType
    space: MemorySpace
    layout: Layout
    trait: Trait
    size: int

    def rank(self) -> int:
        """
        The number of dimensions

        :returns: an int representing the number of dimensions
        """

        return len(self.shape)

    def extent(self, dimension: int) -> int:
        """
        The length of a specific dimension

        :param dimension: the dimension for which the length is needed
        :returns: an int representing the length of the specified dimension
        """

        if dimension >= self.rank() and not (dimension == 0 and self.shape == ()):
            raise ValueError(
                "\"dimension\" must be less than the view's rank")

        if self.shape == ():
            return 0

        return self.shape[dimension]

    def fill(self, value: Union[int, float]) -> None:
        """
        Sets all elements to a scalar value

        :param value: the scalar value
        """

        self.data.fill(value)

    def __getitem__(self, key: Union[int, TeamMember, slice, Tuple]) -> Union[int, float, Subview]:
        """
        Overloads the indexing operator accessing the View

        :param key: the specified index. Can be an int, slice, or a Tuple of ints and slices
        :returns: a primitive type value if key is an int, a Subview otherwise
        """

        if self.shape == () and key == 0:
            return self.data

        if isinstance(key, int) or isinstance(key, TeamMember):
            return self.data[key]

        length: int = 1 if isinstance(key, slice) else len(key)
        if length != self.rank():
            raise ValueError("Please include slices for all dimensions")

        subview = Subview(self, key)

        return subview

    def __setitem__(self, key: Union[int, TeamMember], value: Union[int, float]) -> None:
        """
        Overloads the indexing operator setting an item in the View.

        :param key: the specified index. Can be an int or TeamMember.
        :param value: the new value at the index.
        """

        self.data[key] = value

    def __bool__(self):
        # TODO: more complete implementation
        if self.shape == (1,) or self.shape == ():
            return bool(self.data)


    def __len__(self) -> int:
        """
        Implements the len() function

        :returns: the length of the first dimension
        """

        # NOTE: careful with 0-D treatments and __bool__
        # related handling; you can have shape () and
        # still be True for example...
        if len(self.shape) == 0:
            if self.data != 0:
                return 1
            else:
                return 0
        return self.shape[0]

    def __iter__(self) -> Iterator:
        """
        Implements iteration for Subview

        :returns: an iterator over the data
        """

        if self.data.ndim > 0:
            return (n for n in self.data)
        else:
            # 0-D case returns empty generator
            return zip()

    def __str__(self) -> str:
        """
        Implements the str() function

        :returns: the string representation of the data
        """

        return str(self.data)

    def __deepcopy__(self, memo):
        """
        Implements the deepcopy() function as a shallow copy
        """

        return self


    def _scalarfunc(self, func):
        # based on approach used in
        # numpy/lib/user_array.py for
        # handling scalar conversions
        if self.ndim == 0 or (self.ndim == 1 and self.size == 1):
            return func(self[0])
        else:
            raise TypeError("only single element arrays can be converted to Python scalars.")


    def __float__(self):
        return self._scalarfunc(float)


    def __int__(self):
        return self._scalarfunc(int)


class View(ViewType):
    def __init__(
        self,
        shape: Union[List[int], Tuple[int]],
        dtype: Union[DataTypeClass, type] = real,
        space: MemorySpace = MemorySpace.MemorySpaceDefault,
        layout: Layout = Layout.LayoutDefault,
        trait: Trait = Trait.TraitDefault,
        array: Optional[np.ndarray] = None
    ):
        """
        View constructor.

        :param shape: the shape of the view as a list or tuple of integers
        :param dtype: the data type of the view, either a pykokkos DataType or "int" or "float".
        :param space: the memory space of the view. Will be set to the execution space of the view by default.
        :param layout: the layout of the view in memory.
        :param trait: the memory trait of the view
        :param array: the numpy array if trait is Unmanaged
        """

        self._init_view(shape, dtype, space, layout, trait, array)

    def resize(self, dimension: int, size: int) -> None:
        """
        Resizes a dimension of the view

        :param dimension: the dimension to be resized
        :param size: the new size
        """

        if dimension >= self.rank() and not (dimension == 0 and self.shape == ()):
            raise ValueError(
                f"Cannot resize dimension {dimension} since rank = {self.rank()}")

        if self.shape != () and self.shape[dimension] == size:
            return

        old_data: np.ndarray = self.data

        shape_list: List[int] = list(self.shape)
        if shape_list == []:
            shape_list.append(size)
        else:
            shape_list[dimension] = size

        self.shape = tuple(shape_list)

        is_cpu: bool = self.space is MemorySpace.HostSpace
        kokkos_lib: ModuleType = km.get_kokkos_module(is_cpu)
        self.array = kokkos_lib.array(
            "", self.shape, None, None, self.dtype.value, self.space.value, self.layout.value, self.trait.value)
        self.data = np.array(self.array, copy=False)

        smaller: np.ndarray = old_data if old_data.size < self.data.size else self.data
        data_slice = tuple([slice(0, i) for i in smaller.shape])
        self.data[data_slice] = old_data[data_slice]

    def set_precision(self, dtype: Union[DataTypeClass, type]) -> None:
        """
        Set the precision of the View, reallocating it

        :param dtype: the data type of the view, either a pykokkos DataType or "int" or "float".
        """

        old_data: np.ndarray = self.data
        self._init_view(self.shape, dtype, self.space, self.layout, self.trait)
        np.copyto(self.data, old_data, casting="unsafe")

    def _init_view(
        self,
        shape: Union[List[int], Tuple[int]],
        dtype: Union[DataTypeClass, type] = real,
        space: MemorySpace = MemorySpace.MemorySpaceDefault,
        layout: Layout = Layout.LayoutDefault,
        trait: Trait = Trait.TraitDefault,
        array: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize the view

        :param shape: the shape of the view as a list or tuple of integers
        :param dtype: the data type of the view, either a pykokkos DataType or "int" or "float".
        :param space: the memory space of the view. Will be set to the execution space of the view by default.
        :param layout: the layout of the view in memory.
        :param trait: the memory trait of the view
        :param array: the numpy array if trait is Unmanaged
        """

        self.shape: Tuple[int] = tuple(shape)
        self.size: int = math.prod(shape)
        self.ndim: int = len(shape)
        self.dtype: Optional[DataType] = self._get_type(dtype)
        if self.dtype is None:
            sys.exit(f"ERROR: Invalid dtype {dtype}")

        if space is MemorySpace.MemorySpaceDefault:
            space = get_default_memory_space(km.get_default_space())

        if layout is Layout.LayoutDefault:
            layout = get_default_layout(space)

        # only allow CudaSpace/HIPSpace view for cupy arrays
        if (space in {MemorySpace.CudaSpace, MemorySpace.HIPSpace}) and trait is not trait.Unmanaged:
            space = MemorySpace.HostSpace

        self.space: MemorySpace = space
        self.layout: Layout = layout
        self.trait: Trait = trait

        is_cpu: bool = self.space is MemorySpace.HostSpace
        kokkos_lib: ModuleType = km.get_kokkos_module(is_cpu)

        if self.dtype == pk.float:
            self.dtype = DataType.float
        elif self.dtype == pk.double:
            self.dtype = DataType.double
        if trait is trait.Unmanaged:
            if array is not None and array.ndim == 0:
                # TODO: we don't really support 0-D under the hood--use
                # NumPy for now...
                self.array = array
            else:
                self.array = kokkos_lib.unmanaged_array(array, dtype=self.dtype.value, space=self.space.value, layout=self.layout.value)
        else:
            if len(self.shape) == 0:
                shape = [1]
            self.array = kokkos_lib.array("", shape, None, None, self.dtype.value, space.value, layout.value, trait.value)
        self.data = np.array(self.array, copy=False)

    def _get_type(self, dtype: Union[DataType, type]) -> Optional[DataType]:
        """
        Get the data type from a DataType or a type that is a subclass of
        DataTypeClass or a primitive type

        :param dtype: the input data type :returns: a DataType Enum
        """

        if isinstance(dtype, DataType):
            if dtype is DataType.real:
                return km.get_default_precision()

            return dtype

        if issubclass(dtype, DataTypeClass):
            if dtype is real:
                return DataType[km.get_default_precision().__name__]

            if dtype == DataType.int64:
                dtype = int64

            return dtype

        if dtype is int:
            return DataType["int32"]
        if dtype is float:
            return DataType["double"]

        return None


    def __eq__(self, other):
        # avoid circular import with scoped import
        from pykokkos.lib.ufuncs import equal
        if isinstance(other, float):
            new_other = pk.View((), dtype=pk.double)
            new_other[:] = other
        elif isinstance(other, bool):
            new_other = pk.View((), dtype=pk.bool)
            new_other[:] = other
        elif isinstance(other, int):
            if self.ndim == 0:
                ret = pk.View((), dtype=pk.bool)
                ret[:] = int(self) == other
                return ret
            if 0 <= other <= 255:
                other_dtype = pk.uint8
            elif 0 <= other <= 65535:
                other_dtype = pk.uint16
            elif 0 <= other <= 4294967295:
                other_dtype = pk.uint32
            elif 0 <= other <= 18446744073709551615:
                other_dtype = pk.uint64
            elif -128 <= other <= 127:
                other_dtype = pk.int8
            elif -32768 <= other <= 32767:
                other_dtype = pk.int16
            elif -2147483648 <= other <= 2147483647:
                other_dtype = pk.int32
            elif -9223372036854775808 <= other <= 9223372036854775807:
                other_dtype = pk.int64
            new_other = pk.View((), dtype=other_dtype)
            new_other[:] = other
        elif isinstance(other, pk.View):
            new_other = other
        else:
            raise ValueError("unexpected types!")
        return equal(self, new_other)



    def __hash__(self):
        try:
            hash_value = hash(self.array)
        except TypeError:
            hash_value = hash(self.array.data.tobytes())
        return hash_value


    def __index__(self) -> int:
        return int(self.data[0])
    
    
    def __array__(self, dtype=None):
        return self.data


    @staticmethod
    def _get_dtype_name(type_name: str) -> str:
        """
        Get the type name of the Kokkos view object as a string

        :param type_name: the string representation of the array type, of the form
                          'pykokkos.bindings.luzhou.libpykokkos.KokkosView_double_LayoutRight_HostSpace_1'
        :returns: the dtype of the Kokkos View
        """

        dtype: str = type_name.split(".")[-1].split("_")[1]

        return dtype

class Subview(ViewType):
    """
    A Subview wraps the "data" member of a View (or Subview) and references a slice of that data.
    Subviews are passed to C++ as unmanaged views.
    This class contains the Python implementation of a subview. The user is not meant to call
    the constructor directly, instead they should slice the original View object.
    """

    def __init__(self, parent_view: Union[Subview, View], data_slice: Union[slice, Tuple]):
        """
        Subview constructor.

        :param parent_view: the View or Subview that is meant to be sliced
        :param data_slice: the slice of the parent_view
        """

        self.parent_view: Union[Subview, View] = parent_view
        self.base_view: View = self._get_base_view(parent_view)

        self.data: np.ndarray = parent_view.data[data_slice]
        self.dtype = parent_view.dtype

        is_cpu: bool = self.parent_view.space is MemorySpace.HostSpace
        kokkos_lib: ModuleType = km.get_kokkos_module(is_cpu)

        if self.data is not None and self.data.ndim == 0:
            # TODO: we don't really support 0-D under the hood--use
            # NumPy for now...
            self.array = self.data
        else:
            self.array = kokkos_lib.array(
                self.data, dtype=parent_view.dtype.value, space=parent_view.space.value,
                layout=parent_view.layout.value, trait=kokkos.Unmanaged)
        self.shape: Tuple[int] = self.data.shape

        if self.data.shape == (0,):
            self.data = np.array([], dtype=self.data.dtype)
            self.shape = ()

        self.parent_slice: List[Union[int, slice]]
        self.parent_slice = self._create_slice(data_slice)
        self.ndim = self.data.ndim
        self.size = self.data.size

    def _create_slice(self, data_slice: Union[slice, Tuple]) -> List[Union[int, slice]]:
        """
        Transforms the slice into a list, removing all None values for start and stop

        :returns: a list of integers and slices representing the full slice
        """

        parent_slice: List[Union[int, slice]] = []

        if isinstance(data_slice, slice):
            data_slice = (data_slice,)

        for i, s in enumerate(data_slice):
            if isinstance(s, slice):
                start: int = 0 if s.start is None else s.start
                stop: int = self.parent_view.extent(
                    i) if s.stop is None else s.stop
                parent_slice.append(slice(start, stop, None))
            elif isinstance(s, int):
                parent_slice.append(s)

        return parent_slice

    def _get_base_view(self, parent_view: Union[Subview, View]) -> View:
        """
        Gets the base view (first ancestor) of the Subview

        :param parent_view: the direct ancestor of the Subview
        :returns: the View object representing the base view
        """

        base_view: View
        if isinstance(parent_view, View):
            base_view = parent_view
        else:
            base_view = parent_view.base_view

        return base_view

    def __eq__(self, other):
        if isinstance(other, View):
            if len(self.data) == 0 and len(other.data) == 0:
                return True
            result_of_eq = self.data == other.data
            return result_of_eq


    def __add__(self, other):
        if isinstance(other, float):
            result = self[0] + other
            return result


    def __mul__(self, other):
        if isinstance(other, float):
            result = self[0] * other
            return result
        elif isinstance(other, Subview):
            if self.size == 1 and other.size == 1:
                result = self[0] * other[0]
                return result


    def __hash__(self):
        hash_value = hash(self.array)
        return hash_value

def from_numpy(array: np.ndarray, space: Optional[MemorySpace] = None, layout: Optional[Layout] = None) -> ViewType:
    """
    Create a PyKokkos View from a numpy array

    :param array: the numpy array
    :param layout: an optional argument for memory space (used by from_cupy)
    :param layout: an optional argument for layout (used by from_cupy)
    :returns: a PyKokkos View wrapping the array
    """

    dtype: DataTypeClass
    np_dtype = array.dtype.type

    if np_dtype is np.int8:
        dtype = int8
    elif np_dtype is np.int16:
        dtype = int16
    elif np_dtype is np.int32:
        dtype = int32
    elif np_dtype is np.int64:
        dtype = int64
    elif np_dtype is np.uint8:
        dtype = uint8
    elif np_dtype is np.uint16:
        dtype = uint16
    elif np_dtype is np.uint32:
        dtype = uint32
    elif np_dtype is np.uint64:
        dtype = uint64
    elif np_dtype is np.float32:
        dtype = DataType.float # PyKokkos float
    elif np_dtype is np.float64:
        dtype = float64
    elif np_dtype is np.bool_:
        dtype = uint8
    else:
        raise RuntimeError(f"ERROR: unsupported numpy datatype {np_dtype}")

    if layout is None and array.ndim > 1:
        if array.flags["F_CONTIGUOUS"]:
            layout = Layout.LayoutLeft
        else:
            layout = Layout.LayoutRight

    if space is None:
        space = MemorySpace.MemorySpaceDefault

    if layout is None:
        layout = Layout.LayoutDefault

    # TODO: pykokkos support for 0-D arrays?
    # temporary/terrible hack here for array API testing..
    if array.ndim == 0:
        ret_list = ()
        if np_dtype == np.bool_:
            if array == 1:
                array = np.array(1, dtype=np.uint8)
            else:
                array = np.array(0, dtype=np.uint8)
        else:
            array = np.array(array, dtype=np_dtype)
    else:
        ret_list = list((array.shape))


    return View(ret_list, dtype, space=space, trait=Trait.Unmanaged, array=array, layout=layout)

def from_cupy(array) -> ViewType:
    """
    Create a PyKokkos View from a cupy array

    :param array: the cupy array
    """

    np_dtype = array.dtype.type

    if np_dtype is np.int8:
        ctype = ctypes.c_int8
    if np_dtype is np.int16:
        ctype = ctypes.c_int16
    elif np_dtype is np.int32:
        ctype = ctypes.c_int32
    elif np_dtype is np.int64:
        ctype = ctypes.c_int64
    elif np_dtype is np.uint8:
        ctype = ctypes.c_uint8
    elif np_dtype is np.uint16:
        ctype = ctypes.c_uint16
    elif np_dtype is np.uint32:
        ctype = ctypes.c_uint32
    elif np_dtype is np.uint64:
        ctype = ctypes.c_uint64
    elif np_dtype is np.float32:
        ctype = ctypes.c_float
    elif np_dtype is np.float64:
        ctype = ctypes.c_double
    else:
        raise RuntimeError(f"ERROR: unsupported numpy datatype {np_dtype}")

    # Inspired by
    # https://stackoverflow.com/questions/23930671/how-to-create-n-dim-numpy-array-from-a-pointer

    ptr = array.data.ptr
    ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    np_array = np.ctypeslib.as_array(ptr, shape=array.shape)

    # need to select the layout here since the np_array flags do not
    # preserve the original flags
    layout: Layout
    if array.flags["F_CONTIGUOUS"]:
        layout = Layout.LayoutLeft
    else:
        layout = Layout.LayoutRight

    memory_space: MemorySpace
    if km.get_gpu_framework() == "Cuda":
        memory_space = MemorySpace.CudaSpace
    elif km.get_gpu_framework() == "HIP":
        memory_space = MemorySpace.HIPSpace

    return from_numpy(np_array, memory_space, layout)


# asarray is required for comformance with the array API:
# https://data-apis.org/array-api/2021.12/API_specification/creation_functions.html#objects-in-api

def asarray(obj, /, *, dtype=None, device=None, copy=None):
    # TODO: proper implementation/design
    # for now, let's cheat and use NumPy asarray() followed
    # by pykokkos from_numpy()
    if obj in {pk.e, pk.pi, pk.inf, pk.nan}:
        if dtype is None:
            dtype = pk.float64
        view = pk.View([1], dtype=dtype)
        view[:] = obj
        return view
    if dtype is not None:
        arr = np.asarray(obj, dtype=dtype.np_equiv)
    else:
        arr = np.asarray(obj)
    ret = from_numpy(arr)
    return ret


def _get_largest_type(type_list: List[DataTypeClass], type_info: Callable) -> DataTypeClass:
    largest_type = type_list[0]
    for dtype in type_list[1:]:
        if type_info(dtype).max > type_info(largest_type).max:
            largest_type = dtype
    return largest_type


def result_type(*arrays_and_dtypes: DataTypeClass) -> DataTypeClass:
    """
    Returns the dtype that results from applying the type promotion rules to the arguments.

    :param arrays_and_dtypes: an arbitrary number of input arrays and/or dtypes
    :returns: the dtype resulting from an operation involving the input arrays and dtypes

    """
    # TODO: we'll probably want objects for "categories
    # of types" to simplify the logic here eventually..
    types_seen = []
    uint_types_seen = []
    int_types_seen = []
    float_types_seen = []
    for element in arrays_and_dtypes:
        if isinstance(element, pk.View):
            raise NotImplementedError("type promotion not yet implemented for Views")
        # dtypes may be added directly
        for known_dtype in DataType.__members__.items():
            if element.value == known_dtype[1].value:
                types_seen.append(element)
                if "uint" in str(element.value):
                    uint_types_seen.append(element)
                elif "int" in str(element.value):
                    int_types_seen.append(element)
                elif "float" in str(element.value) or "double" in str(element.value):
                    float_types_seen.append(element)
                break
    # if there is only one type, we simply
    # return it
    if len(set(types_seen)) == 1:
        return types_seen[0]
    # if we have a mixture of a single "category of types"
    # we simply use the largest one
    if uint_types_seen and (not int_types_seen) and (not float_types_seen):
        return _get_largest_type(type_list=uint_types_seen,
                                 type_info=pk.iinfo)
    if int_types_seen and (not uint_types_seen) and (not float_types_seen):
        return _get_largest_type(type_list=int_types_seen,
                                 type_info=pk.iinfo)
    if float_types_seen and (not uint_types_seen) and (not int_types_seen):
        return _get_largest_type(type_list=float_types_seen,
                                 type_info=pk.finfo)
    raise NotImplementedError("Casting rules not implemented for the input.")



T = TypeVar("T")

class View1D(Generic[T]):
    pass


class View2D(Generic[T]):
    pass


class View3D(Generic[T]):
    pass


class View4D(Generic[T]):
    pass


class View5D(Generic[T]):
    pass


class View6D(Generic[T]):
    pass


class View7D(Generic[T]):
    pass


class View8D(Generic[T]):
    pass


class ScratchView:
    def shmem_size(i: int):
        pass


class ScratchView1D(ScratchView, Generic[T]):
    pass


class ScratchView2D(ScratchView, Generic[T]):
    pass


class ScratchView3D(ScratchView, Generic[T]):
    pass


class ScratchView4D(ScratchView, Generic[T]):
    pass


class ScratchView5D(ScratchView, Generic[T]):
    pass


class ScratchView6D(ScratchView, Generic[T]):
    pass


class ScratchView7D(ScratchView, Generic[T]):
    pass


class ScratchView8D(ScratchView, Generic[T]):
    pass


def astype(view, dtype):
    new_view = pk.View([*view.shape], dtype=dtype)
    new_view[:] = view
    return new_view
