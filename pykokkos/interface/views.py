from __future__ import annotations
import ctypes
from enum import Enum
import sys
from typing import (
    Dict, Generic, Iterator, List, Optional,
    Tuple, TypeVar, Union
)

import numpy as np

from pykokkos.bindings import kokkos
import pykokkos.kokkos_manager as km

from .data_types import (
    DataType, DataTypeClass,
    real,
    int16, int32, int64,
    uint16, uint32, uint64,
    double
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
    shape: List[int]
    dtype: DataType
    space: MemorySpace
    layout: Layout
    trait: Trait

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

        if dimension >= self.rank():
            raise ValueError(
                "\"dimension\" must be less than the view's rank")

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

    def __len__(self) -> int:
        """
        Implements the len() function

        :returns: the length of the first dimension
        """

        return self.shape[0]

    def __iter__(self) -> Iterator:
        """
        Implements iteration for Subview

        :returns: an iterator over the data
        """

        return (n for n in self.data)

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


class View(ViewType):
    def __init__(
        self,
        shape: List[int],
        dtype: Union[DataTypeClass, type] = real,
        space: MemorySpace = MemorySpace.MemorySpaceDefault,
        layout: Layout = Layout.LayoutDefault,
        trait: Trait = Trait.TraitDefault,
        array: Optional[np.ndarray] = None
    ):
        """
        View constructor.

        :param shape: the shape of the view as a list of integers
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

        if dimension >= self.rank():
            raise ValueError(
                f"Cannot resize dimension {dimension} since rank = {self.rank()}")

        if self.shape[dimension] == size:
            return

        old_data: np.ndarray = self.data

        self.shape[dimension] = size
        self.array = kokkos.array(
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
        shape: List[int],
        dtype: Union[DataTypeClass, type] = real,
        space: MemorySpace = MemorySpace.MemorySpaceDefault,
        layout: Layout = Layout.LayoutDefault,
        trait: Trait = Trait.TraitDefault,
        array: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize the view

        :param shape: the shape of the view as a list of integers
        :param dtype: the data type of the view, either a pykokkos DataType or "int" or "float".
        :param space: the memory space of the view. Will be set to the execution space of the view by default.
        :param layout: the layout of the view in memory.
        :param trait: the memory trait of the view
        :param array: the numpy array if trait is Unmanaged
        """

        self.shape: List[int] = shape
        self.dtype: Optional[DataType] = self._get_type(dtype)
        if self.dtype is None:
            sys.exit(f"ERROR: Invalid dtype {dtype}")

        if space is MemorySpace.MemorySpaceDefault:
            space = get_default_memory_space(km.get_default_space())

        if layout is Layout.LayoutDefault:
            layout = get_default_layout(space)

        # only allow CudaSpace view for cupy arrays
        if space is MemorySpace.CudaSpace and trait is not trait.Unmanaged:
            space = MemorySpace.HostSpace

        self.space: MemorySpace = space
        self.layout: Layout = layout
        self.trait: Trait = trait

        if trait is trait.Unmanaged:
            self.array = kokkos.unmanaged_array(array, dtype=self.dtype.value, space=self.space.value, layout=self.layout.value)
        else:
            self.array = kokkos.array("", shape, None, None, self.dtype.value, space.value, layout.value, trait.value)
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

            return DataType[dtype.__name__]

        if dtype is int:
            return DataType["int32"]
        if dtype is float:
            return DataType["double"]

        return None

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
        self.array = kokkos.array(
            self.data, dtype=parent_view.dtype.value, space=parent_view.space.value,
            layout=parent_view.layout.value, trait=kokkos.Unmanaged)
        self.shape: List[int] = list(self.data.shape)

        self.parent_slice: List[Union[int, slice]]
        self.parent_slice = self._create_slice(data_slice)

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

    if np_dtype is np.int16:
        dtype = int16
    elif np_dtype is np.int32:
        dtype = int32
    elif np_dtype is np.int64:
        dtype = int64
    elif np_dtype is np.uint16:
        dtype = uint16
    elif np_dtype is np.uint32:
        dtype = uint32
    elif np_dtype is np.uint64:
        dtype = uint64
    elif np_dtype is np.float32:
        dtype = DataType.float # PyKokkos float
    elif np_dtype is np.float64:
        dtype = double
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

    return View(list(array.shape), dtype, space=space, trait=Trait.Unmanaged, array=array, layout=layout)

def from_cupy(array) -> ViewType:
    """
    Create a PyKokkos View from a cupy array

    :param array: the cupy array
    """

    np_dtype = array.dtype.type

    if np_dtype is np.int16:
        ctype = ctypes.c_int16
    elif np_dtype is np.int32:
        ctype = ctypes.c_int32
    elif np_dtype is np.int64:
        ctype = ctypes.c_int64
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

    return from_numpy(np_array, MemorySpace.CudaSpace, layout)

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
