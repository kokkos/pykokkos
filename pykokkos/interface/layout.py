from enum import Enum

from pykokkos.bindings import kokkos

from .memory_space import MemorySpace


class Layout(Enum):
    LayoutDefault = None
    LayoutLeft = kokkos.LayoutLeft
    LayoutRight = kokkos.LayoutRight
    # LayoutStride = "LayoutStride"
    # LayoutTiled = "LayoutTiled"

def get_default_layout(space: MemorySpace) -> Layout:
    """
    Map from memory spaces to optimal memory layout.

    :param space: the memory space
    :returns: the default layout
    """

    if space is MemorySpace.HostSpace:
        return Layout.LayoutRight
    else:
        return Layout.LayoutLeft