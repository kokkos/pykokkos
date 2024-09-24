from .accumulator import Acc
from .atomic.atomic_fetch_op import (
    atomic_fetch_add, atomic_fetch_and, atomic_fetch_div,
    atomic_fetch_lshift, atomic_fetch_max, atomic_fetch_min,
    atomic_fetch_mod, atomic_fetch_mul, atomic_fetch_or,
    atomic_fetch_rshift, atomic_fetch_sub, atomic_fetch_xor,
    atomic_compare_exchange
)
from .atomic.atomic_op import (
    atomic_add, atomic_increment
)
from .bin_sort import BinSort, BinOp, BinOp1D, BinOp3D
from .data_types import (
    DataType, DataTypeClass,
    int8,
    int16, int32, int64,
    uint8,
    uint16, uint32, uint64,
    float, double, real,
    float32, float64, bool,
)
from .decorators import (
    callback, classtype, Decorator, function, functor, main,
    workload, workunit
)
from .execution_policy import (
    ExecutionPolicy, RangePolicy, MDRangePolicy, TeamPolicy,
    TeamThreadRange, ThreadVectorRange, TeamThreadMDRange, Iterate, Rank
)
from .execution_space import ExecutionSpace, ExecutionSpaceInstance, is_host_execution_space
from .layout import Layout, get_default_layout
from .hierarchical import (
    AUTO, TeamMember, PerTeam, PerThread, single
)
from .mathematical_special_functions import (
    cyl_bessel_j0, cyl_bessel_j1
)
from .memory_space import MemorySpace, get_default_memory_space
from .parallel_dispatch import (
    execute, flush,
    parallel_for, parallel_reduce, parallel_scan,
)
from .random import (
    rand, RandomPool, Random_XorShift64_Pool, Random_XorShift1024_Pool
)
from .timer import Timer
from .views import (
    Subview, Trait,
    View, ViewType, ViewTypeInfo,
    View1D, View2D, View3D, View4D,
    View5D, View6D, View7D, View8D,
    ScratchView, ScratchView1D, ScratchView2D,
    ScratchView3D, ScratchView4D, ScratchView5D,
    ScratchView6D, ScratchView7D, ScratchView8D,
    array, asarray, result_type,
)

from .ext_module import compile_into_module

def fence():
    pass


def printf(fmt_str, *args):
    print(fmt_str % args, end="")


def add_enum_to_globals(enumeration: type) -> None:
    """
    Adds the fields of an enum to the global namespace

    :param enumeration: the enum class
    """

    for member in enumeration:
        globals()[member.name] = member


add_enum_to_globals(ExecutionSpace)
add_enum_to_globals(Layout)
add_enum_to_globals(MemorySpace)
add_enum_to_globals(Trait)
