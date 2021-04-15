from __future__ import annotations
from enum import Enum

import pykokkos as pk


class Units(Enum):
    UNITS_REAL = 0
    UNITS_LJ = 1
    UNITS_METAL = 2


class LatticeType(Enum):
    LATTICE_SC = 0
    LATTICE_FCC = 1


class IntegratorType(Enum):
    INTEGRATOR_NVE = 0


class BinningType(Enum):
    BINNING_KKSORT = 0


class CommType(Enum):
    COMM_SERIAL = 0
    COMM_MPI = 1


class ForceType(Enum):
    FORCE_LJ = 0
    FORCE_LJ_IDIAL = 1
    FORCE_SNAP = 2


class ForceIterationType(Enum):
    FORCE_ITER_CELL_FULL = 0
    FORCE_ITER_NEIGH_FULL = 1
    FORCE_ITER_NEIGH_HALF = 2


class NeighborType(Enum):
    NEIGH_NONE = 0
    NEIGH_CSR = 1
    NEIGH_CSR_MAPCONSTR = 2
    NEIGH_2D = 3


class InputFileType(Enum):
    INPUT_LAMMPS = 0


MAX_TYPES_STACKPARAMS: int = 12


# T_INT => int
# T_FLOAT => float
# T_X_FLOAT => float
# T_F_FLOAT => float


class t_x(pk.View):
    def __init__(self, x: int = 0, y: int = 3, data_type: type = pk.double, layout: pk.Layout = pk.Layout.LayoutRight):
        super().__init__([x, y], data_type, layout=layout)


class t_v(pk.View):
    def __init__(self, x: int = 0, y: int = 3, data_type: type = pk.double):
        super().__init__([x, y], data_type)


class t_f(pk.View):
    def __init__(self, x: int = 0, y: int = 3, data_type: type = pk.double):
        super().__init__([x, y], data_type)


class t_type(pk.View):
    def __init__(self, x: int = 0, data_type: type = pk.int32):
        super().__init__([x], data_type)


class t_id(pk.View):
    def __init__(self, x: int = 0, data_type: type = pk.int32):
        super().__init__([x], data_type)


class t_q(pk.View):
    def __init__(self, x: int = 0, data_type: type = pk.double):
        super().__init__([x], data_type)


class t_mass(pk.View):
    def __init__(self, x: int = 0, data_type: type = pk.double):
        super().__init__([x], data_type)


@pk.classtype
class t_scalar3:
    def __init__(self):
        self.x: int = 0
        self.y: int = 0
        self.z: int = 0

    def init_copy(self, rhs: t_scalar3) -> t_scalar3:
        init: t_scalar3 = t_scalar3()

        init.x = rhs.x
        init.y = rhs.y
        init.z = rhs.z

        return init

    def init_scalar(self, x_: int, y_: int, z_: int) -> t_scalar3:
        init: t_scalar3 = t_scalar3()

        init.x = x_
        init.y = y_
        init.z = z_

        return init
