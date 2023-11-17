from abc import ABCMeta, abstractmethod

import pykokkos as pk

from system import System


class Binning(metaclass=ABCMeta):
    def __init__(self, s: System):
        self.system = s

        self.nbinx: int = 0
        self.nbiny: int = 0
        self.nbinz: int = 0
        self.nhalo: int = 0
        self.minx: float = 0.0
        self.maxx: float = 0.0
        self.miny: float = 0.0
        self.maxy: float = 0.0
        self.minz: float = 0.0
        self.maxz: float = 0.0

    @abstractmethod
    def create_binning(
            self, dx: float, dy: float, dz: float, halo_depth: int,
            do_local: bool, do_ghost: bool, sort: bool) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        return "BinningNone"

    # Typedefs
    class t_bincount(pk.View):
        def __init__(self, x: int = 0, y: int = 0, z: int = 0, data_type: type = pk.int32):
            super().__init__([x, y, z], data_type)

    class t_binoffsets(pk.View):
        def __init__(self, x: int = 0, y: int = 0, z: int = 0, data_type: type = pk.int32):
            super().__init__([x, y, z], data_type)

    class t_permute_vector(pk.View):
        def __init__(self, x: int = 0, data_type: type = pk.int32):
            super().__init__([x], data_type)
