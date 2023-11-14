from typing import List

from system import System
from binning import Binning
from neighbor import Neighbor


class Force:
    def __init__(self, args: List[str], system: System, half_neigh: bool):
        self.half_neigh = half_neigh

        # Will be properly initialized from ExaMiniMD
        self.comm_newton = False

    def init_coeff(self, nargs: int, args: List[str]) -> None:
        pass

    def compute(self, system: System, binning: Binning, neigh: Neighbor) -> None:
        pass

    def compute_energy(self, system: System, binning: Binning, neigh: Neighbor) -> float:
        return 0.0

    def name(self) -> str:
        return "ForceNone"
