from system import System
from binning import Binning
from neighbor import Neighbor
from force import Force
from comm import Comm


class PotE:
    def __init__(self, comm: Comm):
        self.comm = comm

    def compute(self, system: System, binning: Binning, neighbor: Neighbor, force: Force) -> float:
        PE: float = force.compute_energy(system, binning, neighbor)
        self.comm.reduce_float(PE, 1)
        return PE
