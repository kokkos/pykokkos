
from system import System
from binning import Binning
import pykokkos as pk

class Neighbor:
    def __init__(self):
        # TODO: Unused
        # int neigh_type;
        # bool comm_newton;
        pass

    def init(self, neighcut : float) -> None:
        # TODO: Unused
        pass

    def create_neigh_list(self, system : System, binning : Binning, half_neigh_ : bool, ghost_neighs_ : bool):
        # TODO: Unused
        pass

    def get_neigh_list(self) -> pk.View2D:
        # TODO: Unused
        pass

    def name(self) -> str:
        # TODO: Unused
        pass

#template<int Type>
#struct NeighborAdaptor {
#  typedef Neighbor type;
#};
#include <modules_neighbor.h>
