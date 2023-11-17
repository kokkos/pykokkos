from neighbor import Neighbor
from neighbor_types.neighbor_2d import Neighbor2D
from types_h import NeighborType
from input import Input


def neighbor_modules_instantiation(exa_input: Input) -> Neighbor:
    if exa_input.neighbor_type == NeighborType.NEIGH_2D.value:
        neighbor = Neighbor2D()
        neighbor.init(exa_input.force_cutoff + exa_input.neighbor_skin)
        return neighbor

    return None
