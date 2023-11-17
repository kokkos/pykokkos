from binning import Binning
from force import Force
from force_types.force_lj_neigh import ForceLJNeigh
from neighbor_types.neighbor_2d import Neighbor2D
from system import System
from types_h import ForceIterationType, ForceType, NeighborType
from input import Input


def force_modules_instantiation(exa_input: Input, system: System, binning: Binning) -> Force:
    half_neigh: bool = exa_input.force_iteration_type == ForceIterationType.FORCE_ITER_NEIGH_HALF.value
    if exa_input.force_type == ForceType.FORCE_LJ.value:
        if exa_input.neighbor_type == NeighborType.NEIGH_2D.value:
            return ForceLJNeigh(exa_input.input_data.words[exa_input.force_line], system, half_neigh)
    elif exa_input.force_type == ForceType.FORCE_LJ_IDIAL.value:
        return None
    elif exa_input.force_type == ForceType.FORCE_SNAP.value:
        return None

    return None
