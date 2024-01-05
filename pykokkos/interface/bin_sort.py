from typing import List, Union

from .views import View


class BinOp:
    def __init__(
            self,
            keys: View,
            max_bins: Union[int, List[int]],
            min_value: Union[float, List[float]],
            max_value: Union[float, List[float]]):
        self.keys = keys
        self.max_bins = max_bins
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def get_type(dim: int, key_view_type: str) -> str:
        return f"Kokkos::BinOp{dim}D<{key_view_type}>"


class BinOp1D(BinOp):
    def __init__(
            self,
            keys: View,
            max_bins: int,
            min_value: float,
            max_value: float):
        super().__init__(keys, max_bins, min_value, max_value)


class BinOp3D(BinOp):
    def __init__(
            self,
            keys: View,
            max_bins: List[int],
            min_value: List[float],
            max_value: List[float]):
        super().__init__(keys, max_bins, min_value, max_value)


class BinSort:
    def __init__(
            self,
            keys: View,
            bin_op: BinOp,
            sort_within_bins: bool = False):
        self.keys = keys
        self.bin_op = bin_op
        self.sort_within_bins = sort_within_bins

    @staticmethod
    def get_type(key_view_type: str, bin_op_type: str, space: str) -> str:
        return f"Kokkos::BinSort<{key_view_type},{bin_op_type},{space},int>"

    def sort(self, values: View) -> None:
        pass

    def get_bin_count(self) -> View:
        pass

    def get_bin_offsets(self) -> View:
        pass

    def get_permute_vector(self) -> View:
        pass

    def create_permute_vector(self) -> None:
        pass
