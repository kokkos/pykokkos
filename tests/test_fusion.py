import unittest
import pykokkos as pk
import pytest


# workunits
@pk.workunit
def init_view(i, view, init):
    view[i] = init

@pk.workunit
def init_view_annotated(i: int, view: pk.View1D[int], init: int):
    view[i] = init

@pk.workunit
def add_views(i, view_1, view_2):
    view_1[i] += view_2[i]

@pk.workunit
def add_views_annotated(i: int, view_1: pk.View1D[int], view_2: pk.View1D[int]):
    view_1[i] += view_2[i]

@pk.workunit
def init_view_condition(i, view, var):
    value: int = 0
    if var == 0:
        value = 11
    else:
        value = 22

    view[i] = value

@pk.workunit
def init_view_types(i, view, init):
    view[i] = init


class TestKernelFusion(unittest.TestCase):
    def setUp(self):
        self.iterations: int = 10

    def test_fusion(self):
        value1 = 5
        value2 = 6
        v1 = pk.View((self.iterations,), int)
        v2 = pk.View((self.iterations,), int)

        pk.parallel_for(self.iterations, [init_view, init_view_annotated], args_0={"view": v1, "init": value1}, args_1={"view": v2, "init": value2})
        self.assertEqual(v1[0], value1)
        self.assertEqual(v2[0], value2)

    def test_fusion_after_call(self):
        v1 = pk.View((self.iterations,), int)
        v2 = pk.View((self.iterations,), int)

        v1.fill(0)
        v2.fill(1)

        pk.parallel_for(self.iterations, add_views, view_1=v1, view_2=v2)
        pk.parallel_for(self.iterations, add_views_annotated, view_1=v1, view_2=v2)

        pk.parallel_for(self.iterations, [add_views, add_views_annotated], args_0={"view_1": v1, "view_2": v2}, args_1={"view_1": v1, "view_2": v2})
        self.assertEqual(v1[0], 4)

    def test_fusion_condition(self):
        v1 = pk.View((self.iterations,), int)
        v2 = pk.View((self.iterations,), int)

        pk.parallel_for(self.iterations, [init_view_condition, init_view_condition], args_0={"view": v1, "var": 0}, args_1={"view": v2, "var": 1})
        self.assertEqual(v1[0], 11)
        self.assertEqual(v2[0], 22)

    def test_fusion_change_types(self):
        value1 = 5
        value2 = 6.0
        v1 = pk.View((self.iterations,), int)
        v2 = pk.View((self.iterations,), float)

        pk.parallel_for(self.iterations, [init_view_types, init_view_types], args_0={"view": v1, "init": value1}, args_1={"view": v2, "init": value2})
        self.assertEqual(v1[0], value1)
        self.assertEqual(v2[0], value2)

        pk.parallel_for(self.iterations, [init_view_types, init_view_types], args_0={"view": v2, "init": value2}, args_1={"view": v1, "init": value1})
        self.assertEqual(v1[0], value1)
        self.assertEqual(v2[0], value2)

if __name__ == '__main__':
    unittest.main()
