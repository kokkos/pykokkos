# import unittest
# import pykokkos
# from pykokkos.workload import KokkosWorkload, work_unit
# from pykokkos.views import View1D

# class AddWorkload1D(KokkosWorkload):
#     def __init__(self, total_threads: int, initial_value: int, added_value: int):
#         self.total_threads = total_threads
#         self.initial_value = initial_value
#         self.added_value = added_value
#         self.view = View1D(total_threads)
    
#     def run_work_units(self):
#         pykokkos.parallel_for(self.total_threads, self.add_work_unit)

#     @work_unit
#     def init_view(self, tid: int):
#         self.view[tid] = self.initial_value

#     @work_unit
#     def add_work_unit(self, tid: int):
#         self.view[tid] += 1

#     def use_results(self):
#         return

# class ParallelForTest(unittest.TestCase):
#     def setUp(self):
#         self.total_threads: int = 10000
        
#     def test_add_workload_1D(self):
#         initial_value: int = 5
#         added_value: int = 7
#         expected_result: int = initial_value + added_value

#         workload = AddWorkload1D(self.total_threads, initial_value, added_value)
#         workload.execute()

#         for i in range(self.total_threads):
#             result: int = workload.view[i]
#             self.assertEqual(result, expected_result)

# if __name__ == '__main__':
#     unittest.main()