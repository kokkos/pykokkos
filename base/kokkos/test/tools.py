#!@PYTHON_EXECUTABLE@
# ************************************************************************
#
#                        Kokkos v. 3.0
#       Copyright (2020) National Technology & Engineering
#               Solutions of Sandia, LLC (NTESS).
#
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Christian R. Trott (crtrott@sandia.gov)
#
# ************************************************************************
#

from __future__ import absolute_import

__author__ = "Jonathan R. Madsen"
__copyright__ = (
    "Copyright 2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
)
__credits__ = ["Kokkos"]
__license__ = "BSD-3"
__version__ = "@PROJECT_VERSION@"
__maintainer__ = "Jonathan R. Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"


import kokkos
import unittest


data = {}
temp = {}
count = 0
regions = []


def print_help(arg0):
    pass


def parse_args(argc, argv):
    data["parse_args"] = (argc, argv)


def initialize(seq, ver, deviceCount, deviceInfo):
    if "initialize" not in data:
        data["initialize"] = 1
    else:
        data["initialize"] += 1


def finalize():
    if "finalize" not in data:
        data["finalize"] = 1
    else:
        data["finalize"] += 1


def begin_parallel(name, devid):
    global count

    count += 1
    kernid = count
    data[f"begin_{name}"] = kernid
    temp[kernid] = name
    return kernid


def end_parallel(kernid):
    name = temp[kernid]
    data[f"end_{name}"] = kernid


def push_region(name):
    global regions

    data[name] = [True, False]
    regions += [name]


def pop_region():
    data[regions[-1]][1] = True


def alloc_data(handle, label, ptr, size):
    data[f"alloc_{label}"] = (size, handle.name)


def dealloc_data(handle, label, ptr, size):
    data[f"dealloc_{label}"] = (size, handle.name)


def create_prof(name):
    global count

    count += 1
    secid = count
    data[f"create_{name}"] = secid
    temp[secid] = name
    return secid


def start_prof(secid):
    name = temp[secid]
    data[f"start_{name}"] = secid


def stop_prof(secid):
    name = temp[secid]
    data[f"stop_{name}"] = secid


def destroy_prof(secid):
    name = temp[secid]
    data[f"destroy_{name}"] = secid


def prof_event(name):
    data[f"{name}"] = True


def begin_deep_copy(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr, size):
    data[f"begin_{dst_name}/{dst_handle.name}/dst"] = [size, True, False]
    data[f"begin_{src_name}/{src_handle.name}/src"] = [size, True, False]
    temp["deep_copy"] = (dst_handle, dst_name, src_handle, src_name, size)


def end_deep_copy():
    (dst_handle, dst_name, src_handle, src_name, size) = temp["deep_copy"]
    del temp["deep_copy"]

    data[f"begin_{dst_name}/{dst_handle.name}/dst"][-1] = True
    data[f"begin_{src_name}/{src_handle.name}/src"][-1] = True


class PyKokkosBaseToolsTests(unittest.TestCase):
    """ToolsTests"""

    @classmethod
    def setUpClass(self):
        import gc

        kokkos.tools.set_parse_args_callback(parse_args)
        kokkos.tools.set_print_help_callback(print_help)
        kokkos.tools.set_init_callback(initialize)
        kokkos.tools.set_finalize_callback(finalize)

        kokkos.initialize()
        # self.assertTrue(data["initialize"])

        # configure internal testing data before setting other callbacks
        kokkos.tools._internal.setup()

        for itr in ("parallel_for", "parallel_reduce", "parallel_scan", "fence"):
            import sys

            begin_func = getattr(sys.modules[__name__], "begin_parallel")
            end_func = getattr(sys.modules[__name__], "end_parallel")
            getattr(kokkos.tools, f"set_begin_{itr}_callback")(begin_func)
            getattr(kokkos.tools, f"set_end_{itr}_callback")(end_func)

        kokkos.tools.set_push_region_callback(push_region)
        kokkos.tools.set_pop_region_callback(pop_region)
        kokkos.tools.set_allocate_data_callback(alloc_data)
        kokkos.tools.set_deallocate_data_callback(dealloc_data)
        kokkos.tools.set_create_profile_section_callback(create_prof)
        kokkos.tools.set_start_profile_section_callback(start_prof)
        kokkos.tools.set_stop_profile_section_callback(stop_prof)
        kokkos.tools.set_destroy_profile_section_callback(destroy_prof)
        kokkos.tools.set_profile_event_callback(prof_event)
        kokkos.tools.set_begin_deep_copy_callback(begin_deep_copy)
        kokkos.tools.set_end_deep_copy_callback(end_deep_copy)

        # run internal testing (parallel_for, parallel_reduce, etc.)
        kokkos.tools._internal.test()

        # disable parallel callbacks
        for itr in ("parallel_for", "parallel_reduce", "parallel_scan"):
            getattr(kokkos.tools, f"set_begin_{itr}_callback")(None)
            getattr(kokkos.tools, f"set_end_{itr}_callback")(None)

        # generate some profiling data in python
        source = kokkos.array(
            "python_source", [10], space=kokkos.Host, dtype=kokkos.int64
        )
        target = kokkos.array(
            "python_target", [10], space=kokkos.Host, dtype=kokkos.int64
        )
        kokkos.deep_copy(target, source)

        # delete the source and target and run garbage collector
        del source
        del target
        gc.collect()

    @classmethod
    def tearDownClass(self):
        print("")
        for key, item in data.items():
            print("{:30} : {}".format(key, item))

        for itr in (
            "parse_args",
            "print_help",
            "init",
            "finalize",
            "push_region",
            "pop_region",
            "allocate_data",
            "deallocate_data",
            "create_profile_section",
            "start_profile_section",
            "stop_profile_section",
            "destroy_profile_section",
            "begin_parallel_for",
            "begin_parallel_reduce",
            "begin_parallel_scan",
            "begin_fence",
            "begin_deep_copy",
            "end_parallel_for",
            "end_parallel_reduce",
            "end_parallel_scan",
            "end_fence",
            "end_deep_copy",
            "profile_event",
        ):
            getattr(kokkos.tools, f"set_{itr}_callback")(None)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def fib(self, n):
        return n if n < 2 else self.fib(n - 1) + self.fib(n - 2)

    def test_routines(self):
        """cxx_routines"""

        self.assertTrue("begin_cxx_parallel_for" in data)
        self.assertTrue("begin_cxx_parallel_reduce" in data)
        self.assertTrue("begin_cxx_parallel_scan" in data)

        self.assertTrue("end_cxx_parallel_for" in data)
        self.assertTrue("end_cxx_parallel_reduce" in data)
        self.assertTrue("end_cxx_parallel_scan" in data)

        self.assertTrue("create_cxx_created_section" in data)
        self.assertTrue("start_cxx_created_section" in data)
        self.assertTrue("stop_cxx_created_section" in data)
        self.assertTrue("destroy_cxx_created_section" in data)

        self.assertTrue("create_python_profile_section" in data)
        self.assertTrue("start_python_profile_section" in data)
        self.assertTrue("stop_python_profile_section" in data)
        self.assertTrue("destroy_python_profile_section" in data)

        self.assertCountEqual(data["begin_cxx_target/Host/dst"], [40, True, True])
        self.assertCountEqual(data["begin_cxx_source/Host/src"], [40, True, True])

        self.assertCountEqual(data["cxx_push_region"], [True, True])

        self.assertCountEqual(data["alloc_python_source"], [80, "Host"])
        self.assertCountEqual(data["alloc_python_target"], [80, "Host"])
        self.assertCountEqual(data["dealloc_python_source"], [80, "Host"])
        self.assertCountEqual(data["dealloc_python_target"], [80, "Host"])
        self.assertCountEqual(data["dealloc_cxx_source"], [40, "Host"])
        self.assertCountEqual(data["dealloc_cxx_target"], [40, "Host"])

    def test_region(self):
        """python_region"""
        kokkos.tools.push_region(self.shortDescription())
        kokkos.tools.pop_region()

    def test_profile_section(self):
        """python_profile_section"""
        idx = kokkos.tools.create_profile_section(self.shortDescription())
        kokkos.tools.start_section(idx)
        self.fib(10)
        kokkos.tools.stop_section(idx)
        kokkos.tools.destroy_profile_section(idx)

    def test_mark_event(self):
        """python_mark_event"""
        kokkos.tools.mark_event(self.shortDescription())

    def test_declare_metadata(self):
        """python_declare_metadata"""
        kokkos.tools.declare_metadata("dogs", "good")


# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    run()
