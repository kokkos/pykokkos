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

try:
    from . import _conftest as conf
except ImportError:
    import kokkos.test._conftest as conf


class PyKokkosBaseViewsTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        kokkos.initialize()

    @classmethod
    def tearDownClass(self):
        if not kokkos.is_finalized():
            kokkos.finalize()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _print_info(self, _data):
        print(
            "[{}]> concrete type  : {}{}".format(
                self.shortDescription(), " " * 7, type(_data[0]).__name__
            )
        )
        print(
            "[{}]> dynamic type   : {}".format(
                self.shortDescription(), type(_data[1]).__name__
            )
        )

    def _print_data(self, _data, _zero, _idx):
        print(
            "[{}]> concrete zero  : {}".format(
                self.shortDescription(), _data[0].create_mirror_view()[_zero]
            )
        )
        print(
            "[{}]> concrete value : {}".format(
                self.shortDescription(), _data[0].create_mirror_view()[_idx]
            )
        )
        print(
            "[{}]> dynamic zero   : {}".format(
                self.shortDescription(), _data[1].create_mirror_view()[_zero]
            )
        )
        print(
            "[{}]> dynamic value  : {}".format(
                self.shortDescription(), _data[1].create_mirror_view()[_idx]
            )
        )

    def test_view_access(self):
        """view_access"""

        print("")
        for itr in conf.get_variants({"memory_traits": [kokkos.Atomic]}):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 1
            _host[1][_idx] = 2

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self._print_data(_data, _zeros, _idx)

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 1)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 2)

    def test_view_iadd(self):
        """view_iadd"""

        print("")
        for itr in conf.get_variants({"memory_traits": [kokkos.Atomic]}):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 1
            _host[1][_idx] = 2

            _host[0][_idx] += 3
            _host[1][_idx] += 3

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 4)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 5)

    def test_view_isub(self):
        """view_isub"""

        print("")
        for itr in conf.get_variants({"memory_traits": [kokkos.Atomic]}):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 10
            _host[1][_idx] = 20

            _host[0][_idx] -= 3
            _host[1][_idx] -= 3

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 7)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 17)

    def test_view_imul(self):
        """view_imul"""

        print("")
        for itr in conf.get_variants({"memory_traits": [kokkos.Atomic]}):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 1
            _host[1][_idx] = 2

            _host[0][_idx] *= 3
            _host[1][_idx] *= 3

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 3)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 6)

    #
    def test_view_create_mirror(self):
        """view_create_mirror"""
        print("")
        for itr in conf.get_variants(
            {"memory_traits": [kokkos.Atomic], "layouts": [kokkos.Unmanaged]}
        ):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 1
            _host[1][_idx] = 2

            _host[0][_idx] *= 3
            _host[1][_idx] *= 3

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 3)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 6)

            _mirror_data = [
                kokkos.create_mirror(_data[0], copy=True),
                kokkos.create_mirror(_data[1], copy=True),
            ]

            self.assertEqual(_mirror_data[0][_zeros], 0)
            self.assertEqual(_mirror_data[1][_zeros], 0)
            self.assertEqual(_mirror_data[0][_idx], 3)
            self.assertEqual(_mirror_data[1][_idx], 6)

            _mirror_data = [
                kokkos.create_mirror(_data[0], copy=False),
                kokkos.create_mirror(_data[1], copy=False),
            ]

            self.assertEqual(_mirror_data[0][_zeros], 0)
            self.assertEqual(_mirror_data[1][_zeros], 0)
            self.assertNotEqual(_mirror_data[0][_idx], 3)
            self.assertNotEqual(_mirror_data[1][_idx], 6)

    #
    def test_view_create_mirror_view(self):
        """view_create_mirror_view"""

        print("")
        for itr in conf.get_variants({"memory_traits": [kokkos.Atomic]}):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 1
            _host[1][_idx] = 2

            _host[0][_idx] *= 3
            _host[1][_idx] *= 3

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 3)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 6)

            _mirror_data = [
                kokkos.create_mirror_view(_data[0], copy=True),
                kokkos.create_mirror_view(_data[1], copy=True),
            ]

            self.assertEqual(_mirror_data[0][_zeros], 0)
            self.assertEqual(_mirror_data[1][_zeros], 0)
            self.assertEqual(_mirror_data[0][_idx], 3)
            self.assertEqual(_mirror_data[1][_idx], 6)

            _mirror_data = [
                kokkos.create_mirror_view(_data[0], copy=False),
                kokkos.create_mirror_view(_data[1], copy=False),
            ]

            self.assertEqual(_mirror_data[0][_zeros], 0)
            self.assertEqual(_mirror_data[1][_zeros], 0)

            if kokkos.get_host_accessible(_data[0].space):
                self.assertEqual(_mirror_data[0][_idx], 3)
            else:
                self.assertNotEqual(_mirror_data[0][_idx], 3)

            if kokkos.get_host_accessible(_data[1].space):
                self.assertEqual(_mirror_data[1][_idx], 6)
            else:
                self.assertNotEqual(_mirror_data[1][_idx], 6)

    #
    def test_view_deep_copy(self):
        """view_deep_copy"""
        print("")
        for itr in conf.get_variants(
            {"memory_traits": [kokkos.Atomic], "layouts": [kokkos.Unmanaged]}
        ):
            _shape = itr[0]
            _idx = itr[1]
            _zeros = itr[2]
            _kwargs = itr[3]
            if _kwargs["trait"] != kokkos.Managed:
                continue

            _data = conf.generate_variant(_shape, **_kwargs)

            self._print_info(_data)

            _host = [_data[0].create_mirror_view(), _data[1].create_mirror_view()]

            _host[0][_idx] = 1
            _host[1][_idx] = 2

            _host[0][_idx] *= 3
            _host[1][_idx] *= 3

            _data[0].deep_copy(_host[0])
            _data[1].deep_copy(_host[1])

            self.assertEqual(_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_data[0].create_mirror_view()[_idx], 3)
            self.assertEqual(_data[1].create_mirror_view()[_idx], 6)

            _copied_data = conf.generate_variant(_shape, **_kwargs)
            kokkos.deep_copy(_copied_data[0], _data[0])
            kokkos.deep_copy(_copied_data[1], _data[1])

            self.assertEqual(_copied_data[0].create_mirror_view()[_zeros], 0)
            self.assertEqual(_copied_data[1].create_mirror_view()[_zeros], 0)
            self.assertEqual(_copied_data[0].create_mirror_view()[_idx], 3)
            self.assertEqual(_copied_data[1].create_mirror_view()[_idx], 6)


# main runner
def run():
    # run all tests
    unittest.main()


if __name__ == "__main__":
    run()
