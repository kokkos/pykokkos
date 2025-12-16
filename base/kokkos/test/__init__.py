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


import os
import unittest


# discover and run all unittests in the current directory
def run_all_tests():
    # auto discover unittests from *.py files starting character into the test suite
    test_suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern="[a-z]*.py",
    )

    # print the loaded tests
    print(f"============= Loaded Tests =============\n\n {test_suite}\n")

    # create a results object to store test results
    result = unittest.TestResult(verbosity=2)

    # run all tests in test_suite, use result object to store results
    print("\n============= Tests Stdout =============\n")
    # run the tests
    test_suite.run(result)

    # print the results
    print(f"\n============= Results =============\n{result}\n")
    return result
