#!/home/nalawar/anaconda3/envs/py38/bin/python3.8
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
import os
import sys
import traceback

__author__ = "Jonathan R. Madsen"
__copyright__ = "Copyright 2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
__credits__ = ["Kokkos"]
__license__ = "BSD-3"
__version__ = "3.1.1"
__maintainer__ = "Jonathan R. Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"

try:
    from . import libpykokkos
    from .libpykokkos import *
    from .utility import *

    __all__ = ['version_info',
               'build_info',
               'version',
               'libpykokkos',
               'array',
               ]

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.modules[__name__].__setattr__(
    "version_info",
    (0,
     0,
     1))
sys.modules[__name__].__setattr__(
    "version",
    "0.0.1")
sys.modules[__name__].__setattr__(
    "build_info",
    {"library_architecture": "x86_64",
     "system_name": "Linux",
     "system_version": "5.4.0-51-generic",
     "build_type": "Release",
     "compiler": "/home/nalawar/Kokkos/kokkos/bin/nvcc_wrapper",
     "compiler_id": "GNU",
     "compiler_version": "7.5.0"})

version_info = sys.modules[__name__].__getattribute__("version_info")
'''Tuple of version fields'''

build_info = sys.modules[__name__].__getattribute__("build_info")
'''Build information'''

version = sys.modules[__name__].__getattribute__("version")
'''Version string'''
