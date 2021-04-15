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
from . import libpykokkos as lib

__author__ = "Jonathan R. Madsen"
__copyright__ = "Copyright 2020, National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
__credits__ = ["Kokkos"]
__license__ = "BSD-3"
__version__ = "3.1.1"
__maintainer__ = "Jonathan R. Madsen"
__email__ = "jrmadsen@lbl.gov"
__status__ = "Development"


def array(label, shape, dtype=lib.double, space=lib.HostSpace, layout=None,
          trait=None, dynamic=False):
    # print("dtype = {}, space = {}".format(dtype, space))
    _prefix = "KokkosView"
    if dynamic:
        _prefix = "KokkosDynView"
    _space = lib.get_memory_space(space)
    _dtype = lib.get_dtype(dtype)
    _name = None
    if layout is not None:
        _layout = lib.get_layout(layout)
        # LayoutRight is the default
        # if _layout != "LayoutRight":
        _name = "{}_{}_{}_{}".format(_prefix, _dtype, _layout, _space)
    if trait is not None:
        _trait = lib.get_memory_trait(trait)
        if _trait == "Unmanaged":
            raise ValueError("Use unmanaged_array() for the unmanaged view memory trait")
        _name = "{}_{}_{}_{}".format(_prefix, _dtype, _space, _trait)
    if _name is None:
        _name = "{}_{}_{}".format(_prefix, _dtype, _space)
    if not dynamic:
        _name = "{}_{}".format(_name, len(shape))

    return getattr(lib, _name)(label, shape)

def unmanaged_array(array, dtype=lib.double, space=lib.HostSpace, dynamic=False):
    _prefix = "KokkosView"
    if dynamic:
        _prefix = "KokkosDynView"
    _dtype = lib.get_dtype(dtype)
    _space = lib.get_memory_space(space)
    _unmanaged = lib.get_memory_trait(lib.Unmanaged)
    _name = "{}_{}_{}_{}_{}".format(_prefix, _dtype, _space, _unmanaged, array.ndim)

    return getattr(lib, _name)(array, array.shape)
