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


def get_max_concrete_dims():
    import kokkos

    return kokkos.max_concrete_rank


def get_dtypes():
    import kokkos

    return kokkos.dtypes


def get_memory_spaces(only_available=True):
    import kokkos

    return [
        e
        for e in kokkos.memory_spaces
        if not only_available or kokkos.get_memory_space_available(e)
    ]


def get_layouts(only_available=True):
    import kokkos

    return [
        e
        for e in kokkos.layouts
        if not only_available or kokkos.get_layout_available(e)
    ]


def get_memory_traits(only_available=True):
    import kokkos

    return [
        e
        for e in kokkos.memory_traits
        if not only_available or kokkos.get_memory_trait_available(e)
    ]


def generate_variant(*_args, **_kwargs):
    """Generate a view from the variant arguments"""

    import kokkos
    import numpy as np

    con_arr = None
    dyn_arr = None
    if _kwargs["trait"] == kokkos.Unmanaged:
        con_arr = np.zeros(_args[0], dtype=kokkos.convert_dtype(_kwargs["dtype"]))
        con_view = kokkos.unmanaged_array(con_arr, **_kwargs, dynamic=False)
        dyn_arr = np.zeros(_args[0], dtype=kokkos.convert_dtype(_kwargs["dtype"]))
        dyn_view = kokkos.unmanaged_array(dyn_arr, **_kwargs, dynamic=True)
    else:
        con_view = kokkos.array(*_args, **_kwargs, dynamic=False)
        dyn_view = kokkos.array(*_args, **_kwargs, dynamic=True)
    # retain con_arr and dyn_arr since python might run GC and delete them
    return [con_view, dyn_view, con_arr, dyn_arr]


def get_variants(exclude=[]):
    """Return a list of all view variants"""
    _variants = []

    def _skip(_key, _value):
        return _key in exclude.keys() and _value in exclude[_key]

    for _dims in range(1, get_max_concrete_dims()):
        _shape = []
        _idx = []
        _zeros = []

        for i in range(_dims):
            _shape.append(2)
            _zeros.append(0)
            _idx.append((i + 1) % 2)

        for _dtype in get_dtypes():
            if _skip("dtypes", _dtype):
                continue
            for _space in get_memory_spaces():
                if _skip("memory_spaces", _space):
                    continue
                for _layout in get_layouts():
                    if _skip("layouts", _layout):
                        continue
                    for _trait in get_memory_traits():
                        if _skip("memory_traits", _trait):
                            continue
                        _variant = {}
                        _variant["dtype"] = _dtype
                        _variant["space"] = _space
                        _variant["layout"] = _layout
                        _variant["trait"] = _trait
                        _variants.append([_shape, _idx, _zeros, _variant])
    return _variants
