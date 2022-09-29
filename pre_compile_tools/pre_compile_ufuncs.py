"""
The purpose of this module is to pre-compile
all ufunc implementations for cases where
the overhead from the JIT workflow causes
intractable slowdowns. A common use case
is the hypothesis library-driven generation
of large sets of test cases in the array API
conformance test suite.
"""

from inspect import getmembers, isfunction
import logging

import pykokkos as pk
from pykokkos.lib import ufuncs

from tqdm import tqdm


def main():
    # disable logging of translation/compile times
    # to get a nice clean progress bar for the overall
    # ufunc compile progress
    logging.disable(logging.INFO)
    function_list = getmembers(ufuncs, isfunction)
    # only call the parent ufunc, not the lower
    # level kernels/workunits directly
    filtered_function_list = []
    for f in function_list:
        if not "impl" in f[0]:
            filtered_function_list.append(f)
    # TODO: expand types and view dimensions for
    # ufunc pre-compilation as the support
    # grows more broadly for more dims and types in ufuncs
    v = pk.View([2], dtype=pk.double)
    v2 = pk.View([2, 1], dtype=pk.double)
    for func in tqdm(filtered_function_list):
        func_obj = func[1]
        # try compiling the ufunc as binary, then
        # as unary if that fails
        try:
            func_obj(v, v)
        except TypeError:
            func_obj(v)
        except RuntimeError:
            # some cases like matmul have stricter
            # signature requirements
            func_obj(v, v2)


if __name__ == "__main__":
    main()
