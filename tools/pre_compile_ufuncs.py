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
        if not "impl" in f[0] and not f[0].startswith("_") and "broadcast_view" not in f[0]:
            filtered_function_list.append(f)
    # TODO: expand types and view dimensions for
    # ufunc pre-compilation as the support
    # grows more broadly for more dims and types in ufuncs
    for dtype in [pk.float64,
                  pk.float32,
                  pk.int8,
                  pk.int16,
                  pk.int32,
                  pk.int64,
                  pk.uint8,
                  pk.uint16,
                  pk.uint32,
                  pk.uint64,
                  ]:
        for shape_v in [[2], [2, 2]]:
            v = pk.View(shape_v, dtype=dtype)
            for func in tqdm(filtered_function_list):
                if len(shape_v) > 1 and "matmul" in func[0]: 
                    continue
                func_obj = func[1]
                # try compiling the ufunc as binary, then
                # as unary if that fails
                try:
                    func_obj(v, v)
                except (NotImplementedError, KeyError):
                    pass
                except TypeError:
                    try:
                        func_obj(v)
                    except (NotImplementedError, RuntimeError, KeyError):
                        pass
                except RuntimeError:
                    # some cases like matmul have stricter
                    # signature requirements
                    if "matmul" in func[0]:
                        new_shape = shape_v[:]
                        new_shape.append(1)
                        v2 = pk.View(new_shape, dtype=dtype)
                        try:
                            func_obj(v, v2)
                        except RuntimeError:
                            pass
                    else:
                        pass


def test_main():
    # force pytest to run main() and produce a pk_cpp
    # directory structure that is useful for saving time
    # with the array API suite invoked by pytest

    # TODO: we shouldn't need to do this long-term; the pk_cpp
    # compilation directory structure shouldn't exclude reuse
    # by other modules
    main()


if __name__ == "__main__":
    main()
