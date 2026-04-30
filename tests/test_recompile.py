"""
pytest: PyKokkos JIT kernel recompilation test.

Tests that modifying a @pk.workunit source file triggers recompilation
and that both versions of the kernel produce correct results.

Workflow:
  1. Write the "buggy" kernel (arr[i] += 2) to a temp module file.
  2. Import it and run it — verify each element increased by 2.
  3. Overwrite the module file with the "fixed" kernel (arr[i] += 1).
  4. Invalidate Python's module cache so the new source is picked up.
  5. Re-import and run — verify each element increased by 1.
"""

import sys
import textwrap
import importlib
from pathlib import Path

import numpy as np
import pykokkos as pk

BUGGY_SOURCE = textwrap.dedent(
    """\
    import pykokkos as pk
 
    @pk.workunit
    def add1(i: int, arr: pk.View1D[int]):
        arr[i] += 2
"""
)

CORRECT_SOURCE = textwrap.dedent(
    """\
    import pykokkos as pk
 
    @pk.workunit
    def add1(i: int, arr: pk.View1D[int]):
        arr[i] += 1
"""
)

MODULE_NAME = "_test_jit_kernel_add1"


def _load_fresh(path: Path):
    """Force a fresh import of the module at *path*, bypassing sys.modules."""
    # Remove any previously cached version.
    sys.modules.pop(MODULE_NAME, None)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


# ---------- Test JIT Recompilation ----------


def test_recompilation(tmp_path):
    kernel_file = tmp_path / "_test_jit_recompile.py"

    # ---- Stage 1: buggy kernel
    # write buggy source
    kernel_file.write_text(BUGGY_SOURCE, encoding="utf-8")

    # load buggy source
    mod_buggy = _load_fresh(kernel_file)

    # run the buggy kernel
    n = 5
    arr_buggy = np.zeros(n, dtype=np.int32)
    pk.parallel_for(n, mod_buggy.add1, arr=arr_buggy)

    # assert add2 array is correct
    try:
        np.testing.assert_equal(arr_buggy, np.zeros(n, dtype=np.int32) + 2)
    except AssertionError as e:
        raise AssertionError("buggy kernel is incorrect") from e

    # ---- Stage 2: correct kernel

    # reload pykokkos to clear cache
    importlib.reload(sys.modules["pykokkos"])

    kernel_file.write_text(CORRECT_SOURCE, encoding="utf-8")
    mod_correct = _load_fresh(kernel_file)

    arr_correct = np.zeros(n, dtype=np.int32)
    pk.parallel_for(n, mod_correct.add1, arr=arr_correct)
    expected = np.zeros(n, dtype=np.int32) + 1
    try:
        np.testing.assert_equal(arr_correct, expected)
    except AssertionError as e:
        raise AssertionError(
            f"kernel is incorrect\nactual:  {arr_correct}\ndesired: {expected}"
        ) from e
