# pykokkos-base
[![linux-ci](https://github.com/kokkos/pykokkos-base/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/kokkos/pykokkos-base/actions/workflows/linux-ci.yml)
[![python-package](https://github.com/kokkos/pykokkos-base/actions/workflows/python-package.yml/badge.svg)](https://github.com/kokkos/pykokkos-base/actions/workflows/python-package.yml)

> Additional Documentation can be found in [Wiki](https://github.com/kokkos/pykokkos-base/wiki)

## Overview

This package contains the minimal set of bindings for [Kokkos](https://github.com/kokkos/kokkos)
interoperability with Python:

- Free-standing function bindings
    - `Kokkos::initialize(...)`
    - `Kokkos::finalize()`
    - `Kokkos::is_initialized()`
    - `Kokkos::deep_copy(...)`
    - `Kokkos::create_mirror(...)`
    - `Kokkos::create_mirror_view(...)`
    - `Kokkos::Tools::profileLibraryLoaded()`
    - `Kokkos::Tools::pushRegion(...)`
    - `Kokkos::Tools::popRegion()`
    - `Kokkos::Tools::createProfileSection(...)`
    - `Kokkos::Tools::destroyProfileSection(...)`
    - `Kokkos::Tools::startSection(...)`
    - `Kokkos::Tools::stopSection(...)`
    - `Kokkos::Tools::markEvent(...)`
    - `Kokkos::Tools::declareMetadata(...)`
    - `Kokkos::Tools::Experimental::set_<...>_callback(...)`
- Data structures
    - `Kokkos::View<...>`
    - `Kokkos::DynRankView<...>`
    - `Kokkos_Profiling_KokkosPDeviceInfo`
    - `Kokkos_Profiling_SpaceHandle`

By importing this package in Python, you can pass the supported Kokkos Views and DynRankViews
from C++ to Python and vice-versa. Furthermore, in Python, these bindings provide interoperability
with numpy and cupy arrays:

```python
import kokkos
import numpy as np

view = kokkos.array([2, 2], dtype=kokkos.double, space=kokkos.CudaUVMSpace,
                    layout=kokkos.LayoutRight, trait=kokkos.RandomAccess,
                    dynamic=False)

arr = np.array(view, copy=False)
```

## Writing Kokkos in Python

In order to write native Kokkos in Python, see [pykokkos](https://github.com/kokkos/pykokkos).

## Installation

You can install this package via CMake or Python's `setup.py`. The important cmake options are:

- `ENABLE_VIEW_RANKS` (integer)
- `ENABLE_LAYOUTS` (bool)
- `ENABLE_MEMORY_TRAITS` (bool)
- `ENABLE_INTERNAL_KOKKOS` (bool)

By default, CMake will enable the layouts and memory traits options if the Kokkos installation was not
built with CUDA support.
If Kokkos was built with CUDA support, `ENABLE_MEMORY_TRAITS` will be disabled by default due to unreasonable
compilation times (> 1 hour).
The `ENABLE_VIEW_RANKS` option (defaults to a value of 4) is the max number of ranks for
`Kokkos::View<...>` that can be returned to Python. For example, value of 4 means that
views of data type `T*`, `T**`, `T***`, and `T****` can be returned to python but
`T*****` and higher cannot. Increasing this value up to 7 can dramatically increase the length
of time required to compile the bindings.

Installation command with specific arguments looks like this:

```
PYKOKKOS_BASE_SETUP_ARGS="-DKokkos_ENABLE_THREADS=OFF -DKokkos_ENABLE_OPENMP=ON -DENABLE_CUDA=ON -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3" pip install ./ --verbose
```

### Kokkos Installation

If the `ENABLE_INTERNAL_KOKKOS` option is not specified the first time CMake is run, CMake will try to
find an existing Kokkos installation. If no existing installation is found, it will build and install
Kokkos from a submodule. When Kokkos is added as a submodule, you can configure the submodule
as you would normally configure Kokkos. However, due to some general awkwardness configuring cmake
from `setup.py` (especially via `pip install`), CMake tries to "automatically" configure
reasonable default CMake settings for the Kokkos submodule.

Here are the steps when Kokkos is added as a submodule:

- Set `BUILD_SHARED_LIBS=ON`
- Set `Kokkos_ENABLE_SERIAL=ON`
- `find_package(OpenMP)`
    - Was OpenMP found?
        - **YES**: set `Kokkos_ENABLE_OPENMP=ON`
        - **NO**: `find_package(Threads)`
            - Was Threads found?
                - **YES**: set `Kokkos_ENABLE_THREADS=ON` (if not Windows)
- `find_package(CUDA)`
    - Was CUDA found?
        - **YES**: set:
            - `Kokkos_ENABLE_CUDA=ON`
            - `Kokkos_ENABLE_CUDA_UVM=ON`
            - `Kokkos_ENABLE_CUDA_LAMBDA=ON`

### Configuring Options via CMake

```console
cmake -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF /path/to/source
```

### Configuring Options via `setup.py`

There are three ways to configure the options:

1. Via the Python argparse options `--enable-<option>` and `--disable-<option>`
2. Setting the `PYKOKKOS_BASE_SETUP_ARGS` environment variable to the CMake options
3. Passing in the CMake options after a `--`

All three lines below are equivalent (deprecated format):

```console
python setup.py install --enable-layouts --disable-memory-traits
PYKOKKOS_BASE_SETUP_ARGS="-DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF" python setup.py install
python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF
```

### Configuring Options via `pip`

Pip does not handle build options well. Thus, it is recommended to use the `PYKOKKOS_BASE_SETUP_ARGS`
environment variable noted above. 

We suggest using the following line to install pykokkos-base:

```
PYKOKKOS_BASE_SETUP_ARGS="-DKokkos_ENABLE_THREADS=OFF \
    -DKokkos_ENABLE_OPENMP=ON \
    -DENABLE_CUDA=ON \
    -DENABLE_LAYOUTS=ON \
    -DENABLE_MEMORY_TRAITS=OFF \
    -DENABLE_VIEW_RANKS=3" \
    pip install ./ --verbose
```

`--verbose` is optional, but it shows installation progress in real time.

> `pip install ./` will build against the latest release in the PyPi repository.
> In order to pip install from this repository, use `pip install --user -e .`

## Differences vs. Kokkos C++

### Deep Copy and Host Mirror

If you are not familiar with `Kokkos::deep_copy(...)`, `Kokkos::create_mirror(...)`, `Kokkos::create_mirror_view(...)`, read this
[Kokkos Wiki entry](https://github.com/kokkos/kokkos/wiki/View#643-deep-copy-and-hostmirror).

When Kokkos views are allocated on a non-host memory space, this data is not directly accessible in Python. Any
attempt to read or modify the data will result in a fatal error. In C++, Kokkos developers usually
perform two distinct operations: create a mirror or mirror-view and then execute a deep-copy, e.g.:

```cpp
// assume MemorySpace is Kokkos::Cuda or similar
Kokkos::View<int*, MemorySpace> a ("a", 10);

// Allocate a view in HostSpace with the layout and padding of a
auto b = create_mirror(a);
// This is always a memcopy
Kokkos::deep_copy (b, a);

// This may not allocate a new view if a is in host space
auto c = Kokkos::create_mirror_view(a);
// This is a no-op if MemorySpace is HostSpace
Kokkos::deep_copy (c, a)
```

The python equivalent is available via standalone functions:

```python
# assume MemorySpace is kokkos.CudaSpace or similar
a = kokkos.array("a", shape=[10], space=MemorySpace)

# Allocate a view in HostSpace with the layout and padding of a
b = kokkos.create_mirror(a)
# copy memory
kokkos.deep_copy(b, a)

# This may not allocate a new view if a is in host space
c = kokkos.create_mirror_view(a)
# This is a no-op if MemorySpace is HostSpace
kokkos.deep_copy(c, a)
```

However, this makes it cumbersome to print data in python:

```python
# assume MemorySpace is kokkos.CudaSpace or similar
a = kokkos.array("a", shape=[10], space=MemorySpace)

def print_data(inp):
    v = kokkos.create_mirror_view(inp)
    kokkos.deep_copy(v, inp)
    for i in range(v.shape[0]):
        print(f"v({i}) = {v[i]}")

print_data(a)
```

Thus, the _member functions_ `create_mirror()` and `create_mirror_view()` accept a boolean
`copy` argument which **defaults to True**, e.g.:

```python
a = kokkos.array("a", shape=[10], space=MemorySpace)

# this:
b = a.create_mirror()

# is implicitly:
b = a.create_mirror(copy=True)
```

Thus, our `print_data` function above does not need handle mirror creation because
we can replace `print_data(a)` with `print_data(a.create_mirror())` or `print_data(a.create_mirror_view())`:

```python
# assume MemorySpace is kokkos.CudaSpace or similar
a = kokkos.array("a", shape=[10], space=MemorySpace)

def print_data(v):
    for i in range(v.shape[0]):
        print(f"v({i}) = {v[i]}")

print_data(a.create_mirror_view())
```

In fact, the free-standing `kokkos.create_mirror(...)` and `kokkoos.create_mirror_view(...)` simply use this member function
and default the `copy` argument to `False`:

```python
def create_mirror(src, copy=False):
    """Performs Kokkos::create_mirror"""
    return src.create_mirror(copy)


def create_mirror_view(src, copy=False):
    """Performs Kokkos::create_mirror_view"""
    return src.create_mirror_view(copy)
```

## Example

### Overview

This example is designed to emulate a work-flow where the user has code using Kokkos in C++ and writes python bindings to those functions. A python script is used as the `"main"`:

- `ex-numpy.py` imports the kokkos bindings
- Calls a routine in the "users" python bindings to a C++ function which returns a `Kokkos::View`
- This view is then converted to a numpy array in python and printed via the numpy capabilities.

### Files

- [ex-generate.cpp](https://github.com/kokkos/kokkos-python/blob/main/examples/ex-generate.cpp)
  - This is the python bindings to the user code
- [user.cpp](https://github.com/kokkos/kokkos-python/blob/main/examples/user.cpp)
  - This is the implementation of the user's code which returns a `Kokkos::View<double**, Kokkos::HostSpace>`
- [ex-numpy.py](https://github.com/kokkos/kokkos-python/blob/main/examples/ex-numpy.py)
  - This is the "main"

#### ex-numpy.py

```python
#!/usr/bin/env python

import argparse
import numpy as np

#
# The python bindings for generate_view are in ex-generate.cpp
# The declaration and definition of generate_view are in user.hpp and user.cpp
# The generate_view function will return a Kokkos::View and will be converted
# to a numpy array
from ex_generate import generate_view, modify_view

#
# Importing this module is necessary to call kokkos init/finalize and
# import the python bindings to Kokkos::View which generate_view will
# return
#
import kokkos


def print_data(label, name, data):
    # write the type info
    print(
        "{:12} : {} (ndim={}, shape={})".format(
            label, type(data).__name__, data.ndim, data.shape
        )
    )

    # print the data
    if data.ndim == 1:
        for i in range(data.shape[0]):
            print("{:8}({}) = {}".format(name, i, data[i]))
    elif data.ndim == 2:
        for i in range(data.shape[0]):
            print(
                "{:8}({}) = [{}]".format(
                    name,
                    i,
                    " ".join("{}".format(data[i, j]) for j in range(data.shape[1])),
                )
            )
    else:
        raise ValueError("only 2 dimensions are supported")


def user_bindings(args):
    # get the kokkos view
    view = generate_view(args.ndim)
    print_data("Kokkos View", "view", view.create_mirror_view())

    # modify view (verify that casting works)
    modify_view(view)
    print_data("Modify View", "view", view.create_mirror_view())

    # wrap the buffer protocal as numpy array without copying the data
    arr = np.array(view.create_mirror_view(), copy=False)
    print_data("Numpy Array", "arr", arr)


def to_numpy(args):
    # get the kokkos view
    view = kokkos.array(
        "python_allocated_view",
        [args.ndim],
        dtype=kokkos.double,
        space=kokkos.DefaultHostMemorySpace,
    )

    for i in range(view.shape[0]):
        view[i] = i * (i % 2)
    print_data("Kokkos View", "view", view)

    # wrap the buffer protocal as numpy array without copying the data
    arr = np.array(view, copy=False)
    print_data("Numpy Array", "arr", arr)


def from_numpy(args):
    arr = np.ones([args.ndim, args.ndim], dtype=np.int32)
    for i in range(args.ndim):
        arr[i, i] = 0

    print_data("Numpy Array", "arr", arr)

    view = kokkos.array(arr, dtype=kokkos.int32, dynamic=True)
    print_data("Kokkos View", "view", view)


if __name__ == "__main__":
    try:
        kokkos.initialize()
        parser = argparse.ArgumentParser()
        parser.add_argument("-n", "--ndim", default=10, help="X dimension", type=int)
        args, argv = parser.parse_known_args()
        print("Executing to numpy...")
        to_numpy(args)
        print("Executing from numpy...")
        from_numpy(args)
        print("Executing user bindings...")
        user_bindings(args)
        kokkos.finalize()
    except Exception as e:
        import sys
        import traceback

        print(f"{e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)
```

### Build and Run

```console
mkdir build
cd build
cmake -DENABLE_EXAMPLES=ON ..
make
python ./ex-numpy.py
```

### Expected Output

```console
[user-bindings]> Generating View... Done.
[user-bindings]> Modifying View... Done.
Executing to numpy...
Kokkos View  : KokkosView_float64_HostSpace_LayoutRight_1 (ndim=1, shape=[10])
view    (0) = 0.0
view    (1) = 1.0
view    (2) = 0.0
view    (3) = 3.0
view    (4) = 0.0
view    (5) = 5.0
view    (6) = 0.0
view    (7) = 7.0
view    (8) = 0.0
view    (9) = 9.0
Numpy Array  : ndarray (ndim=1, shape=(10,))
arr     (0) = 0.0
arr     (1) = 1.0
arr     (2) = 0.0
arr     (3) = 3.0
arr     (4) = 0.0
arr     (5) = 5.0
arr     (6) = 0.0
arr     (7) = 7.0
arr     (8) = 0.0
arr     (9) = 9.0
Executing from numpy...
Numpy Array  : ndarray (ndim=2, shape=(10, 10))
arr     (0) = [0 1 1 1 1 1 1 1 1 1]
arr     (1) = [1 0 1 1 1 1 1 1 1 1]
arr     (2) = [1 1 0 1 1 1 1 1 1 1]
arr     (3) = [1 1 1 0 1 1 1 1 1 1]
arr     (4) = [1 1 1 1 0 1 1 1 1 1]
arr     (5) = [1 1 1 1 1 0 1 1 1 1]
arr     (6) = [1 1 1 1 1 1 0 1 1 1]
arr     (7) = [1 1 1 1 1 1 1 0 1 1]
arr     (8) = [1 1 1 1 1 1 1 1 0 1]
arr     (9) = [1 1 1 1 1 1 1 1 1 0]
Kokkos View  : KokkosDynRankView_int32_HostSpace_LayoutRight (ndim=2, shape=[10, 10, 1, 1, 1, 1, 1])
view    (0) = [0 1 1 1 1 1 1 1 1 1]
view    (1) = [1 0 1 1 1 1 1 1 1 1]
view    (2) = [1 1 0 1 1 1 1 1 1 1]
view    (3) = [1 1 1 0 1 1 1 1 1 1]
view    (4) = [1 1 1 1 0 1 1 1 1 1]
view    (5) = [1 1 1 1 1 0 1 1 1 1]
view    (6) = [1 1 1 1 1 1 0 1 1 1]
view    (7) = [1 1 1 1 1 1 1 0 1 1]
view    (8) = [1 1 1 1 1 1 1 1 0 1]
view    (9) = [1 1 1 1 1 1 1 1 1 0]
Executing user bindings...
Kokkos View  : KokkosView_float64_HostSpace_LayoutRight_2 (ndim=2, shape=[10, 2])
view    (0) = [-1.0 1.0]
view    (1) = [-2.0 2.0]
view    (2) = [-3.0 3.0]
view    (3) = [-4.0 4.0]
view    (4) = [-5.0 5.0]
view    (5) = [-6.0 6.0]
view    (6) = [-7.0 7.0]
view    (7) = [-8.0 8.0]
view    (8) = [-9.0 9.0]
view    (9) = [-10.0 10.0]
Modify View  : KokkosView_float64_HostSpace_LayoutRight_2 (ndim=2, shape=[10, 2])
view    (0) = [-2.0 2.0]
view    (1) = [-4.0 4.0]
view    (2) = [-6.0 6.0]
view    (3) = [-8.0 8.0]
view    (4) = [-10.0 10.0]
view    (5) = [-12.0 12.0]
view    (6) = [-14.0 14.0]
view    (7) = [-16.0 16.0]
view    (8) = [-18.0 18.0]
view    (9) = [-20.0 20.0]
Numpy Array  : ndarray (ndim=2, shape=(10, 2))
arr     (0) = [-2.0 2.0]
arr     (1) = [-4.0 4.0]
arr     (2) = [-6.0 6.0]
arr     (3) = [-8.0 8.0]
arr     (4) = [-10.0 10.0]
arr     (5) = [-12.0 12.0]
arr     (6) = [-14.0 14.0]
arr     (7) = [-16.0 16.0]
arr     (8) = [-18.0 18.0]
arr     (9) = [-20.0 20.0]
```
