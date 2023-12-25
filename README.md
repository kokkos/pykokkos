# PyKokkos

PyKokkos is a framework for writing performance portable kernels in
Python. At a high-level, PyKokkos translates type-annotated Python
code into C++ [Kokkos](https://github.com/kokkos/kokkos/) and
automatically generating bindings for the translated C++ code.
PyKokkos also makes use of [Python
bindings](https://github.com/kokkos/kokkos-python) for constructing
Kokkos Views.

## Installation

Clone [pykokkos-base](https://github.com/kokkos/pykokkos-base) and
create a conda environment:

```bash
git clone https://github.com/kokkos/pykokkos-base.git
cd pykokkos-base/
conda create --name pyk --file requirements.txt
conda activate pyk
```

Once the necessary packages have been downloaded and installed,
install `pykokkos-base` with CUDA and OpenMP enabled:

```bash
python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON
```

Other `pykokkos-base` configuration and installation options can be
found in that project's
[README](https://github.com/kokkos/pykokkos-base/blob/main/README.md).
Note that this step will compile a large number of bindings which can
take a while to complete. Please open an issue if you run into any
problems with `pykokkos-base`.

Once `pykokkos-base` has been installed, clone `pykokkos` and install
its requirements:

```bash
cd ..
git clone https://github.com/kokkos/pykokkos.git
cd pykokkos/
conda install -c conda-forge pybind11 cupy patchelf
pip install --user -e .
```

Note that `cupy` is only required if CUDA is enabled in pykokkos-base.
In some cases, this might result in a `cupy` import error inside
`pykokkos` similar to the following

```
ImportError:
================================================================
Failed to import CuPy.

Original error:
  ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /PATH/TO/ENV/lib/python3.11/site-packages/cupy/_core/core.cpython-311-x86_64-linux-gnu.so)
```

This is due to a mismatch in `libstdc++.so` versions between the
system library which `pykokkos-base` depends on and the library in the
conda environment which `cupy` depends on. This can be solved by
setting the `LD_PRELOAD` environment variable to force loading of the
correct library like so

```bash
export LD_PRELOAD=/PATH/TO/ENV/lib/libstdc++.so.6
```

To verify that `pykokkos` has been installed correctly, install
`pytest` and run the tests:

```bash
conda install pytest
python runtests.py
```

Please open an issue for help with installation.

## Docker

You can use our Docker image to develop pykokkos and run examples.  We
recommend using the `pk` script for interacting with the image.

To run an example in the image, you can execute the following command:

```
./pk pk_example examples/kokkos-tutorials/workload/01.py
```

The command above will pull the image, run a container, include this
repository as a volume, and run the example on the given path.

If you would like to run another example, you simply change the path
(the last argument in the command above).

Note that the example you are running should be in this repository.
If you would like to run from another directory you will need to
include those as a volume; take a look at the `pk` script in that case.

### Design Decision

At the moment, we decided to include this repository as a volume when
starting a container, which enables the development workflow. Namely,
the `pk` script will include the current local version of this
repository, which means that any local modifications (e.g., a change
in `parallel_dispatch.py`) will be used in the subsequent runs of the
`pk` script.  In the future, we might separate a user and development
workflows.

### Limitations

One, as described above, you would need to modify the `pk` script if
you are running examples that are not part of this repository.

Two, if your code requires dependencies (e.g., python libraries not
already included in the image), you would need to build your own image.

## Examples

### Hello World

PyKokkos provides decorators for marking kernel code. The following
code snippet shows a kernel that prints hello world from each thread:

```python
import pykokkos as pk

@pk.workunit
def hello(i: int):
    pk.printf("Hello, World! from i = %d\n", i)
```

Kernels definitions are marked with the `@pk.workunit` decorator. Each
workunit requires an integer argument which represents the thread ID.
This argument has to be type annotated using `int`.

This workunit can be called using the `parallel_for` function:

```python
pk.parallel_for(10, hello)
```

PyKokkos will translate the workunit to C++ and Kokkos and compile it
the first time it is called. Calling the same workunit again will skip
the translation and compilation steps.

### Views

PyKokkos uses Views as its main n-dimensional array data structure. In
Python, Views data behave as NumPy Arrays. The following snippet shows
how a View can be created and some of the basic operations it
supports:

```python
v = pk.View([10], int) # create a 1D integer view of size 10
v.fill(0) # initialize v with zeros
v[0] = 10
print(v) # prints the contents of the view
```

Views and other primitive types can be passed to workunits normally.
The following code snippet shows a workunit that adds a scalar to all
elements of a view.

```python
import pykokkos as pk

@pk.workunit
def add(i: int, v: pk.View1D[int], x: int):
    v[i] += x

if __name__ == "__main__":
    n = 10
    v = pk.View([n], int)
    v.fill(0)

    pk.parallel_for(n, add, v=v, x=1)
```

As with the thread ID, arguments must be type annotated. They can the
be passed via `parallel_for` using keyword arguments.

### Functors

Workunits can also be defined as methods inside a functor. Functors
are Python classes that contain one or workunits as methods. The
following code snippet shows an example of a functor.

```python
@pk.functor
def Functor:
    def __init__(self, v, x):
        self.v: pk.View1D[int] = v
        self.x: int = x

    @pk.workunit
    def add(self, i: int):
        self.v[i] += x

    @pk.workunit
    def print(self, i: int):
        pk.printf("v[%d] = %d\n", i, self.v[i])

if __name__ == "__main__":
    n = 10
    v = pk.View([n], int)
    v.fill(0)

    f = Functor(v, 1)
    pk.parallel_for(n, f.add)
    pk.parallel_for(n, f.print)
```

Workunits defined in functors only include the thread ID argument in
their definition. Instead of arguments, they access Views and other
primitive types as member variables. These member variables must be
defined in the constructor `__init__` with type annotations. This has
the benefit of avoiding repetition of the same type annotations across
multiple non-method workunits.

To call these workunits, the functor class must first be instantiated.
Individual workunits are called using `parallel_for` by passing in the
workunit method as an argument. The member variables will hold the
values the functor instance contains at the time `parallel_for` is
called.

### Other Examples

The following table shows a list of other PyKokkos examples, as well
as their corresponding C++ Kokkos implementations:

| Example  | | |
| -------- | - | - |
| parallel_reduce | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/kokkos-tutorials/functor/02.py) | [Kokkos](https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/02/Solution/exercise_2_solution.cpp) |
| Cuda | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/kokkos-tutorials/functor/04.py) | [Kokkos](https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/04/Solution/exercise_4_solution.cpp) |
| team_policy | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/kokkos-tutorials/functor/team_policy.py) | [Kokkos](https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/team_policy/Solution/team_policy_solution.cpp) |
| team_vector_loop | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/kokkos-tutorials/functor/team_vector_loop.py) | [Kokkos](https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/team_vector_loop/Solution/team_vector_loop_solution.cpp) |
| subview | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/kokkos-tutorials/functor/subview.py) | [Kokkos](https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/subview/Solution/exercise_subview_solution.cpp) |
| mdrange | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/kokkos-tutorials/workload/mdrange.py) | [Kokkos](https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/mdrange/Solution/exercise_mdrange_solution.cpp) |
| nstream | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/ParRes/workload/nstream.py) | [Kokkos](https://github.com/ParRes/Kernels/blob/default/Cxx11/nstream-kokkos.cc) |
| stencil | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/ParRes/workload/stencil.py) | [Kokkos](https://github.com/ParRes/Kernels/blob/default/Cxx11/stencil-kokkos.cc) |
| transpose | [PyKokkos](https://github.com/kokkos/pykokkos/blob/main/examples/ParRes/workload/transpose.py) | [Kokkos](https://github.com/ParRes/Kernels/blob/default/Cxx11/transpose-kokkos.cc) |
| ExaMiniMD | [PyKokkos](https://github.com/kokkos/pykokkos/tree/main/examples/ExaMiniMD) | [Kokkos](https://github.com/ECP-copa/ExaMiniMD) |

## Citation

If you have used PyKokkos in a research project, please cite this
research paper:
```
@inproceedings{AlAwarETAL21PyKokkos,
  author = {Al Awar, Nader and Zhu, Steven and Biros, George and Gligoric, Milos},
  title = {A Performance Portability Framework for Python},
  booktitle = {International Conference on Supercomputing},
  pages = {467-478},
  year = {2021},
}
```

### Acknowledgments

This project is partially funded by the U.S. Department of Energy,
National Nuclear Security Administration under Award Number
DE-NA0003969 (PSAAP III).
