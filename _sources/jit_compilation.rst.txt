JIT Compilation
===============

When a PyKokkos workunit is first invoked, it is translated to C++ and
compiled automatically. Subsequent
calls reuse the compiled binary, so the cost is paid only once per
unique workunit.

Compilation Pipeline
--------------------

The following steps occur the first time a workunit is executed:

1. **Translation** — The Python AST of the ``@pk.workunit`` is
   translated into a C++ Kokkos functor along with pybind11 language
   bindings.
2. **Formatting** *(optional)* — If ``clang-format`` is available and
   the environment variable ``PK_FORMAT=1`` is set, the generated C++
   source files are formatted before compilation.
3. **Compilation** — CMake configures and builds the translated source
   into a shared library that Python loads at runtime.

Output Directory
----------------

All generated C++ source files and compiled artifacts are written under
``BASE_DIR``, defined in ``pykokkos/core/module_setup.py`` as
``.pykokkos/``. The directory structure is::

   .pykokkos/
   └── <main_file>/
       └── <entity_name>/
           └── [types_<hash>/]
               └── AST_<hash>/
                   └── <ExecutionSpace>/
                       ├── functor.hpp
                       ├── functor_cast.hpp
                       ├── bindings.cpp
                       ├── CMakeLists.txt
                       └── build/

The ``AST_<hash>`` directory name encodes a structural hash of the
compiled workunit. If the workunit body changes, the hash changes and
PyKokkos triggers recompilation. Object files from the previous build
are carried over where possible to keep incremental compile times short.

Environment Variables
---------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Effect
   * - ``PK_FORMAT=1``
     - Format generated C++ source with ``clang-format`` before compilation (requires ``clang-format`` to be on ``PATH``).
