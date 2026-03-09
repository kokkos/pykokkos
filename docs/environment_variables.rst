
Environment Variables
=====================

PyKokkos behavior can be controlled through the following environment variables:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Variable
     - Value
     - Description
   * - ``PK_FORMAT``
     - any (presence triggers)
     - Runs ``clang-format`` on intermediate C++ sources.
   * - ``PK_FUSION``
     - ``"naive"``
     - Enables automatic kernel fusion. See :doc:`fusion`.
   * - ``PK_KOKKOS_LIB_PATH``
     - path to directory
     - Overrides search path for the compiled pykokkos-base
       ``lib/`` or ``lib64/`` directory.
   * - ``PK_KOKKOS_INTERFACE``
     - Kokkos version string
     - Selects a specific version of the Kokkos interface to use when
       multiple versions are available.

Usage Examples
--------------

Enable C++ code formatting for generated kernels:

.. code-block:: bash

   export PK_FORMAT=1

Enable naive kernel fusion:

.. code-block:: bash

   export PK_FUSION="naive"

Specify a custom Kokkos library path:

.. code-block:: bash

   export PK_KOKKOS_LIB_PATH="/path/to/pykokkos-base/lib"
