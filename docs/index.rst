
PyKokkos's Documentation
========================

PyKokkos is a framework for writing high-performance Python code
similar to Numba. In contrast to Numba, PyKokkos kernels are primarily
parallel and are also performance portable, meaning that they can run
efficiently on different hardware (CPUs, NVIDIA GPUs, and AMD GPUs)
with no changes required.

PyKokkos is open source and available on `GitHub
<https://github.com/kokkos/pykokkos>`_.

Below is a quick example that uses PyKokkos.

.. code-block:: python

   import pykokkos as pk

   @pk.workunit
   def hello(i: int):
       pk.printf("Hello, World! from i = %d\n", i)

   def main():
       pk.parallel_for(10, hello)

   main()

Kokkos
------

In the background, PyKokkos translates kernels designated by the user
into C++ `Kokkos <https://github.com/kokkos/kokkos>`_ and
automatically generates language bindings to run the generated code.

.. note::

   Knowing Kokkos is not necessary for using PyKokkos.

On each page of this documentation, we will have a section `Kokkos` to
discuss similarities between Kokkos and PyKokkos. We believe that the
`Kokkos` section will be primarily of interest to those already
familiar with the Kokkos framework.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   docker
   concepts
   patterns
   policies
   work-units

..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
