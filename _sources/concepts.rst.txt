
Concepts
========

On this page, we introduce the key PyKokkos concepts.  We will use the
following example to illustrate the key concepts.

.. code-block:: python

   import pykokkos as pk

   @pk.workunit
   def hello(i: int):
       pk.printf("Hello, World! from i = %d\n", i)
    
   def main():
       pk.parallel_for(10, hello)
    
   main()

* **Pattern** specifies the structure of computation (e.g.,
  ``parallel_for``)
* **Policy** specifies the way computations are executed (e.g., ``10``
  parallel units of work)
* **Work Unit** is a function that performs each unit of work (e.g.,
  ``hello``)

Execution of a Pattern creates `work`. In the example above, there
will be 10 `units of work`, such that each unit executes the ``hello``
function. `Index`, which is the first argument of the Work Unit,
identifies a unique unit of work. PyKokkos maps work to execution
resources.

.. note::

   Ordering of execution of units of work is not guaranteed by the
   runtime.

The three key concepts are closely intertwined.

Kokkos
------

.. note::

   This section might be of interest only to those familiar with Kokkos.

Unlike in Kokkos, we do not use the term "Computational Body" (but
rather "Work Unit"), because code for any work is always in a separate
function. Due to the limitation of Python lambdas, there are no
multiple ways like in `Kokkos
<https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html#should-i-use-a-functor-or-a-lambda>`_
to write computational bodies.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
