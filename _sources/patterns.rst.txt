
Patterns
========

Pattern is the first key concept (out of three) in PyKokkos
(:doc:`concepts`).  Patterns specify the structure of computation.
There are three key patterns available in PyKokkos:

* ``parallel_for``, which is also known as a ``map`` operation in
  other frameworks/languages
* ``parallel_reduce``, which is also known as a ``fold`` operation in
  other frameworks/languages
* ``parallel_scan``, which implements a prefix scan

Parallel for
------------

The most commonly used pattern is ``parallel_for``.  The pattern is
available as a function in the ``pykokkos`` library and has the
following signature:

.. code-block:: python

   parallel_for([label], policy, workunit, [keyword arguments])

* **label** is an optional string value helpful for debugging and
  profiling
* **policy** specifies the way computations are executed (execution
  place and number of work units to run in parallel). In its simplest
  form, policy is an integer value that specifies a range of
  values. More details about policies is provided in a separate page
  (:doc:`policies`)
* **workunit** is the name of the ``@pk.workunit`` function that
  performs one unit of work
* **arguments** are keyword arguments passed to the workunit

Based on the policy, the ``parallel_for`` will execute a number of
work units in parallel. Each work unit is executed independently and
there are no guarantees about the execution order. At the same time,
any number of work units might be running in parallel or they might be
executed sequentially if the runtime determines that such an execution
would be beneficial for the overall performance.

Below is an example to illustrate the ``parallel_for`` pattern.

.. code-block:: python

   import pykokkos as pk
   
   @pk.workunit
   def hello(i: int):
       pk.printf("Hello, World! from i = %d\n", i)
   
   def main():
       pk.parallel_for(10, hello)
   
   main()

In this example, the policy is simply an integer value (``10``) that
specifies a range (``0..9``) of unique ids for work units to be
spawned (one work unit for one id).  Here is the output for the
example:

.. code-block:: bash

   Hello, World! from i = 0
   Hello, World! from i = 8
   Hello, World! from i = 4
   Hello, World! from i = 1
   Hello, World! from i = 9
   Hello, World! from i = 5
   Hello, World! from i = 6
   Hello, World! from i = 2
   Hello, World! from i = 3
   Hello, World! from i = 7

Parallel reduce
---------------

The pattern ``parallel_reduce`` implements a reduction. This pattern
is similar in many ways to ``parallel_for`` except that each work unit
produces a value, and all the values are eventually accumulated into a
single value (known as an accumulator).  This pattern is available as
a function in the ``pykokkos`` library and has the following
signature:

.. code-block:: python

   parallel_reduce([label], policy, workunit, [keyword arguments])

* **label** is an optional string value helpful for debugging and
  profiling
* **policy** specifies the way computations are executed (execution
  place and number of workunits to run in parallel). In its simplest
  form, policy is an integer value that specifies a range of
  values. More details about policies is provided in a separate page
  (:doc:`policies`)
* **workunit** is the name of the ``@pk.workunit`` function that
  performs one unit of work
* **arguments** are keyword arguments passed to the workunit

Based on the policy, ``parallel_reduce`` runs a number of work units.
Each work unit receives `two` arguments in addition to the specified
keyword arguments: (1) unique id of the work unit, and (2) an
accumulator.

Below is an example to illustrate the ``parallel_reduce`` pattern:

.. code-block:: python

   import pykokkos as pk
   import numpy as np
   
   @pk.workunit
   def work(wid, acc, a):
       acc += a[wid]
   
   def main():
       N = 10
       a = np.random.randint(100, size=(N))
       print(a)
       total = pk.parallel_reduce("work", N, work, a=a)
       print(total)
   
   main()

In the example, we run ``N`` (which is set to ``10``) work units to
compute the sum of all elements in a numpy array (``a``).  Note that
the first two arguments to the workunit (``wid`` which is a unique
identifier of a work unit, and ``acc`` which is an accumulator) are
provided at runtime by the framework.

Parallel scan
-------------

The pattern ``parallel_scan`` implements a prefix scan.  This pattern
is very much like ``parallel_reduce``, but it also stores all
intermediate results.  The pattern is available as a function in the
``pykokkos`` library and has the following signature:

.. code-block:: python

   parallel_scan([label], policy, workunit, [keyword arguments])

* **label** is an optional string value helpful for debugging and
  profiling
* **policy** specifies the way computations are executed (execution
  place and number of workunits to run in parallel). In its simplest
  form, policy is an integer value that specifies a range of
  values. More details about policies is provided in a separate page
  (:doc:`policies`)
* **workunit** is the name of the ``@pk.workunit`` function that
  performs one unit of work
* **arguments** are keyword arguments passed to the workunit

As before, based on the policy, ``parallel_scan`` runs a number of
units of work. Each unit of work receives `three` arguments in
addition to the given keyword arguments: (1) unique id of the unit of
work, (2) an accumulator, and (3) a boolean flag to indicate if the
scan for the current unit of work is complete.

Below is an example to illustrate the ``parallel_scan`` pattern:

.. code-block:: python

   import pykokkos as pk
   import numpy as np
   
   @pk.workunit
   def work(wid, acc, final, a):
       acc += a[wid]
       if final:
           a[wid] = acc
   
   def main():
       N = 10
       a = np.random.randint(100, size=(N))
       print(a)
   
       pk.parallel_scan("work", N, work, a=a)
       print(a)
   
   main()

The output for the example above for a single run is:

.. code-block:: bash

   [59 60 48 65 41 22 64 59 91 24]
   [ 59 119 167 232 273 295 359 418 509 533]

.. toctree::
   :maxdepth: 2
   :caption: Contents:
