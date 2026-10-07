
N-dimensional Arrays
====================

Two common types of arguments given to workunits (:doc:`workunits`)
are those of primitive types and n-dimensional arrays. PyKokkos nicely
interoperates with ``numpy`` and ``cupy`` arrays, as well as
introduces its own abstraction (``View``) used mostly for research
exploration. We describe the use of all three types of ndarrays on
this page.

The motivation to focus on supporting ``numpy`` and ``cupy`` is that
both libraries are widely used and many applications already have code
that relies on them. The idea is that users can benefit from PyKokkos
kernels without having a need to modify their applications.

Interoperability with NumPy
---------------------------

``numpy`` arrays can be directly passed as arguments to workunits
(:doc:`workunits`). As any other argument to a workunit, the arrays
are passed as keyword arguments when executing a pattern
(:doc:`patterns`).

The example below shows a way to create a ``numpy`` array and add
value 1 to each element of the array using ``parallel_for``.

.. code-block:: python

   import numpy as np
   import pykokkos as pk
   
   @pk.workunit
   def work(wid, a):
       a[wid] = a[wid] + 1
   
   def main():
       N = 10
       a = np.ones(N)
       # ... do anything with a using numpy
       pk.parallel_for("work", 10, work, a=a)
       print(a)
   
   main()

Which will the following result:

.. code-block::

   [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]

As ``numpy`` arrays are in main memory, using execution space other
than ``OpenMP`` (which is the default) would result in an error. In
other words, arrays are not explicitly converted or transferred to a
device. It is on the user to perform this step manually. For example,
a user can convert numpy array to cupy array and then invoke
processing on Cuda (``pk.Cuda``).

Interoperability with CuPy
--------------------------

``cupy`` arrays can be directly passed as arguments to workunits
(:doc:`workunits`). As any other argument to a workunit, the arrays
are passed as keyword arguments when executing a pattern
(:doc:`patterns`).

The example below shows a way to create a ``cupy`` array and add value
1 to each element of the array using ``parallel_for``.

.. code-block:: python

   import cupy as cp
   import pykokkos as pk
   
   @pk.workunit
   def work(wid, a):
       a[wid] = a[wid] + 1
   
   def main():
       N = 10
       a = cp.ones(N)
       pk.set_default_space(pk.Cuda)
       pk.parallel_for("work", 10, work, a=a)
       print(a)
   
   main()

As ``cupy`` arrays are allocated in Cuda memory, using execution space
other than ``pk.Cuda`` would result in an error. In other words,
arrays are not explicitly converted or transferred to host.  It is on
the user to perform this step manually.

Native Views
------------

.. note::

   Views are primarily used for research exploration. It is
   recommended to use Views only when targeting execution spaces other
   than Cuda and OpenMP.

Native n-dimensional arrays are available via an abstraction called
``View``. In many ways, the ``View`` API is similar to those available
for ``numpy`` and ``cupy`` arrays. The following snippet shows how a
View can be created and some of the basic operations it supports:

.. code-block:: python

   v = pk.View([10], int) # create a 1D integer view of size 10
   v.fill(0) # initialize v with zeros
   v[0] = 10
   print(v) # prints the contents of the view

Views and other primitive types can be passed to workunits
normally. The following code snippet shows a workunit that adds a
scalar to all elements of a view.

.. code-block:: python

   import pykokkos as pk

   @pk.workunit
   def add(i: int, v: pk.View1D[int], x: int):
       v[i] += x
   
   if __name__ == "__main__":
       n = 10
       v = pk.View([n], int)
       v.fill(0)
   
       pk.parallel_for(n, add, v=v, x=1)

Recall (:doc:`workunits`) that type annotations are not required.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
