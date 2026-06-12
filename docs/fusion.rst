
Kernel Fusion
=============

PyKokkos includes an automatic kernel fusion feature that can significantly
improve performance when executing many kernel calls. Kernel fusion dynamically
traces and combines multiple kernel launches into a single fused kernel,
reducing launch overhead and improving execution efficiency through better data
reuse and improved compiler optimizations.

Fusion process uses lazy evaluation to record kernel calls in traces and fuses
them when the result is requested by the application. This happens
automatically and transparently without requiring any changes to your
PyKokkos code.

Enabling Fusion
---------------

Kernel fusion is controlled by the ``PK_FUSION`` environment variable.
To enable fusion, set this variable before running your PyKokkos application:

.. code-block:: bash

   export PK_FUSION="naive"

To disable fusion (default behavior):

.. code-block:: bash

   unset PK_FUSION

Performance Example
-------------------

The following example demonstrates the performance benefit of kernel fusion
when executing many small kernel calls in a loop:

.. code-block:: python

   import cupy as cp
   import pykokkos as pk

   @pk.workunit
   def work(wid, a): 
       a[wid] = a[wid] + 1 

   def main():
       B = 100000
       N = 10
       a = cp.ones((B, N)) 
       pk.set_default_space(pk.Cuda)
       for batch in range(B):
           pk.parallel_for("work", 10, work, a=a[batch])
       print(a)

   main()

Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^

Running the above example with and without fusion shows significant speedup:

Machine Specification
^^^^^^^^^^^^^^^^^^^^

The following hardware was used for the performance measurements:

.. code-block:: text

    CPU:  Intel(R) Xeon(R) w5-3433, 32 (16 cores, 2 threads per core)
    GPU:  NVIDIA RTX 5000 Ada Generation 32 GB (2x) 
    CUDA: 12.4
    OS:   Ubuntu 24.04 (Linux)

**Without fusion** (``unset PK_FUSION``):

.. code-block:: text

   real    0m27.213s
   user    0m35.134s
   sys     0m0.990s

**With fusion** (``export PK_FUSION="naive"``):

.. code-block:: text

   real    0m14.840s
   user    0m22.729s
   sys     0m1.136s

In this example, kernel fusion provides approximately **1.8x speedup** by fusing
100,000 kernel calls into fewer fused kernels, reducing kernel launch overhead
and enabling better data reuse and compiler optimizations.

When to Use Fusion
------------------

Kernel fusion is most beneficial when:

* Executing many kernel calls consecutively
* Kernel launch overhead dominates execution time
* Multiple kernels operate on shared data

.. note::

   Fusion is currently most effective on GPU execution spaces where
   kernel launch overhead is more significant. Fusion can achieve
   speedups on NVIDIA and AMD GPUs as well as Intel and AMD CPUs.

For more details on the kernel fusion implementation, see the fuser
paper: `Dynamically Fusing Python HPC Kernels
<https://users.ece.utexas.edu/~gligoric/papers/AlAwarETAL25Fuser.pdf>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
