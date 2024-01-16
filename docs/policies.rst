
Policies
========

Policy is the second key concept in PyKokkos (:doc:`concepts`).  Each
pattern (:doc:`patterns`) accepts a policy as an argument. A policy
specifies the way computation is executed (place and number of units
of work). In this document, we will use the ``parallel_for`` pattern
for illustrations, but similar reasoning applies to other patterns.

RangePolicy
-----------

This is the simplest policy, which specifies unique ids for units of
work as a 1-D range of values.  We can create an instance of
``RangePolicy`` using the following function from the library:

.. code-block:: python

   pk.RangePolicy([ExecutionPlace], begin_value, end_value)

(We discuss ``ExecutionPlace`` on a separate page.)

We can use an range policy like so:

.. code-block:: python

   parallel_for("work", pk.RangePolicy(0, N), work, kwargs_for_work)

This is equivalent to a simple case we have seen on other pages of
this documentation:

.. code-block:: python

   parallel_for("work", N, work, kwargs_for_work)

Below is a complete example:

.. code-block:: python

    import pykokkos as pk
    import numpy as np
    
    @pk.workunit
    def work(wid, a):
        a[wid] += 1
    
    def main():
        N = 10
        a = np.random.randint(100, size=(N))
        print(a)
    
        pk.parallel_for("work", pk.RangePolicy(0, N), work, a=a)
        # OR
        # pk.parallel_for("work", N, work, a=a)
        print(a)
    
    main()

An example output is shown below:

.. code-block:: bash

   [68 30 75 59  0 25 54 80 36 62]
   [69 31 76 60  1 26 55 81 37 63]

.. toctree::
   :maxdepth: 2
   :caption: Contents:
