
Policies
========

Policy is the second key concept (out of three) in PyKokkos
(:doc:`concepts`).  Each pattern (:doc:`patterns`) accepts a policy as
an argument. A policy specifies the way computation is executed and
provides:

* Number of units of work to run by providing a set of unique ids
* Space in which the execution should happen (e.g., Cuda)

PyKokkos currently supports the following policies:

* :ref:`RangePolicy<range_policy>`
* :ref:`TeamPolicy<team_policy>`

In this document, we will use the ``parallel_for`` pattern for
illustrations in our examples, but similar reasoning applies to other
patterns.

.. _range_policy:

RangePolicy
-----------

This is the simplest policy, which specifies unique ids for units of
work as a 1-D range of values. One can create an instance of
``RangePolicy`` using the following function from the library:

.. code-block:: python

   pk.RangePolicy([ExecutionSpace], begin_value, end_value)

``ExecutionSpace`` is optional and discussed in detail in
:ref:`ExecutionSpace<execution_space>`. If ``ExecutionSpace`` is not
provided, the default space will be used.

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

.. code-block::

   [68 30 75 59  0 25 54 80 36 62]
   [69 31 76 60  1 26 55 81 37 63]

.. _team_policy:

Team Policy
-----------

``TeamPolicy`` is used to implemente hierarchical parallelism.
Threads are grouped into teams, and there can be an arbitrary many
teams. Each team has a number of threads (the team size).  All threads
in a team are guaranteed to run concurrently.  One can create an
instance of ``TeamPolicy`` using the following function from the
library:

.. code-block:: python

   pk.TeamPolicy([ExecutionSpace], league_size, team_size)

``ExecutionSpace`` is optional and discussed in detail in
:ref:`ExecutionSpace<execution_space>`. If ``ExecutionSpace`` is not
provided, the default space will be used.

Below is an example of adding one to each element of an array, which
uses ``TeamPolicy`` to organize the computation.

.. code-block:: python

   import numpy as np
   import pykokkos as pk
   
   @pk.workunit
   def work(team_member, view):
       j: int = team_member.league_rank()
       k: int = team_member.team_size()
       
       def inner(i: int):
           view[j * k + i] = view[j * k + i] + 1
   
       pk.parallel_for(pk.TeamThreadRange(team_member, k), inner)
   
   def main():
       pk.set_default_space(pk.OpenMP)
       a = np.zeros(100)
       pk.parallel_for("work", pk.TeamPolicy(50, 2), work, view=a)
       print(a)
   
   main()

.. note::

   Those familiar with Cuda might want to think about league_rank as
   block id, team_size as block size, and team_rank as a thread id.

.. _execution_space:

Execution Space
---------------

``ExecutionSpace`` specifies the place where units of work will be
executed. The following are valid values:

* ``pk.OpenMP`` - execution on the host using OpenMP
* ``pk.Cuda`` - execution on a Cuda device
* ``pk.HIP`` - execution on an AMD GPU

If the execution space is not provided in a policy (at the time of a
pattern execution), then the default execution space will be used. The
default execution space is set to ``pk.OpenMP`` when an application
starts. The default execution space can be changed at any point use
the ``pk.set_default_space()`` function.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
