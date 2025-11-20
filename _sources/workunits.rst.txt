
Work Units
==========

Work units are the third key concept (out of three) in PyKokkos
(:doc:`concepts`). PyKokkos work units are specified as one of the
arguments to patterns (:doc:`patterns`). Each work unit is a Python
function decorated with ``@pk.workunit``. At the time a pattern is
executed (e.g., ``parallel_for``), the work unit function is invoked
for each unique id depending on the policy used.

.. note::

   PyKokkos work units are written in a subset of Python. These
   functions are translated to C++ and compiled to various target
   architectures. Thus, work units should be lightweight and free from
   use of various Python libraries.

Signatures
----------

Depending on the pattern used to invoke a work unit (e.g.,
``parallel_for``), the signature of the work unit function differs.

In case of ``parallel_for``, the work unit function is required to
accept, as the first argument, unique id for one unit of work.

.. code-block:: python

   @pk.workunit
   def work(wid, [keyword arguments])

In case of ``parallel_reduce``, the work unit function is required to
accept two arguments: (1) unique id for one unit of work, and (2) an
accumulator.

.. code-block:: python

   @pk.workunit
   def work(wid, acc, [keyword arguments])

Finally, in case of ``parallel_scan``, the work unit function is
requires to accept three arguments: (1) unique id for one unit of
work, (2) an accumulator, and (3) a boolean flag to indicate if the
scan for that unique id is complete.

.. code-block:: python

   @pk.workunit
   def work(wid, acc, final, [keyword arguments])

Keyword Arguments
-----------------

Each work unit function can accept an arbitrary number of arguments
after those required in the signature.  At the time of a pattern
execution, the arguments have to be given as keyword arguments.

Type Annotations
----------------

Type annotations for the arguments are optional, but recommended. We
use type annotations heavily in the examples in the PyKokkos
repository.

.. note::

   Type annotations for variables defined inside a work unit are
   currently required.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
