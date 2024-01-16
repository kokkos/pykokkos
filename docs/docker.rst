
Using Docker
============

You can use the PyKokkos Docker image to develop PyKokkos itself, as
well as develop and run applications.  We recommend using the ``pk``
script for interacting with the image and containers.

To run an application in a container, you can execute the following
command:

.. code-block:: bash

   ./pk pk_example examples/kokkos-tutorials/workload/01.py

The command above will pull the image from the Docker Hub, run a
container, include this repository as a volume, and run the example
application from the given path.

If you would like to run another example application, you can simply
change the path (the last argument in the command above).

Note that code you are running should be in the PyKokkos repository.
If you would like to run from another directory you will need to
include the directory as a volume; take a look at the ``pk`` script in
that case.

Design Decision
---------------

At the moment, we decided to include the PyKokkos repository as a
volume when starting a container, which enables the development
workflow. Namely, the ``pk`` script will include the current local
version of this repository, which means that any local modifications
(e.g., a change in ``parallel_dispatch.py``) will be used in the
subsequent runs of the ``pk`` script. In the future, we might separate
user and development workflows.

Limitations
-----------

One, as described above, you would need to modify the ``pk`` script if
you are running code that is not part of this repository.

Two, if your code requires dependencies (e.g., python libraries not
already included in the image), you would need to install it
(temporarily) in the container or build your own image.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
