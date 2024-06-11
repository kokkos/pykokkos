
Installation
============

We recommend using Docker for running applications and developing (see
:ref:`Using Docker<using_docker>`, but detailed installation
instructions are available (see :ref:`Native
Installation<native_installation>`).

.. _using_docker:

Using Docker
------------

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
^^^^^^^^^^^^^^^

At the moment, we decided to include the PyKokkos repository as a
volume when starting a container, which enables the development
workflow. Namely, the ``pk`` script will include the current local
version of this repository, which means that any local modifications
(e.g., a change in ``parallel_dispatch.py``) will be used in the
subsequent runs of the ``pk`` script. In the future, we might separate
user and development workflows.

Limitations
^^^^^^^^^^^

One, as described above, you would need to modify the ``pk`` script if
you are running code that is not part of this repository.

Two, if your code requires dependencies (e.g., python libraries not
already included in the image), you would need to install it
(temporarily) in the container or build your own image.

.. _native_installation:

Native Installation
-------------------

Clone `pykokkos-base <https://github.com/kokkos/pykokkos-base>`_ and
create a conda environment:

.. code-block:: bash

   git clone https://github.com/kokkos/pykokkos-base.git
   cd pykokkos-base/
   conda create --name pyk --file requirements.txt python=3.11
   conda activate pyk

Once the necessary packages have been downloaded and installed,
install ``pykokkos-base`` with CUDA and OpenMP enabled:

.. code-block:: bash

   python setup.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_CUDA=ON -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON

Other ``pykokkos-base`` configuration and installation options can be
found in that project's `README
<https://github.com/kokkos/pykokkos-base/blob/main/README.md>`_.  Note
that this step will compile a large number of bindings which can take
a while to complete. Please open an issue if you run into any problems
with ``pykokkos-base``.

Once ``pykokkos-base`` has been installed, clone ``pykokkos`` and
install its requirements:

.. code-block:: bash

   cd ..
   git clone https://github.com/kokkos/pykokkos.git
   cd pykokkos/
   conda install -c conda-forge pybind11 cupy patchelf
   pip install --user -e .

Note that ``cupy`` is only required if CUDA is enabled in
pykokkos-base.  In some cases, this might result in a ``cupy`` import
error inside ``pykokkos`` similar to the following:

.. code-block::

   ImportError:
   ================================================================
   Failed to import CuPy.

   Original error:
   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /PATH/TO/ENV/lib/python3.11/site-packages/cupy/_core/core.cpython-311-x86_64-linux-gnu.so)

This is due to a mismatch in ``libstdc++.so`` versions between the
system library which ``pykokkos-base`` depends on and the library in
the conda environment which ``cupy`` depends on. This can be solved by
setting the ``LD_PRELOAD`` environment variable to force loading of
the correct library like so:

.. code-block:: bash

   export LD_PRELOAD=/PATH/TO/ENV/lib/libstdc++.so.6

To verify that ``pykokkos`` has been installed correctly, install
``pytest`` and run the tests:

.. code-block:: bash

   conda install pytest
   python runtests.py

.. note::

  Please open an issue for help with installation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
