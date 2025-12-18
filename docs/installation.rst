
Installation
============

We provide a Docker container for running applications and developing
on Linux x86 systems with the Serial and OpenMP host execution space
(see :ref:`Using Docker<using_docker>`).
We provide detailed install instructions for all other
operating systems and executions spaces
(see :ref:`Native Installation<native_installation>`).

.. _using_docker:

Using Docker
------------

Currently deprecated. Work in progress.
.. You can use the PyKokkos Docker image to develop PyKokkos itself, as
.. well as develop and run applications.  We recommend using the ``pk``
.. script for interacting with the image and containers.

.. To run an application in a container, you can execute the following
.. command:

.. .. code-block:: bash

..    ./pk pk_example examples/kokkos-tutorials/workload/01.py

.. The command above will pull the image from the Docker Hub, run a
.. container, include this repository as a volume, and run the example
.. application from the given path.

.. If you would like to run another example application, you can simply
.. change the path (the last argument in the command above).

.. Note that code you are running should be in the PyKokkos repository.
.. If you would like to run from another directory you will need to
.. include the directory as a volume; take a look at the ``pk`` script in
.. that case.

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

Clone `pykokkos <https://github.com/kokkos/pykokkos>`_ and
create a conda environment:

.. code-block:: bash

   git clone https://github.com/kokkos/pykokkos.git
   cd pykokkos/
   conda create --name pyk --file base/requirements.txt python=3.13
   conda activate pyk

Once the necessary packages have been downloaded and installed,
install ``base`` with required CMake flags (example performs an install with  OpenMP and CUDA enabled):

.. code-block:: bash
   PYKOKKOS_BASE_SETUP_ARGS="\
    -DKokkos_ENABLE_THREADS=OFF \ # trigger pthreads execution space
    -DKokkos_ENABLE_OPENMP=ON \   # trigger openmp execution space
    -DENABLE_CUDA=ON \            # trigger cuda execution space
    -DENABLE_HIP=OFF \            # trigger hip execution space
    -DENABLE_LAYOUTS=ON \         # trigger layout left/right ordering
    -DENABLE_MEMORY_TRAITS=OFF \  # trigger memory space concept
    -DENABLE_VIEW_RANKS=3 \       # maximum number of view ranks enabled
    " \
   python install_base.py install
See `Kokkos CMake Options <https://kokkos.org/kokkos-kernels/docs/cmake-keywords.html>`_ for a complete list of CMake flags.
Other ``pykokkos`` configuration and installation options can be
found in that project's `README
<https://github.com/kokkos/pykokkos/blob/main/base/README.md>`_.  Note
that this step will compile a large number of bindings which can take
a while to complete. 

.. note::
        Please open an issue
        or reach out in the `Kokkos slack <https://kokkos.org/community/chat/>`_ 
        **#pykokkos** channel
        if you run into any problems
        with ``base``.

Once ``base`` has been installed, you can install ``pykokkos`` itself:

.. code-block:: bash
   conda install -c conda-forge pybind11 cupy patchelf
   pip install --user -e .

Note that ``cupy`` is only required if CUDA is enabled in
base.  In some cases, this might result in a ``cupy`` import
error inside ``pykokkos`` similar to the following:

.. code-block::

   ImportError:
   ================================================================
   Failed to import CuPy.

   Original error:
   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /PATH/TO/ENV/lib/python3.11/site-packages/cupy/_core/core.cpython-311-x86_64-linux-gnu.so)

This is due to a mismatch in ``libstdc++.so`` versions between the
system library which ``base`` depends on and the library in
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
        Please open an issue
        or reach out in the `Kokkos slack <https://kokkos.org/community/chat/>`_ 
        **#pykokkos** channel
        if you run into any problems
        with ``pykokkos``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
