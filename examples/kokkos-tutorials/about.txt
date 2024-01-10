This directory contains examples translated from the kokkos-tutorials
repository: https://github.com/kokkos/kokkos-tutorials

01: Allocates buffers using malloc instead of View for the sake of
    simplicity since this is the first exercise. This means that
    this exercise cannot run with Cuda unless we use UVM.

02: Uses Views, but assumes that Views are always in HostSpace and
    initializes them on the host. This means that this exercise cannot
    run with Cuda unless we use UVM.

03: First exercise to use mirror Views. Host mirrors are initialized
    on the host and then copied to the device. Can be run with Cuda
    with or without UVM.

04: Uses mirror Views but explicitly sets the layout to LayoutLeft to
    show the impact of memory layouts on performance, unlike 03 where
    the default layout is set based on the default execution space.
    This results in bad performance on CPUs since the layout is not
    optimal.

team_policy: All previous exercises have a parallel reduce that contains
             an inner sequential for loop. This exercise replaces that inner
             for loop with another parallel reduce.

team_vector_loop: Adds one more dimension to all Views and so adds
                  one more level of parallelism.
