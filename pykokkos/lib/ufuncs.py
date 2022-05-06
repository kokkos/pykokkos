import pykokkos as pk

# NOTE: it might be slick if we could encapsulate
# the workload code inside of the ufunc definitions,
# but we currently require global scoping


@pk.workload
class ReciprocalView1D:
    def __init__(self, view: pk.View1D[pk.double]):
        # TODO: why doesn't view have a "size"
        # attribute? it is basically the shape product
        # in higher dims
        self.threads: int = view.shape[0] # type: ignore
        self.view: pk.View1D[pk.double] = view

    @pk.main
    def run(self) -> None:
        pk.parallel_for(self.threads, self.pfor)

    @pk.workunit
    def pfor(self, tid: int) -> None:
        self.view[tid] = 1 / self.view[tid] # type: ignore


# TODO: how are we going to "ufunc dispatch" when the view
# has a different type/precision? i.e., something other than
# pk.double?

def reciprocal(view: pk.View1D[pk.double]):
    """
    Return the reciprocal of the argument, element-wise.

    Parameters
    ----------
    view : pk.View1D
           Input view.

    Returns
    -------
    y : pk.View1D
        Output view.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    """
    workload_instance = ReciprocalView1D(view=view)
    # NOTE: should our ufuncs have an option to allow
    # CUDA vs. OpenMP execution directly in their call
    # signature?
    pk.execute(pk.ExecutionSpace.Default, workload_instance)
    # TODO: how should we design this?
    # it seems a bit problematic to both mutate the data
    # underneath a view in-place and return a reference to the view?
    # returning a new view/"array" would be more similar to
    # NumPy?
    return workload_instance.view
    
