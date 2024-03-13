from .loop_fuse import loop_fuse
from .memory_ops_fuse import memory_ops_fuse
from .restrict_views import (
    add_restrict_views, adjust_kokkos_function_call, adjust_kokkos_function_definition,
    get_restrict_views, get_restrict_ptr_name, index_restrict_view,
)