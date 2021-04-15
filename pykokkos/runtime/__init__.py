from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from pykokkos.core import Runtime

# This module holds a reference to a singleton runtime object
# accessible from anywhere. The runtime object will be initialized
# elsewhere.

class RuntimeSingleton:
    def __init__(self):
        self.runtime: Optional[Runtime] = None


runtime_singleton = RuntimeSingleton()
