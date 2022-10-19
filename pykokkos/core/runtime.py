import importlib.util
import sys
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np

from pykokkos.core.keywords import Keywords
from pykokkos.core.translators import PyKokkosMembers
from pykokkos.core.visitors import visitors_util
from pykokkos.interface import (
    DataType, ExecutionPolicy, ExecutionSpace, MemorySpace,
    RandomPool, RangePolicy, TeamPolicy, View, ViewType,
    is_host_execution_space
)
import pykokkos.kokkos_manager as km

from .compiler import Compiler
from .module_setup import ModuleSetup
from .run_debug import run_workload_debug, run_workunit_debug


class Runtime:
    """
    Executes (and optionally compiles) PyKokkos workloads
    """

    def __init__(self):
        self.compiler: Compiler = Compiler()

        # cache module_setup objects using a workload/workunit and space tuple
        self.module_setups: Dict[Tuple, ModuleSetup] = {}

    def run_workload(self, space: ExecutionSpace, workload: object) -> None:
        """
        Run the workload

        :param space: the execution space of the workload
        :param workload: the workload object
        """

        if self.is_debug(space):
            run_workload_debug(workload)
            return

        module_setup: ModuleSetup = self.get_module_setup(workload, space)
        members: Optional[PyKokkosMembers] = self.compiler.compile_object(module_setup, space, km.is_uvm_enabled())

        if members is None:
            raise RuntimeError("ERROR: members cannot be none")

        self.execute(workload, module_setup, members, space)
        self.run_callbacks(workload, members)

    def precompile_workunit(
        self,
        workunit: Callable[..., None],
        space: ExecutionSpace = "Default"
        ) -> Optional[PyKokkosMembers]:
        """
        precompile the workunit

        :param workunit: the workunit function object
        :param space: the ExecutionSpace for which the bindings are generated
        :returns: the members the functor is containing
        """

        module_setup: ModuleSetup = self.get_module_setup(workunit, space)
        members: Optional[PyKokkosMembers] = self.compiler.compile_object(module_setup, space, km.is_uvm_enabled())
        return members

    def run_workunit(
        self,
        name: Optional[str],
        policy: ExecutionPolicy,
        workunit: Callable[..., None],
        operation: Optional[str] = None,
        initial_value: Union[float, int] = 0,
        **kwargs
    ) -> Optional[Union[float, int]]:
        """
        Run the workunit

        :param name: the name of the kernel
        :param policy: the execution policy of the operation
        :param workunit: the workunit function object
        :param kwargs: the keyword arguments passed to the workunit
        :param operation: the name of the operation "for", "reduce", or "scan"
        :param initial_value: the initial value of the accumulator
        :returns: the result of the operation (None for parallel_for)
        """

        if self.is_debug(policy.space):
            if operation is None:
                raise RuntimeError("ERROR: operation cannot be None for Debug")
            return run_workunit_debug(policy, workunit, operation, initial_value, **kwargs)

        members: Optional[PyKokkosMembers] = self.precompile_workunit(workunit,policy.space)
        if members is None:
            raise RuntimeError("ERROR: members cannot be none")

        module_setup: ModuleSetup = self.get_module_setup(workunit, policy.space)
        return self.execute(workunit, module_setup, members, policy.space, policy=policy, name=name, **kwargs)

    def is_debug(self, space: ExecutionSpace) -> bool:
        """
        Check if the execution space is Debug and account for Default space

        :param space: the execution space
        :returns: True or False
        """

        return space is ExecutionSpace.Debug or (space is ExecutionSpace.Default
                and km.get_default_space() is ExecutionSpace.Debug)

    def execute(
        self,
        entity: Union[object, Callable[..., None]],
        module_setup: ModuleSetup,
        members: PyKokkosMembers,
        space: ExecutionSpace,
        policy: Optional[ExecutionPolicy] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Optional[Union[float, int]]:
        """
        Imports the module containing the bindings and executes the necessary function

        :param entity: the workload or workunit object
        :param module_path: the path to the compiled module
        :param members: a collection of PyKokkos related members
        :param space: the execution space
        :param policy: the execution policy for workunits
        :param name: the name of the kernel
        :param kwargs: the keyword arguments passed to the workunit
        :returns: the result of the operation (None for "for" and workloads)
        """

        module_path: str
        if is_host_execution_space(space) or not km.is_multi_gpu_enabled():
            module_path = module_setup.path
        else:
            device_id: int = km.get_device_id()
            module_path = module_setup.gpu_module_paths[device_id]

        module = self.import_module(module_setup.name, module_path)

        args: Dict[str, Any] = self.get_arguments(entity, members, policy, **kwargs)
        if name is None:
            args["pk_kernel_name"] = ""
        else:
            args["pk_kernel_name"] = name

        is_workunit_or_functor: bool = isinstance(entity, Callable)
        result = self.call_wrapper(entity, members, args, module, is_workunit_or_functor)

        if not is_workunit_or_functor:
            self.retrieve_results(entity, members, args)

        return result

    def import_module(self, module_name: str, module_path: str):
        """
        Import a compiled module

        :param module_name: the name of the compiled module
        :param module_path: the path to the compiled module
        :returns: the imported module
        """

        hashed_name: str = module_name.replace("kernel", f"kernel_{km.get_device_id()}")

        if hashed_name in sys.modules:
            return sys.modules[hashed_name]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[hashed_name] = module
        spec.loader.exec_module(module)

        return module

    def get_arguments(
        self,
        entity: Union[object, Callable[..., None]],
        members: PyKokkosMembers,
        policy: Optional[ExecutionPolicy],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get the arguments for a wrapper function, including fields, views, etc

        :param entity: the workload or workunit object
        :param members: a collection of PyKokkos related members
        :param policy: the execution policy of the operation
        :param kwargs: the keyword arguments passed to a workunit
        """

        args: Dict[str, Any] = {}

        entity_members: Dict[str, type]
        is_workload: bool = not isinstance(entity, Callable)

        if is_workload:
            args.update(self.get_result_arguments(members))
            entity_members = entity.__dict__

        else:
            if policy is None:
                raise RuntimeError("ERROR: execution policy is None")

            args.update(self.get_policy_arguments(policy))
            is_functor: bool = hasattr(entity, "__self__")
            if is_functor:
                functor: object = entity.__self__
                entity_members = functor.__dict__
            else:
                entity_members = kwargs

        args.update(self.get_fields(entity_members))
        args.update(self.get_views(entity_members))
        args.update(self.get_randpool_args(entity_members))

        return args

    def call_wrapper(
        self,
        entity: Union[object, Callable[..., None]],
        members: PyKokkosMembers,
        args: Dict[str, Any],
        module,
        return_wrapper: bool
    ) -> Optional[Union[float, int]]:
        """
        Call the wrapper in the imported module

        :param entity: the workload or workunit object
        :param members: a collection of PyKokkos related members
        :param args: the arguments to be passed to the wrapper
        :param module: the imported module
        :param return_wrapper: whether to return the wrapper (assuming it will be called later)
        """

        is_workunit: bool = isinstance(entity, Callable)
        wrapper: str = "wrapper"
        if is_workunit:
            wrapper += f"_{entity.__name__}"

        if members.has_real:
            precision: str = self.get_precision(members, args)
            wrapper += f"_{precision}"

        func = getattr(module, wrapper)

        if return_wrapper:
            return func, args

        return func(**args)

    def get_precision(self, members: PyKokkosMembers, args: Dict[str, Any]) -> str:
        """
        Get the precision of the entity by comparing the data types of the
        views in members with the views of the entity. If the real precision
        is not consistent, throw an error.

        :param members: a collection of PyKokkos related members
        :param args: the arguments to be passed to the wrapper
        """

        precision: str = ""
        view: str = ""

        for n in members.real_dtype_views:
            name: str = n.declname
            dtype: str = View._get_dtype_name(str(type(args[name])))

            if precision == "":
                precision = dtype
                view = name
            elif precision != dtype:
                sys.exit(f"ERROR: view \"{name}\"'s type does not match current precision,"
                         f" determined to be {precision} from view \"{view}\"")

        if dtype == "float32":
            dtype = "float"
        elif dtype == "float64":
            dtype = "double"

        precision = visitors_util.view_dtypes[dtype].value

        return precision

    def get_result_arguments(self, members: PyKokkosMembers) -> Dict[str, Any]:
        """
        Get the views that are passed as arguments to hold the results for workloads

        :param members: a collection of PyKokkos related members
        :returns: a dictionary of argument name to value
        """

        args: Dict[str, Any] = {}

        for result in members.reduction_result_queue:
            name: str = f"reduction_result_{result}"
            result_view = View([1], DataType.double, MemorySpace.HostSpace)
            args[name] = result_view.array

        for result in members.timer_result_queue:
            name: str = f"timer_result_{result}"
            result_view = View([1], DataType.double, MemorySpace.HostSpace)
            args[name] = result_view.array

        return args

    def get_policy_arguments(self, policy: ExecutionPolicy) -> Dict[str, Any]:
        """
        Get the arguments that are used for to hold the results for workloads

        :param policy: the execution policy of the operation
        :returns: a dictionary of argument name to value
        """

        args: Dict[str, Any] = {}

        if isinstance(policy, RangePolicy):
            args["pk_threads_begin"] = policy.begin
            args["pk_threads_end"] = policy.end
        elif isinstance(policy, TeamPolicy):
            args["pk_league_size"] = policy.league_size
            args["pk_team_size"] = policy.team_size
            args["pk_vector_length"] = policy.vector_length

        return args

    def get_fields(self, members: Dict[str, type]) -> Dict[str, Any]:
        """
        Gets all the primitive type fields from the workload object

        :param workload: the dictionary containing all members
        :returns: a dict mapping from field name to value
        """

        fields: Dict[str, Any] = {}
        for key, value in members.items():
            if type(value) in (int, float, bool, np.int32, np.int64, np.uint32):
                fields[key] = value

        return fields

    def get_views(self, members: Dict[str, type]) -> Dict[str, Any]:
        """
        Gets all the views from the workload object

        :param workload: the dictionary containing all members
        :returns: a dict mapping from view name to object
        """

        views: Dict[str, Any] = {}
        for key, value in members.items():
            if isinstance(value, ViewType):
                views[key] = value.array

        return views

    def retrieve_results(self, workload: object, members: PyKokkosMembers, args: Dict[str, Any]) -> None:
        """
        Get the results for workloads

        :param workload: the workload object
        :param members: a collection of PyKokkos related members
        :param args: the arguments passed to the wrapper, including views that hold results
        """

        for result in members.reduction_result_queue:
            name: str = f"reduction_result_{result}"
            view: View = args[name]
            setattr(workload, result, view[0])

        for result in members.timer_result_queue:
            name: str = f"timer_result_{result}"
            view: View = args[name]
            setattr(workload, result, view[0])


    def run_callbacks(self, workload: object, members: PyKokkosMembers) -> None:
        """
        Run all methods in the workload that are annotated with @pk.callback

        :param workload: the workload object
        :param members: a collection of PyKokkos related members in workload
        """

        callbacks = members.pk_callbacks
        for name in callbacks:
            callback = getattr(workload, name.declname)
            callback()

    def get_module_setup(self, entity: Union[object, Callable[..., None]], space: ExecutionSpace) -> ModuleSetup:
        """
        Get the compiled module setup information unique to an entity + space

        :param entity: the workload or workunit object
        :param space: the execution space
        :returns: the ModuleSetup object
        """

        space: ExecutionSpace = km.get_default_space() if space is ExecutionSpace.Debug else space

        module_setup_id = self.get_module_setup_id(entity, space)
        if module_setup_id in self.module_setups:
            return self.module_setups[module_setup_id]

        module_setup = ModuleSetup(entity, space)
        self.module_setups[module_setup_id] = module_setup

        return module_setup

    def get_module_setup_id(self, entity: Union[object, Callable[..., None]], space: ExecutionSpace) -> Tuple:
        """
        Get a unique module setup id for an entity + space
        combination. For workunits, the idenitifier is just the
        workunit and execution space. For workloads and functors, we
        need the type of the class as well as the file containing it.

        :param entity: the workload or workunit object
        :param space: the execution space
        :returns: a unique tuple per entity and space
        """

        is_workload: bool = not isinstance(entity, Callable)
        is_functor: bool = hasattr(entity, "__self__")

        if is_workload:
            workload_type: Type = type(entity)
            module_setup_id: Tuple[Callable, str, ExecutionSpace] = (
                workload_type, workload_type.__module__, space)
        elif is_functor:
            functor_type: Type = type(entity.__self__)
            module_setup_id: Tuple[Callable, str, str, ExecutionSpace] = (
                type(functor_type), functor_type.__module__, entity.__name__, space)
        else:
            module_setup_id: Tuple[Callable, ExecutionSpace] = (entity, space)

        return module_setup_id

    def get_randpool_args(self, members: Dict[str, type]) -> Dict[str, int]:
        """
        Get the arguments to pass to the RandPool constructor

        :param members: the dictionary containing all members
        :returns: a dict mapping from argument name to value
        """

        arguments: Dict[str, Any] = {}
        for key, value in members.items():
            if isinstance(value, RandomPool):
                arguments[Keywords.RandPoolSeed.value] = value.seed
                arguments[Keywords.RandPoolNumStates.value] = value.num_states
                break

        if len(arguments) == 0:
            arguments[Keywords.RandPoolSeed.value] = 0
            arguments[Keywords.RandPoolNumStates.value] = 0

        return arguments
