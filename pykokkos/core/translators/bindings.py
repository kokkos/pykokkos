import sys
import re
from typing import Dict, List, Optional, Tuple

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.core.visitors import cpp_view_type, KokkosMainVisitor, visitors_util
from pykokkos.interface.data_types import DataType

from .members import PyKokkosMembers

def is_hierarchical(workunit: Optional[cppast.MethodDecl]) -> bool:
    """
    Checks if a workunit uses hierarchical parallelism by checking if it has a TeamMember instead of a thread ID

    :param workunit: the workunit definition or None for a workload
    :returns: true if hierarchical false otherwise
    """

    if workunit is None:
        return False

    # Iterate over each parameter (skipping the tag)
    for p in workunit.params[1:]:
        if isinstance(p.decltype, cppast.ClassType):
            return True

    return False

def get_view_memory_space(view_type: cppast.ClassType, location: str) -> str:
    """
    Get the memory space of a view. Return the default memory space if
    space was not specified in ViewTypeInfo

    :param view_type: the cppast type of the view extracted from the
        source
    :param location: where the view_type is needed, either "bindings"
        or "functor"
    :returns: the memory space template parameter
    """

    # Check if there is a MemorySpace template parameter
    for t in view_type.template_params:
        if isinstance(t, cppast.DeclRefExpr):
            name: str = t.declname

            if name.endswith("Space"):
                return f"Kokkos::{name}"

    if location == "functor":
        return f"typename ExecSpace::memory_space"
    if location == "bindings":
        return Keywords.ArgMemSpace.value

def get_kernel_params(
    members: PyKokkosMembers,
    is_hierarchical: bool,
    is_workload: bool,
    real: Optional[str]
) -> Dict[str, str]:
    """
    Get the parameters of the kernel. The parameters include the fields, the views,
    the views holding the reduction results, and the view holding the timer results.
    Also add parameters for the parameters of the execution policy.

    :param members: an object containing the fields and views
    :param is_hierarchical: does the workunit use hierarchical parallelism
    :param real: the precision for which to generate a binding
    :returns: a dict mapping from argument name to type
    """

    s = cppast.Serializer()
    params: Dict[str, str] = {}
    for n, t in members.fields.items():
        params[n.declname] = s.serialize(t)

    for n, t in members.views.items():
        # skip subviews
        if t is None:
            continue

        space: str = get_view_memory_space(t, "bindings")
        layout: str = f"{Keywords.DefaultExecSpace.value}::array_layout"
        params[n.declname] = cpp_view_type(t, space=space, layout=layout, real=real)

    params[Keywords.DefaultExecSpaceInstance.value] = Keywords.DefaultExecSpace.value

    if not is_workload:
        params[Keywords.KernelName.value] = "const std::string&"

        if is_hierarchical:
            params[Keywords.LeagueSize.value] = "int"
            params[Keywords.TeamSize.value] = "int"
            params[Keywords.VectorLength.value] = "int"
        else:
            params[Keywords.ThreadsBegin.value] = "int"
            params[Keywords.ThreadsEnd.value] = "int"

    params[Keywords.RandPoolSeed.value] = "int"
    params[Keywords.RandPoolNumStates.value] = "int"

    for result in members.reduction_result_queue:
        view_name = f"reduction_result_{result}"
        view_type = cppast.ClassType("View1D")
        view_type.add_template_param(cppast.DeclRefExpr("double"))
        view_type.add_template_param(cppast.DeclRefExpr("HostSpace"))
        params[view_name] = cpp_view_type(view_type, space="Kokkos::HostSpace", layout="Kokkos::LayoutRight")

    for result in members.timer_result_queue:
        view_name = f"timer_result_{result}"
        view_type = cppast.ClassType("View1D")
        view_type.add_template_param(cppast.DeclRefExpr("double"))
        view_type.add_template_param(cppast.DeclRefExpr("HostSpace"))
        params[view_name] = cpp_view_type(view_type, space="Kokkos::HostSpace", layout="Kokkos::LayoutRight")

    return params

def get_device_views(members: PyKokkosMembers) -> Dict[str, str]:
    """
    Get a map from views to device views

    :param members: an object containing the fields and views
    :returns: a list of names of the device names
    """

    return {v.declname: f"pk_d_{v.declname}" for v in members.views \
            if members.views[v] is not None}

def generate_functor_instance(functor: str, members: PyKokkosMembers, with_random_args: bool=True, functor_exec_space: Optional[str] = None, always_use_kokkos_copy: bool = False) -> str:
    """
    Generate the functor instance

    :param functor: the name of the functor
    :param members: an object containing the fields and views
    :param with_random_args: bool indicating if the constructor call should have the random args. default = True
    :param functor_exec_space: optional parameter of type str that contains the ExecSpace template argument to the functor
    :param always_use_kokkos_copy: optional parameter specifying if all copies should be done with kokkos functionality
    :returns: the source code for instantiating the functor
    """

    args: List[str] = []

    for f in members.fields:
        args.append(f.declname)

    mirror_views: str
    exec_space_instance: str = Keywords.DefaultExecSpaceInstance.value

    if functor_exec_space is None:
        mirror_views = ""
    else:
        mirror_views = f"{functor_exec_space} {exec_space_instance};"

    device_views: Dict[str, str] = get_device_views(members)
    for v, d_v in device_views.items():
        args.append(d_v)

        if not always_use_kokkos_copy:
            view_type: cppast.ClassType = members.views[cppast.DeclRefExpr(v)]
            if get_view_memory_space(view_type, "bindings") == Keywords.ArgMemSpace.value:
                mirror_views += f"auto {d_v} = Kokkos::create_mirror_view_and_copy({exec_space_instance}, {v});"
            else:
                mirror_views += f"auto {d_v} = {v};"
        else:
            mirror_views += f"auto {d_v} = Kokkos::create_mirror_view_and_copy({exec_space_instance}, {v});"

    # Kokkos fails to compile a functor if there are no parameters in its constructor
    if len(args) == 0:
        args.append("0")

    if with_random_args:
        args.append(Keywords.RandPoolSeed.value)
        args.append(Keywords.RandPoolNumStates.value)

    constructor: str = f"{functor} {Keywords.Instance.value}"
    constructor += "(" + ",".join(args) + ");"

    return mirror_views + constructor

def generate_copy_back_from_dict(members: PyKokkosMembers,deep_copy_args: Dict[str,str]) -> str:
    """
    Generate the code that does the resize and deep_copy

    :param members: an object containing the fields and views
    :param deep_copy_args: dict of pairs of deep_copy args
    :returns: the source code for instantiating the functor
    """

    copy_back: str = ""

    for v, d_v in deep_copy_args.items():
        view_type: cppast.ClassType = members.views[cppast.DeclRefExpr(v)]
        # skip subviews
        if view_type is None:
            continue

        # skip views with user-set memory spaces
        if get_view_memory_space(view_type, "bindings") != Keywords.ArgMemSpace.value:
            continue

        # Need to resize views for binsort. Unmanaged views cannot be resized.
        if cppast.DeclRefExpr("Unmanaged") not in view_type.template_params:
            rank = int(re.search(r'\d+', view_type.typename).group())
            resize_args: List[str] = [v]

            for i in range(rank):
                resize_args.append(f"{d_v}.extent({i})")

            copy_back += f"Kokkos::resize("
            copy_back += ",".join(resize_args)
            copy_back += ");"

        copy_back += f"Kokkos::deep_copy({v}, {d_v});"

    return copy_back

def generate_copy_back(members: PyKokkosMembers) -> str:
    """
    Generate the code that copies back the views

    :param members: an object containing the fields and views
    :returns: the source code for instantiating the functor
    """
    device_views: Dict[str, str] = get_device_views(members)

    return generate_copy_back_from_dict(members,device_views)

def get_return_type(operation: str, workunit: cppast.MethodDecl) -> str:
    """
    Get the return type of a binding

    :param operation: the type of the operation (for, reduce, scan, or workload)
    :param workunit: the workunit for which the binding is being generated
    :returns: the return type as a string
    """ 

    acc_decl: Optional[cppast.ParmVarDecl] = None
    if operation == "reduce":
        acc_decl = workunit.params[-1]
    elif operation == "scan":
        acc_decl = workunit.params[-2]

    return_type: str
    if acc_decl is None:
        return_type = "void"
    else:
        return_type = acc_decl.decltype.typename.value

    return return_type

def generate_kernel_signature(return_type: str, kernel: str, params: Dict[str, str]) -> str:
    """
    Generate the kernel signature

    :param return_type: the return type of the kernel
    :param kernel: the name of the kernel
    :param params: the parameters of the kernel
    :returns: the kernel signature
    """

    signature: str = f"{return_type} {kernel}("
    signature += ",".join([f"{t} {n}" for n, t in params.items()])
    signature += ")"

    return signature

def generate_fence_call() -> str:
    """
    Generate a C++ function call to Kokkos fence

    :returns: the call to the current execution space's fence
    """

    return f"{Keywords.DefaultExecSpaceInstance.value}.fence();"

def generate_call(operation: str, functor: str, members: PyKokkosMembers, tag: cppast.DeclRefExpr, is_hierarchical: bool) -> str:
    """
    Generate the calls to the operation

    :param operation: the type of the operation i.e. "for", "reduce", or "scan"
    :param functor: the name of the functor
    :param members: an object containing the fields and views
    :param tag: the name of the workunit
    :param is_hierarchical: is the workunit used with hierarchical parallelism
    :returns: the source code for creating the subviews
    """

    call: str = f"Kokkos::parallel_{operation}("

    args: List[str] = [Keywords.KernelName.value]

    tag_name: str = tag.declname+"_tag"
    if is_hierarchical:
        args.append(f"Kokkos::TeamPolicy<{Keywords.DefaultExecSpace.value},{functor}::{tag_name}>({Keywords.DefaultExecSpaceInstance.value}, {Keywords.LeagueSize.value},Kokkos::AUTO,{Keywords.VectorLength.value})")
    else:
        args.append(f"Kokkos::RangePolicy<{Keywords.DefaultExecSpace.value},{functor}::{tag_name}>({Keywords.DefaultExecSpaceInstance.value}, {Keywords.ThreadsBegin.value},{Keywords.ThreadsEnd.value})")

    args.append(Keywords.Instance.value)

    if operation in ("reduce", "scan"):
        args.append(Keywords.Accumulator.value)

    call += ",".join(args)
    call += ");"

    if is_hierarchical:
        # Create an if-else statement. In the body of the if, call the workunit
        # with Kokkos::AUTO for team_size. In the else, pass in pk_team_size

        custom_call: str = call.replace("Kokkos::AUTO", Keywords.TeamSize.value)
        call = f"if({Keywords.TeamSize.value} == -1) {{ {call}"
        call += f"}} else {{ {custom_call} }}"

    call += generate_copy_back(members)
    # call += generate_fence_call()

    if operation in ("reduce", "scan"):
        call += f"return {Keywords.Accumulator.value};"

    return call

def generate_wrapper(
    members: PyKokkosMembers,
    operation: str,
    workunit: cppast.MethodDecl,
    wrapper: str,
    kernel: str,
    real: Optional[str]
) -> str:
    """
    Generate the wrapper that calls the kernel and its binding

    :param members: an object containing the fields and views
    :param operation: the type of the operation (for, reduce, scan, or workload)
    :param workunit: the workunit for which the binding is being generated
    :param wrapper: the name of the wrapper
    :param kernel: the name of the kernel
    :param real: the precision for which to generate a binding
    :returns: the wrapper source
    """

    is_workload: bool = True if operation == "workload" else False
    params: Dict[str, str] = get_kernel_params(members, is_hierarchical(workunit), is_workload, real)
    return_type: str = get_return_type(operation, workunit)

    args: List[str] = []
    for name, param_type in params.items():
        if param_type == "const std::string&":
            args.append(f"kwargs[\"{name}\"].cast<std::string>()")
        else:
            args.append(f"kwargs[\"{name}\"].cast<{param_type}>()")

    kernel_call: str = f"{kernel}("
    kernel_call += ",".join(args)
    kernel_call += ");"

    definition: str = f"{return_type} {wrapper}(pybind11::kwargs kwargs) {{"
    if return_type != "void":
        definition += f"return {kernel_call};"
    else:
        definition += f"{kernel_call};"
    definition += "}"

    return definition

def generate_kernel(
    functor: str,
    members: PyKokkosMembers,
    operation: str,
    workunit: cppast.MethodDecl,
    tag: cppast.DeclRefExpr,
    kernel: str,
    real: Optional[str]
) -> str:
    """
    Generate the kernel that calls the workunit

    :param functor: the functor class name
    :param members: an object containing the fields and views
    :param operation: the type of the operation (for, reduce, or scan)
    :param workunit: the workunit for which the binding is being generated
    :param tag: the name of the workunit
    :param kernel: the name of the kernel
    :param real: the precision for which to generate a binding
    :returns: the kernel source
    """

    hierarchical: bool = is_hierarchical(workunit)
    params: Dict[str, str] = get_kernel_params(members, hierarchical, False, real)
    return_type: str = get_return_type(operation, workunit)
    signature: str = generate_kernel_signature(return_type, kernel, params)

    acc: str = ""
    if operation in ("reduce", "scan"):
        acc = f"{return_type} {Keywords.Accumulator.value} = 0;"

    if members.has_real:
        functor += f"<{Keywords.DefaultExecSpace.value},{real}>"
    else:
        functor += f"<{Keywords.DefaultExecSpace.value}>"

    instance: str = generate_functor_instance(functor, members)
    call: str = generate_call(operation, functor, members, tag, hierarchical)

    kernel: str = f"{signature} {{ {acc} {instance} {call} }}"

    return kernel

def bind_wrappers(module: str, wrappers: List[str]) -> str:
    """
    Generate the binding code for all wrappers

    :returns: the binding code
    """

    variable: str = "k"
    binding: str = f"PYBIND11_MODULE({module}, {variable}) {{"
    for w in wrappers:
        binding += f"{variable}.def(\"{w}\", &{w});"
    binding += "}"

    return binding

def bind_workunits_single(
    functor: str,
    members: PyKokkosMembers,
    workunits: Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]],
    precision: Optional[DataType]
) -> Tuple[List[str], List[str]]:
    """
    Generates the bindings for a group of workunits. Each workunit is
    called inside a kernel, and each kernel is called from a wrapper
    that is bound with pybind.

    :param functor: the functor class name
    :param members: an object containing the fields and views
    :param workunits: a dictionary mapping form workunit name to a tuple of operation type and source
    :param precision: the precision for which to generate a binding
    :returns: a tuple of lists of strings of representing the wrapper names, and the kernels and wrappers
    """

    bindings: List[str] = []
    wrappers: List[str] = []

    real: Optional[str] = None
    if precision is not None:
        real = visitors_util.view_dtypes[precision.name].value

    for n, t in workunits.items():
        workunit_name: str = n.declname
        wrapper_name: str = f"wrapper_{workunit_name}"
        kernel_name: str = f"run_{workunit_name}"

        if precision is not None:
            wrapper_name += f"_{real}"
            kernel_name += f"_{real}"

        wrappers.append(wrapper_name)

        operation: str = t[0]
        workunit: cppast.MethodDecl = t[1]

        kernel: str = generate_kernel(functor, members, operation, workunit, n, kernel_name, real)
        wrapper: str = generate_wrapper(members, operation, workunit, wrapper_name, kernel_name, real)

        bindings.extend([kernel, wrapper])

    return wrappers, bindings

def bind_workunits(
    functor: str,
    members: PyKokkosMembers,
    workunits: Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]],
    module: str
) -> List[str]:
    """
    Generates the bindings for a group of workunits. Each workunit is
    called inside a kernel, and each kernel is called from a wrapper
    that is bound with pybind.

    :param functor: the functor class name
    :param members: an object containing the fields and views
    :param workunits: a dictionary mapping form workunit name to a tuple of operation type and source
    :param module: the name of the generated module
    :returns: a list of strings of all kernels, wrappers, and bindings
    """

    bindings: List[str] = []
    wrapper_names: List[str] = []
    if members.has_real:
        for d in DataType:
            if d is DataType.real:
                continue
            w, b = bind_workunits_single(functor, members, workunits, d)
            bindings.extend(b)
            wrapper_names.extend(w)
    else:
        w, b = bind_workunits_single(functor, members, workunits, None)
        bindings.extend(b)
        wrapper_names.extend(w)

    bindings.append(bind_wrappers(module, wrapper_names))

    return bindings

def translate_mains(source: Tuple[List[str], int], functor: str, members: PyKokkosMembers, pk_import: str) -> List[str]:
    """
    Translate all PyKokkos main functions

    :param source: the python source code of the workload
    :param functor: the name of the functor
    :param members: an object containing the fields and views
    :returns: a list of strings of translated source code
    """

    node_visitor = KokkosMainVisitor(
        {}, source, members.views, members.pk_workunits,
        members.fields, members.pk_functions,
        members.classtype_methods, functor, pk_import, debug=True)

    translation: List[str] = []

    for main in members.pk_mains.values():
        try:
            translation.append(node_visitor.visit(main))
        except NotImplementedError:
            print(f"Translation of {main.name} failed")
            sys.exit(1)

    members.reduction_result_queue = node_visitor.reduction_result_queue
    members.timer_result_queue = node_visitor.timer_result_queue

    return translation

def bind_main_single(
    functor: str,
    members: PyKokkosMembers,
    source: Tuple[List[str], int],
    pk_import: str,
    precision: Optional[DataType]
) -> Tuple[str, str]:
    """
    Generates the kernel and its python binding

    :param functor: the functor class name
    :param members: an object containing the fields and views
    :param source: the python source code of the workload
    :param pk_import: the pykokkos import alias
    :param precision: the precision for which to generate a binding
    :returns: a tuple of strings containing the wrapper name, and the kernel and wrapper
    """

    wrapper_name: str = "wrapper"
    kernel_name: str = "run"

    real: Optional[str] = None
    if precision is not None:
        real = visitors_util.view_dtypes[precision.name].value
        wrapper_name += f"_{real}"
        kernel_name += f"_{real}"
        functor += f"<{Keywords.DefaultExecSpace.value},{real}>"
    else:
        functor += f"<{Keywords.DefaultExecSpace.value}>"

    main: List[str] = translate_mains(source, functor, members, pk_import)
    params: Dict[str, str] = get_kernel_params(members, False, True, real)

    # fall back to the old hard-coded default
    # for now--this includes cases where an
    # accumulator is not even defined
    acc_type = "double"

    for element in source[0]:
        # TODO: support more types
        if "pk.Acc" in element:
            if "pk.int64" in element:
                acc_type = "int64_t"
            elif "pk.double" in element:
                acc_type = "double"

    signature: str = generate_kernel_signature("void", kernel_name, params)
    instantiation: str = generate_functor_instance(functor, members)
    acc: str = f"{acc_type} {Keywords.Accumulator.value} = 0;"
    body: str = "".join(main)
    copy_back: str = generate_copy_back(members)
    # fence: str = generate_fence_call()

    kernel: str = f"{signature} {{ {instantiation} {acc} {body} {copy_back} }}"
    wrapper: str = generate_wrapper(members, "workload", None, wrapper_name, kernel_name, real)
    binding: str = f"{kernel} {wrapper}"

    return wrapper_name, binding

def bind_main(
    functor: str,
    members: PyKokkosMembers,
    source: Tuple[List[str], int],
    pk_import: str,
    module: str
) -> List[str]:
    """
    Generates the kernel and its python binding

    :param functor: the functor class name
    :param members: an object containing the fields and views
    :param source: the python source code of the workload
    :param pk_import: the pykokkos import alias
    :param module: the name of the generated module
    :returns: a list of strings containing the kernel, wrapper, and binding
    """

    bindings: List[str] = []
    wrapper_names: List[str] = []
    if members.has_real:
        for d in DataType:
            if d is DataType.real:
                continue
            w, b = bind_main_single(functor, members, source, pk_import, d)
            bindings.append(b)
            wrapper_names.append(w)
    else:
        w, b = bind_main_single(functor, members, source, pk_import, None)
        bindings.append(b)
        wrapper_names.append(w)

    bindings.append(bind_wrappers(module, wrapper_names))

    return bindings
