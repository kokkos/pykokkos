from typing import Dict, List, Optional, Tuple

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.core.visitors import cpp_view_type, visitors_util
from pykokkos.interface.data_types import DataType
from .members import PyKokkosMembers
from .bindings import generate_functor_instance,generate_copy_back,get_view_memory_space,generate_copy_back_from_dict


def generate_cast_from_object(functor: str, members: PyKokkosMembers, precision: Optional[DataType]) -> str:
    """
    Generates the source code of a c++ template for a "cast" operation from the python object to a c++ functor.
    DOES NOT CHECK ANY FIELDS/TYPES/ETC

    :param functor: the name of the functor without template args
    :param members: an object containing the fields and views
    :param precision: optional argument of type DataType for floatingpoint
    :returns: the source code of the functor cast
    """

    real_str: Optional[str] = None
    if precision is not None:
        real_str = visitors_util.view_dtypes[precision.name].value

    if members.has_real:
        functor_type = f"{functor}<FunctorExecSpace,{real_str}>"
        template_decl = f"template <class FunctorExecSpace,class ArgumentMemorySpace,class {real_str}>"
    else:
        functor_type = f"{functor}<ExecSpace>"
        template_decl = f"template <class ExecSpace,class ArgumentMemorySpace>"

    cast_source = f"{template_decl} {functor_type} {functor}_from_pyObject(pybind11::object obj) {{"

    # get functor members incl types
    s = cppast.Serializer()
    params: Dict[str, str] = {}
    for n, t in members.fields.items():
        params[n.declname] = s.serialize(t)

    for n, t in members.views.items():
        # skip subviews
        if t is None:
            continue

        space: str = "ArgumentMemorySpace"
        layout: str = f"typename ExecSpace::array_layout"
        params[n.declname] = cpp_view_type(t, space=space, layout=layout,real=real_str)

    #cast arguments into cpp types
    for name, param_type in params.items():
        if param_type == "const std::string&":
            cast_source += f"std::string {name} = getattr(obj,\"{name}\").cast<std::string>();"
        else:
            cast_source += f"{param_type} {name} = getattr(obj,\"{name}\").cast<{param_type}>();"


    cast_source += generate_functor_instance(functor_type, members, False, "ExecSpace", True)
    cast_source += f"return {Keywords.Instance.value};"
    cast_source += "}"

    return cast_source


def generate_cast_to_object(functor: str, members: PyKokkosMembers, precision: Optional[DataType]) -> str:
    """
    Generates the source code of a c++ template for a "cast" operation from the c++ functor to a python object.
    DOES NOT CHECK ANY FIELDS/TYPES/ETC

    :param functor: the name of the functor without template args
    :param members: an object containing the fields and views
    :param precision: optional argument of type DataType for floatingpoint
    :returns: the source code of the functor cast
    """

    real_str: Optional[str] = None
    if precision is not None:
        real_str = visitors_util.view_dtypes[precision.name].value

    if members.has_real:
        functor_type = f"{functor}<ExecSpace,{real_str}>"
        template_decl = f"template <class ExecSpace,class ArgumentMemorySpace, class {real_str}>"
    else:
        functor_type = f"{functor}<ExecSpace>"
        template_decl = f"template <class ExecSpace,class ArgumentMemorySpace>"

    cast_source = f"{template_decl} void {functor}_to_pyObject({functor_type}& functor, pybind11::object obj) {{"

    # get functor members incl types
    s = cppast.Serializer()
    params: Dict[str, str] = {}
    for n, t in members.fields.items():
        params[n.declname] = s.serialize(t)

    for n, t in members.views.items():
        # skip subviews
        if t is None:
            continue

        space: str = "ArgumentMemorySpace"
        layout: str = f"typename ExecSpace::array_layout"
        params[n.declname] = cpp_view_type(t, space=space, layout=layout,real=real_str)

    #cast members into cpp types
    for name, param_type in params.items():
        if param_type == "const std::string&":
            cast_source += f"std::string {name} = pybind11::getattr(obj,\"{name}\").cast<std::string>();"
        else:
            cast_source += f"{param_type} {name} = pybind11::getattr(obj,\"{name}\").cast<{param_type}>();"


    deep_copy_args: Dict[str, str] = {v.declname: f"functor.{v.declname}" for v in members.views \
            if members.views[v] is not None}

    cast_source += generate_copy_back_from_dict(members,deep_copy_args)

    #assign members from cpp types
    for name, param_type in params.items():
        cast_source += f"pybind11::setattr(obj,\"{name}\",pybind11::cast({name}));"

    cast_source += "}"

    return cast_source

def generate_cast(functor: str, members: PyKokkosMembers) -> List[str]:
    """
    Generates the c++ templates for casting a functor to/from a python object. A pybind11 cast operator would need to know how to
    construct the python object the functor wants to cast into. This is a problem as we do not know where the user defines 
    the functor object and I did not want to import the main script into the c++ binding code just to know
    how to construct the functor object in c++.
    This is solved by passing in a python object when casting from the c++ object to the python object.
    This leaves it to the user that calls the cast that the python object actually has all the members the cast assigns to ... 
    the function has no guarantees or checks, as the interpreter should throw a runtime error if any of the assignments fails,
    which is as good as it gets.
    Both "casts" take care of the deep_copy and resize operations. This said, they are more like glorified copy operations

    :param functor: the name of the functor without template arguments
    :param members: object containing fields and views of the functor
    :returns: the source code of the cast operators
    """

    source: List[str] = []
    if members.has_real:
        for d in DataType:
            if d is DataType.real:
                continue
            source.append(generate_cast_from_object(functor,members,d))
            source.append(generate_cast_to_object(functor,members,d))

    else:
        source.append(generate_cast_from_object(functor,members,None))
        source.append(generate_cast_to_object(functor,members,None))

    return source
