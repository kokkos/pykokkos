import ast
import os
import re
import sys
from typing import Dict, List, Optional, Set, Union

from pykokkos import kokkos_manager as km
from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.interface import Layout, MemorySpace, Trait

def pretty_print(node):
    print(ast.dump(node, indent=4))


allowed_types: Dict[str, str] = {
    "int": "int",
    "float": "float",
    "double": "double",
    "bool": "bool",
    "TeamMember": f"Kokkos::TeamPolicy<{Keywords.DefaultExecSpace.value}>::member_type",
    "cpp_auto": "auto"
}

# Maps from the DataType enum to cppast
view_dtypes: Dict[str, Union[cppast.BuiltinType, str]] = {
    "int8": cppast.BuiltinType.INT8,
    "int16": cppast.BuiltinType.INT16,
    "int32": cppast.BuiltinType.INT32,
    "int64": cppast.BuiltinType.INT64,
    "uint8": cppast.BuiltinType.UINT8,
    "uint16": cppast.BuiltinType.UINT16,
    "uint32": cppast.BuiltinType.UINT32,
    "uint64": cppast.BuiltinType.UINT64,
    "float": cppast.BuiltinType.FLOAT,
    "double": cppast.BuiltinType.DOUBLE,

    "int": cppast.BuiltinType.INT32,

    "real": Keywords.RealPrecision.value
}

op2str: Dict[type, str] = {
    # Binary
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Mod: "%",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",  # Pow, Div, and FloorDiv handled separately
    # Unary
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "!",
    ast.Invert: "~",
    # Logical
    ast.And: "&&",
    ast.Or: "||",
    # Comparisons
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    # ast.Is: "is",
    # ast.IsNot: "is not",
    # ast.In: "in",
    # ast.NotIn: "not in",
}

# TODO: provide mapping to cmath versions
math_functions: Set = {
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "copysign",
    "cos",
    "cosh",
    "degrees",
    "exp",
    "expm1",
    "fabs",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "hypot",
    "isfinite",
    "isinf",
    "isnan",
    "isqrt",
    "log",
    "log10",
    "log1p",
    "log2",
    "modf",
    "pow",
    "radians",
    "remainder",
    "round",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "nan",
}

math_constants: Dict[str, str] = {
    "e": "M_E",
    "pi": "M_PI",
    "tau": "2*M_PI",
    "inf": "1.0/0.0",
    "nan": "0.0/0.0",
}


def error(src, debug: bool, node, message) -> None:
    if hasattr(node, "lineno"):
        print(f"\n\033[31m\033[01mError on line {node.lineno} \033[0m: {message}")
    else:
        print(f"\n\033[31m\033[01mError\033[0m: {message}")

    if debug:
        print("DEBUG AST:")
        pretty_print(node)

    if hasattr(node, "lineno"):
        print(src[0][node.lineno - src[1] - 1], end="")
        err_len = node.end_col_offset - node.col_offset if node.end_col_offset else 1
        print(" " * node.col_offset + "^" * err_len)

    sys.exit("PyKokkos: Translation failed")


def generic_error(src, debug: bool, node) -> None:
    error(src, debug, node, "Not supported for translation")


def get_op_str(op: ast.expr) -> str:
    if type(op) not in op2str:
        raise NotImplementedError

    return op2str[type(op)]


def get_math_constant_str(constant: str) -> str:
    if constant not in math_constants:
        raise NotImplementedError

    return math_constants[constant]


def get_allowed_type_str(python_type: str) -> str:
    if python_type not in allowed_types:
        raise NotImplementedError

    return allowed_types[python_type]


def is_math_function(function: str) -> bool:
    return function in math_functions


def get_node_name(node: Union[ast.Attribute, ast.Name]) -> str:
    name: str = ""
    if isinstance(node, ast.Attribute):
        # TODO: check fully qualified names
        name = node.attr
    else:
        name = node.id

    return name


def get_type(annotation: Union[ast.Attribute, ast.Name, ast.Subscript], pk_import: str) -> Optional[cppast.Type]:
    if isinstance(annotation, ast.Attribute):
        if annotation.value.id == pk_import:
            type_name: str = get_node_name(annotation)

            if type_name in view_dtypes:
                return cppast.PrimitiveType(view_dtypes[type_name])

            if type_name in allowed_types:
                type_name = allowed_types[type_name]

            return cppast.ClassType(type_name)

    if isinstance(annotation, ast.Index):
        # ast.Index has been deprecated since Python 3.9;
        # this module attempts to shim around it.

        # should convert to ast.Name:
        annotation = annotation.value

    if isinstance(annotation, ast.Name):
        type_name: str = annotation.id

        if type_name == "int":
            return cppast.PrimitiveType(cppast.BuiltinType.INT)

        if type_name == "float":
            return cppast.PrimitiveType(cppast.BuiltinType.DOUBLE)

        if type_name == "bool":
            return cppast.PrimitiveType(cppast.BuiltinType.BOOL)

        if type_name in allowed_types:
            return cppast.ClassType(type_name)

        if type_name == "List":
            return None

    if isinstance(annotation, ast.Subscript):
        value: Union[ast.Name, ast.Attribute] = annotation.value
        subscript: ast.Index = annotation.slice

        id: str = ""
        if isinstance(value, ast.Name):
            id = value.id
        elif isinstance(value, ast.Attribute):
            id = value.value.id

        if id == "List":
            if sys.version_info.minor <= 8:
                # In Python >= 3.9, ast.Index is deprecated
                # (see # https://docs.python.org/3/whatsnew/3.9.html)
                value = subscript.value
            else:
                value = subscript
            member_type: cppast.Type = get_type(value, pk_import)

            return member_type

        if id == pk_import:
            type_name: str = get_node_name(value)

            if sys.version_info.minor <= 8:
                # In Python >= 3.9, ast.Index is deprecated
                # (see # https://docs.python.org/3/whatsnew/3.9.html)
                dtype_node = subscript.value if isinstance(subscript, ast.Index) else subscript
            else:
                dtype_node: ast.Attribute = subscript

            if type_name == "Acc":
                return get_type(dtype_node, pk_import)

            dtype: cppast.PrimitiveType = get_type(dtype_node, pk_import)
            if dtype is None:
                return None

            view_type = cppast.ClassType(type_name)
            view_type.add_template_param(dtype)

            return view_type

    return None

def parse_view_template_params(
    view_type: cppast.ClassType,
    rank: Optional[int] = None,
    space: Optional[str] = None,
    layout: Optional[str] = None,
    real: Optional[str] = None,
) -> Dict[str, str]:
    """
    Parse the template params of a view type node

    :param view_type: the cppast representation of the view
    :param rank: optionally provide the rank (used by subviews)
    :param space: optionally provide a memory space
    :param layout: optionally provide a layout
    :param real: optionally provide the precision of pk.real dtypes
    :returns: a dict with an entry for each template parameter
    """

    py_type: str = view_type.typename
    is_scratch_view: bool = py_type.startswith("ScratchView")

    if rank is None:
        rank = int(re.search(r'\d+', py_type).group())

    if not 0 < rank < 8:
        raise ValueError(f"View rank {rank} is not allowed")

    params: Dict[str, str] = {}

    # unmanaged views cannot have a layout
    unmanaged: bool = False
    if is_scratch_view:
        unmanaged = True

    template_params: List[cppast.Node] = view_type.template_params
    s = cppast.Serializer()
    for t in template_params:
        parameter: str = s.serialize(t)

        if parameter in ("int", "double", "float",
                            "int8_t", "int16_t", "int32_t", "int64_t",
                            "uint8_t", "uint16_t", "uint32_t", "uint64_t"):
            datatype: str = parameter + "*" * rank
            params["dtype"] = datatype

        elif parameter == Keywords.RealPrecision.value:
            real_dtype: str = Keywords.RealPrecision.value
            if real is not None:
                real_dtype = real
            datatype: str = real_dtype + "*" * rank
            params["dtype"] = datatype

        elif parameter in Trait.__members__:
            if parameter not in ("TraitDefault", "Managed", "Unmanaged"):
                params["trait"] = f"Kokkos::MemoryTraits<Kokkos::{parameter}>"

        elif parameter in Layout.__members__:
            if parameter != "LayoutDefault":
                params["layout"] = f"Kokkos::{parameter}"

        elif parameter in MemorySpace.__members__:
            space = f"Kokkos::{parameter}"

    if "layout" not in params and not unmanaged:
        if layout is not None:
            params["layout"] = layout
        else:
            params["layout"] = f"{Keywords.DefaultExecSpace.value}::array_layout"

    if space is not None:
        if space == "Kokkos::HIPSpace":
            space = "Kokkos::Experimental::HIPSpace"
        params["space"] = space
    elif is_scratch_view:
        params["space"] = f"{Keywords.DefaultExecSpace.value}::scratch_memory_space"
    else:
        params["space"] = f"{Keywords.DefaultExecSpace.value}::memory_space"

    if is_scratch_view:
        params["trait"] = f"Kokkos::MemoryTraits<Kokkos::Unmanaged>"

    return params


def cpp_view_type(
    view_type: cppast.ClassType,
    rank: Optional[int] = None,
    space: Optional[str] = None,
    layout: Optional[str] = None,
    real: Optional[str] = None,
) -> str:
    """
    Get the C++ type of a view

    :param view_type: the cppast representation of the view
    :param rank: optionally provide the rank (used by subviews)
    :param space: optionally provide a memory space
    :param layout: optionally provide a layout
    :param real: optionally provide the precision of pk.real dtypes
    :returns: string representation of C++ view type
    """

    params = parse_view_template_params(view_type, rank, space, layout, real)

    params_ordered: List[str] = []
    params_ordered.append(params["dtype"])
    if "layout" in params:
        params_ordered.append(params["layout"])
    if "space" in params:
        params_ordered.append(params["space"])
    if km.get_kokkos_version() >= 3.7:
        params_ordered.append("Kokkos::Experimental::DefaultViewHooks")
    if "trait" in params:
        params_ordered.append(params["trait"])

    cpp_type: str = "Kokkos::View<"
    cpp_type += ",".join(params_ordered) + ">"

    return cpp_type
