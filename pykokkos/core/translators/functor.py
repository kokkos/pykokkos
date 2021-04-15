from typing import Dict, List, Tuple

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.core.visitors import cpp_view_type

from .members import PyKokkosMembers

def get_view_type(view: cppast.ClassType) -> str:
    """
    Get the view type with the default memory space and
    array layout macros as a string

    :param view_type: the cppast representation of the view
    :return: the string representation of the view type
    """

    space: str = f"{Keywords.DefaultExecSpace.value}::memory_space"
    layout: str = f"{Keywords.DefaultExecSpace.value}::array_layout"
    view_type: str = cpp_view_type(view, space=space, layout=layout)

    return view_type


def generate_assignments(members: Dict[cppast.DeclRefExpr, cppast.Type]) -> List[cppast.AssignOperator]:
    """
    Generate the assignments in the constructor

    :param members: the members being assigned
    :returns: the list of assignments
    """

    assignments: List[cppast.AssignOperator] = []

    for n, t in members.items():
        op = cppast.BinaryOperatorKind.Assign
        field = cppast.MemberExpr(cppast.DeclRefExpr("this"), n.declname)
        field.is_pointer = True
        assign = cppast.AssignOperator([field], n, op)

        assignments.append(assign)

    return assignments


def generate_constructor(
    name: str,
    fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType],
    views: Dict[cppast.DeclRefExpr, cppast.ClassType]
) -> cppast.ConstructorDecl:
    """
    Generate the functor constructor

    :param name: the functor class name
    :param fields: a dict mapping from field name to type
    :param views: a dict mapping from view name to type
    :returns: the cppast representation of the constructor
    """

    params: List[cppast.ParmVarDecl] = []
    assignments: List[cppast.AssignOperator] = []

    for n, t in fields.items():
        params.append(cppast.ParmVarDecl(t, n))

    for n, t in views.items():
        # skip subviews
        if t is None:
            continue
        view_type: str = get_view_type(t)
        params.append(cppast.ParmVarDecl(view_type, n))

    # Kokkos fails to compile a functor if there are no parameters in its constructor
    if len(params) == 0:
        decl = cppast.DeclRefExpr("pk_field")
        type = cppast.PrimitiveType(cppast.BuiltinType.INT)
        params.append(cppast.ParmVarDecl(type, decl))

    assignments.extend(generate_assignments(fields))
    # skip subviews
    assignments.extend(generate_assignments({v: views[v] for v in views if views[v]}))

    body = cppast.CompoundStmt(assignments)

    return cppast.ConstructorDecl("", name, params, body)


def generate_functor(
    name: str,
    members: PyKokkosMembers,
    workunits: Dict[cppast.DeclRefExpr, Tuple[str, cppast.MethodDecl]],
    functions: List[cppast.MethodDecl],
) -> cppast.RecordDecl:
    """
    Generate the functor source

    :param name: the functor class name
    :param members: an object containing the fields and views
    :param workunits: a dict mapping from workunit name to a tuple of operation type and source
    :param functions: a list of KOKKOS_FUNCTIONS defined in the functor
    :param has_real: whether the function contains a pk.real datatype
    :returns: the cppast representation of the functor
    """

    fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType] = members.fields
    views: Dict[cppast.DeclRefExpr, cppast.ClassType] = members.views

    decls: List[cppast.DeclStmt] = []

    # Create the tags needed to call individual workunits. Tags in Kokkos are empty structs.
    for n in workunits:
        tag = cppast.RecordDecl(cppast.ClassType(n.declname), [])
        tag.is_definition = True
        decls.append(tag)

    for n, t in fields.items():
        decls.append(cppast.DeclStmt(cppast.FieldDecl(t, n)))

    for n, t in views.items():
        # skip subviews
        if t is None:
            continue
        view_type: str = get_view_type(t)
        decls.append(cppast.DeclStmt(cppast.FieldDecl(view_type, n)))

    decls.append(generate_constructor(name, fields, views))
    for _, s in workunits.values():
        decls.append(s)

    for f in functions:
        decls.append(cppast.DeclStmt(f))

    struct = cppast.RecordDecl(cppast.ClassType(name), decls)
    struct.is_definition = True

    if members.has_real:
        struct.add_template_param(Keywords.RealPrecision.value)
        s = cppast.Serializer()

    return struct
