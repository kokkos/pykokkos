from typing import Dict, List, Optional, Tuple

from pykokkos.core import cppast
from pykokkos.core.keywords import Keywords
from pykokkos.core.visitors import cpp_view_type

from .bindings import get_view_memory_space
from .members import PyKokkosMembers

def get_view_type(view: cppast.ClassType) -> str:
    """
    Get the view type with the default memory space and
    array layout macros as a string

    :param view_type: the cppast representation of the view
    :return: the string representation of the view type
    """

    space: str = get_view_memory_space(view, "functor")
    layout: str = f"typename ExecSpace::array_layout"
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

def generate_rand_pool_params() -> Tuple[cppast.ParmVarDecl, cppast.ParmVarDecl]:
    """
    Generate the parameters that initialize the random pool

    :returns: a tuple of the seed parameter and the number of states
        parameter
    """

    seed = cppast.DeclRefExpr(Keywords.RandPoolNumStates.value)
    seed_type = cppast.PrimitiveType(cppast.BuiltinType.INT)
    num_states = cppast.DeclRefExpr(Keywords.RandPoolSeed.value)
    num_states_type = cppast.PrimitiveType(cppast.BuiltinType.INT)

    seed_param = cppast.ParmVarDecl(seed_type, seed)
    num_states_param = cppast.ParmVarDecl(num_states_type, num_states)

    return seed_param, num_states_param

def generate_rand_pool(
    seed: cppast.DeclRefExpr,
    num_states: cppast.DeclRefExpr,
    random_pool: Optional[Tuple[cppast.DeclRefExpr, cppast.ClassType]]
) -> Tuple[cppast.ParmVarDecl, cppast.ParmVarDecl, cppast.AssignOperator, cppast.CallStmt]:
    """
    Generate the code that initializes the random pool

    :param random_pool: a tuple of the pool name and type
    :returns: a tuple of the random pool initialization and the call
        to init()
    """


    # Call to random pool constructor and assignment to field
    op = cppast.BinaryOperatorKind.Assign
    pool_name: str = random_pool[0].declname
    field = cppast.MemberExpr(cppast.DeclRefExpr("this"), pool_name)
    field.is_pointer = True
    value = cppast.CallExpr(cppast.DeclRefExpr(f"Kokkos::{random_pool[1].typename}<>"), [])
    assign = cppast.AssignOperator([field], value, op)

    # Call to random pool init function
    seed = cppast.DeclRefExpr(Keywords.RandPoolNumStates.value)
    init_randpool = cppast.CallStmt(cppast.MemberCallExpr(field, cppast.DeclRefExpr("init"), [seed, num_states]))

    return assign, init_randpool

def generate_constructor(
    name: str,
    fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType],
    views: Dict[cppast.DeclRefExpr, cppast.ClassType],
    random_pool: Optional[Tuple[cppast.DeclRefExpr, cppast.ClassType]],
    has_rand_call: bool
) -> cppast.ConstructorDecl:
    """
    Generate the functor constructor

    :param name: the functor class name
    :param fields: a dict mapping from field name to type
    :param views: a dict mapping from view name to type
    :param random_pool: a tuple of the pool name and type
    :param has_rand_call: whether the function contains a call to pk.rand()
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

    seed_param: cppast.ParmVarDecl
    num_states_param: cppast.ParmVarDecl
    seed_param, num_states_param = generate_rand_pool_params()

    params.append(seed_param)
    params.append(num_states_param)

    if has_rand_call:
        assign, init_randpool = generate_rand_pool(seed_param.declname, num_states_param.declname, random_pool)

        assignments.append(assign)
        assignments.append(init_randpool)

    body = cppast.CompoundStmt(assignments)

    return cppast.ConstructorDecl("", name, params, body)

def generate_constructor_without_rand(
    name: str,
    fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType],
    views: Dict[cppast.DeclRefExpr, cppast.ClassType]
) -> cppast.ConstructorDecl:
    """
    Generate the functor constructor

    :param name: the functor class name
    :param fields: a dict mapping from field name to type
    :param views: a dict mapping from view name to type
    :param random_pool: a tuple of the pool name and type
    :param has_rand_call: whether the function contains a call to pk.rand()
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
    has_rand_call: bool
) -> cppast.RecordDecl:
    """
    Generate the functor source

    :param name: the functor class name
    :param members: an object containing the fields and views
    :param workunits: a dict mapping from workunit name to a tuple of operation type and source
    :param functions: a list of KOKKOS_FUNCTIONS defined in the functor
    :param has_rand_call: whether a kernel contains a call to pk.rand()
    :returns: the cppast representation of the functor
    """

    fields: Dict[cppast.DeclRefExpr, cppast.PrimitiveType] = members.fields
    views: Dict[cppast.DeclRefExpr, cppast.ClassType] = members.views

    decls: List[cppast.DeclStmt] = []

    # Create the tags needed to call individual workunits. Tags in Kokkos are empty structs.
    for n in workunits:
        tag = cppast.RecordDecl(cppast.ClassType(n.declname+"_tag"), [])
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

    random_pool: Optional[Tuple[cppast.DeclRefExpr, cppast.ClassType]] = members.random_pool
    if has_rand_call:
        pool_name: cppast.DeclRefExpr = random_pool[0]
        pool_type: str = random_pool[1].typename
        decls.append(cppast.DeclStmt(cppast.FieldDecl(f"Kokkos::{pool_type}<>", pool_name)))

    decls.append(generate_constructor(name, fields, views, random_pool, has_rand_call))
    decls.append(generate_constructor_without_rand(name, fields, views))
    for _, s in workunits.values():
        decls.append(s)

    for f in functions:
        decls.append(cppast.DeclStmt(f))

    struct = cppast.RecordDecl(cppast.ClassType(name), decls)
    struct.is_definition = True

    struct.add_template_param("ExecSpace")
    if members.has_real:
        struct.add_template_param(Keywords.RealPrecision.value)
        s = cppast.Serializer()

    return struct
