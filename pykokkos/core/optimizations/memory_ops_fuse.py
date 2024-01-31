import ast
from collections import deque
import itertools
from typing import Deque, Dict, List, Tuple

from pykokkos.core.translators import StaticTranslator

from .util import ExpressionFinder, MemoryOpInfo


def partition_by_array(memory_ops: List[MemoryOpInfo]) -> List[List[MemoryOpInfo]]:
    """
    Partition the list of memory operations into sets that correspond
    to operations on the same array

    :param memory_ops: the current list of operations
    :returns: a list of lists of memory operations per array
    """

    return [list(group) for _, group in itertools.groupby(memory_ops, key=lambda op: (op.array_name, op.index_key))]


def partition_by_index(memory_ops: List[List[MemoryOpInfo]]) -> List[List[MemoryOpInfo]]:
    """
    Partition the list of memory operations into sets that access the
    array by the same index

    :param memory_ops: a previously partitioned list of lists
        operations
    """

    partitioned_ops: List[List[MemoryOpInfo]] = []
    for ops in memory_ops:
        current_ops: Deque[MemoryOpInfo] = deque(ops)
        indices_list_map: Dict[str, List[MemoryOpInfo]] = {}

        while len(current_ops) > 0:
            next_op: MemoryOpInfo = current_ops.popleft()
            index: Tuple[str, ...] = next_op.index_key

            if index in indices_list_map:
                indices_list_map[index].append(next_op)
            else:
                indices_list_map[index] = [next_op]

        partitioned_ops.extend(list(indices_list_map.values()))

    return partitioned_ops


def partition_by_access(memory_ops: List[List[MemoryOpInfo]]) -> List[List[MemoryOpInfo]]:
    """
    Partition the list of memory operations at memory writes

    :param memory_ops: a previously partitioned list of lists
        operations
    :returns: a list of lists of memory operations partitioned at
        memory writes
    """

    partitioned_ops: List[List[MemoryOpInfo]] = []

    for ops in memory_ops:
        current_ops: Deque[MemoryOpInfo] = deque(ops)

        same_access: List[MemoryOpInfo] = []
        while len(current_ops) > 0:
            next_op: MemoryOpInfo = current_ops.popleft()
            print(f"next op is {next_op}")

            if isinstance(next_op.context, ast.Load):
                same_access.append(next_op)
            else:
                if len(same_access) > 0:
                    partitioned_ops.append(same_access)
                partitioned_ops.append([next_op]) # Put the Store in its own list
                same_access = []

        if len(same_access) > 0:
            partitioned_ops.append(same_access)

    return partitioned_ops


def identify_fusable_operations(memory_ops_in_scope: Dict[int, List[MemoryOpInfo]]) -> List[List[MemoryOpInfo]]:
    """
    Partition the list of memory operations into sets that can be
    fused

    :param memory_ops_in_scope: a dictionary mapping from each scope to a list
        of memory operations
    :returns: a list of memory operations that can be fused into one
    """

    fusable_memory_ops: List[List[MemoryOpInfo]] = []

    for memory_ops in memory_ops_in_scope.values():
        memory_ops.sort(key=lambda op: (op.array_name, op.parent_stmt.idx_in_parent))

        from pprint import pprint

        current_ops: List[List[MemoryOpInfo]] = partition_by_array(memory_ops)
        print("after array partition")
        pprint(current_ops)
        current_ops = partition_by_index(current_ops)
        print("after index partition")
        pprint(current_ops)
        current_ops = partition_by_access(current_ops)
        print("after access partition")
        pprint(current_ops)
        fusable_memory_ops.extend(current_ops)

    return fusable_memory_ops


def find_memory_ops(AST: ast.FunctionDef) -> Dict[int, List[MemoryOpInfo]]:
    """
    Find all memory ops in the kernel and gather basic information on
    them.

    :param AST: the AST of the workunit
    :returns: a list of memory operations
    """

    memory_ops_finder = ExpressionFinder()
    memory_ops_finder.visit(AST)

    return memory_ops_finder.memory_ops_in_scope


def generate_fused_load(
    op_info: MemoryOpInfo,
    array_access_counts: Dict[Tuple[str, Tuple[str, ...]], int],
    pk_import: str
) -> ast.AnnAssign:
    """
    Generate a definition of the fused memory load in a Python AST
    Node

    :param op_info: information relating to the current memory
        operation
    :param array_access_counts: maintains how many times each
        combination of array + index has been accessed
    :param pk_import: the alias used in the pykokkos import
    :returns: the Python AST node of the fused load
    """

    array_name: str = op_info.array_name
    index = "_".join(op_info.index_key)

    access_count: int = array_access_counts.get((array_name, index), 0)
    array_access_counts[(array_name, index)] = access_count + 1

    target = ast.Name(id=f"pk_fused_{array_name}_{index}_{access_count}", ctx=ast.Store())
    annotation = ast.Attribute(value=ast.Name(id=pk_import, ctx=ast.Load()), attr="cpp_auto", ctx=ast.Load())
    value: ast.Subscript = op_info.memory_op

    # From https://docs.python.org/3/library/ast.html, "simple is a
    # boolean integer set to True for a Name node in target that do
    # not appear in between parenthesis and are hence pure names and
    # not expressions" (which is the case for our generated fused
    # loads)
    return ast.AnnAssign(target=target, annotation=annotation, value=value, simple=1)


def replace_with_fused_load(memory_ops: List[MemoryOpInfo], fused_name: str) -> None:
    """
    Replace the memory loads in a given list with an access to the
    fused load

    :param memory_ops: the list of memory operations being modified to
        use the fused load
    :param fused_load: the name of the variable holding the loaded
        value
    """

    for op in memory_ops:
        parent: ast.AST = op.memory_op.parent
        parent_accessor: str = op.memory_op.parent_accessor

        fused_node = ast.Name(id=fused_name, ctx=ast.Load())

        ref_in_parent = getattr(parent, parent_accessor)
        print(ast.dump(op.memory_op, indent=4))
        print(ast.dump(parent, indent=4))
        if isinstance(ref_in_parent, list):
            ref_in_parent[op.memory_op.idx_in_parent] = fused_node
        else:
            print(f"setting parent {parent} field {parent_accessor} to {fused_name}")
            setattr(parent, parent_accessor, fused_node)


def insert_fused_loads(fused_loads: List[Tuple[ast.AnnAssign, ast.stmt]]) -> None:
    """
    Inset a list of fused memory loads at the beginning of their
    parent statement's scope

    :param fused_loads: a list of tuples of each new variable
        definition holding a fused load and its parent statement node
    """

    for fused_load, parent_stmt in fused_loads:
        scope_node: ast.AST = parent_stmt.parent
        parent_accessor: str = parent_stmt.parent_accessor

        ref_in_parent: List[ast.AST] = getattr(scope_node, parent_accessor)
        assert isinstance(ref_in_parent, list)

        ref_in_parent.insert(0, fused_load)


def fuse_memory_ops(fusable_ops: List[List[MemoryOpInfo]], pk_import: str) -> None:
    """
    Fuse memory ops corresponding to the same value

    :param fusable_ops: a list of list of memory operations that can
        be fused
    :param pk_import: the alias used in the pykokkos import
    """

    # Used to assign a unique index to each tuple of array name + index
    array_access_counts: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    # Keep track of all generated fused loads and their parent
    # statement, which will be used to store them in the right place
    fused_loads: List[Tuple[ast.AnnAssign, ast.stmt]] = []

    for ops in fusable_ops:
        if len(ops) == 1:
            continue

        assert isinstance(ops[0].context, ast.Load)

        fused_load: ast.AnnAssign = generate_fused_load(ops[0], array_access_counts, pk_import)
        fused_loads.append((fused_load, ops[0].parent_stmt))
        replace_with_fused_load(ops, fused_load.target.id)

    insert_fused_loads(fused_loads)


def memory_ops_fuse(AST: ast.FunctionDef, pk_import: str) -> None:
    """
    Eliminate redundant memory operations within a kernel by storing
    loaded and written values into registers

    :param AST: the AST of the workunit to optimize
    :param pk_import: the alias used in the pykokkos import
    """

    # This information might be out of date following loop fusion
    StaticTranslator.add_parent_refs(AST)

    memory_ops: Dict[int, List[MemoryOpInfo]] = find_memory_ops(AST)
    from pprint import pprint
    pprint(f"got memory ops")
    pprint(memory_ops)
    fusable_ops: List[List[MemoryOpInfo]] = identify_fusable_operations(memory_ops)
    fuse_memory_ops(fusable_ops, pk_import)

    # print(ast.dump(AST, indent=4))

    # print(ast.unparse(AST))
