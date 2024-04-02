import ast
from collections import deque
from typing import Deque, Dict, List, Set, Tuple

from pykokkos.core.fusion.util import DeclarationsVisitor, VariableRenamer

from .util import add_parent_refs, ExpressionFinder, LoopInfo


def find_loops(AST: ast.FunctionDef) -> Dict[int, List[LoopInfo]]:
    """
    Find all loops in a given AST

    :param AST: the AST containing the loops
    :returns: a dict mapping scope ids to lists of loops they contain
    """

    loop_finder = ExpressionFinder()
    loop_finder.visit(AST)

    return loop_finder.loops_in_scope


def group_equivalent_ranges(loops: List[LoopInfo]) -> List[List[LoopInfo]]:
    """
    Partition a list of loops into sets that have equivalent iteration
    spaces 

    :param loops: the list of loops to be partitioned (already in the
        same scope)
    :returns: a list of loop lists that have the same iteration space
    """

    equivalent_range_loops: Dict[int, List[LoopInfo]] = {}
    parent_scope: int = loops[0].parent_scope

    for loop in loops:
        assert loop.parent_scope == parent_scope

        key = loop.get_range_key()
        if key not in equivalent_range_loops:
            equivalent_range_loops[key] = []
        equivalent_range_loops[key].append(loop)

    return list(equivalent_range_loops.values())


def safe_to_move(node: ast.stmt) -> bool:
    """
    Checks if a node is safe to move. Right now, this only checks if
    the nodes between them are variable declarations initialized with
    a constant, or an expression that can be evaluated at
    compile-time, as these could not possibly affect anything else.

    :param node: the node being considered
    :returns: True if the node can be moved
    """

    if not isinstance(node, ast.AnnAssign):
        return False

    if isinstance(node.value, (ast.Constant, ast.Name)):
        return True

    value: str = ast.unparse(node.value)
    eval_success: bool
    try:
        eval(value)
        eval_success = True
    except:
        eval_success = False

    return eval_success


def can_make_adjacent(current_loop: LoopInfo, next_loop: LoopInfo) -> bool:
    """
    This function checks if two loops that are not adjacent can be
    made adjacent by checking if the nodes between them can be moved
    prior to the first loop.

    :param current_loop: the loop that occurs first
    :param next_loop: the loop that occurs second
    :returns: True if the nodes separating them can be moved
    """

    current_parent: ast.stmt = current_loop.original_parent_node
    next_parent: ast.stmt = next_loop.original_parent_node

    nodes_between: List[ast.stmt]
    if current_parent is next_parent:
        nodes_between = current_parent.body[current_loop.original_idx_in_parent + 1:next_loop.original_idx_in_parent]
    else:
        nodes_current: List[ast.stmt] = current_parent.body[current_loop.original_idx_in_parent + 1:]
        nodes_next: List[ast.stmt] = next_parent.body[:next_loop.original_idx_in_parent]
        nodes_between = nodes_current + nodes_next

    for node in nodes_between:
        if not safe_to_move(node):
            return False

    return True


def make_adjacent(current_loop: LoopInfo, next_loop: LoopInfo) -> None:
    """
    This function moves all nodes after the current loop and before
    the next loop to before the current loops.

    :param current_loop: the loop that occurs first
    :param next_loop: the loop that occurs second
    """

    current_idx: int = current_loop.idx_in_parent
    next_idx: int = next_loop.idx_in_parent
    current_parent: ast.stmt = current_loop.parent_node

    if current_idx + 1 == next_idx:
        return

    between_nodes: List[ast.stmt] = current_parent.body[current_idx + 1: next_idx]

    del current_parent.body[current_idx + 1: next_idx]

    # Insert these nodes right before the loop
    current_parent.body[current_idx:current_idx] = between_nodes
    current_loop.idx_in_parent += len(between_nodes)


def group_adjacent_loops(loop_sets: List[List[LoopInfo]]) -> List[List[LoopInfo]]:
    """
    Given a list of candidate sets of loops for fusion, partition
    these sets into sets that are adjacent, i.e. that occur
    consecutively in the parent's body

    :param loop_sets: lists of loops that belong to the same scope
    :returns: a further partitioned list of loop sets that are
        adjacent or can be made adjacent
    """

    adjacent_loops: List[List[LoopInfo]] = []

    for loop_list in loop_sets:
        if len(loop_list) == 1:
            continue

        # Reverse and pop elements (more efficient for lists)
        loop_list.sort(key=lambda loop: loop.idx_in_parent, reverse=True)
        current_list: List[LoopInfo] = [loop_list.pop()]

        while len(loop_list) > 0:
            current_loop: LoopInfo = current_list[-1]
            next_loop: LoopInfo = loop_list.pop()

            if next_loop.idx_in_parent == current_loop.idx_in_parent + 1 or can_make_adjacent(current_loop, next_loop):
                current_list.append(next_loop)
            else:
                adjacent_loops.append(current_list)
                current_list = [next_loop]

        adjacent_loops.append(current_list)        

    return adjacent_loops


def fuse_scopes(loop_list: List[LoopInfo], loops_in_scope: Dict[int, List[LoopInfo]]) -> None:
    """
    After fusing loops, the different scopes must be combined into
    one. We will need to update the loops_in_scope to reflect these
    changes. This function selects a loop in loop_list which the other
    loops will be fused into, and updates the parent scope and node
    information of the loops contained in the fused loops accordingly.

    :param loop_list: the loops that will be fused
    :param loops_in_scope: a dict mapping from each scope to the list of
        loops it contains
    """

    new_id: int = loop_list[0].scope
    new_parent: ast.stmt = loop_list[0].for_node
    old_ids: List[int] = [l.scope for l in loop_list[1:]]

    for old_id in old_ids:
        if old_id not in loops_in_scope:
            continue
        if new_id not in loops_in_scope:
            loops_in_scope[new_id] = []

        # Used to calculate the new index of a for loop in its
        # parent after fusion. This is needed to keep track of
        # which loops will be adjacent after fusion of their
        # parents, but the actual fusion is performed at a
        # later stage.
        new_idx_start: int = len(new_parent.body)

        for loop in loops_in_scope[old_id]:
            loop.parent_scope = new_id
            loop.parent_node = new_parent
            loop.idx_in_parent += new_idx_start
            new_idx_start += len(loop.for_node.body)
            loops_in_scope[new_id].append(loop)


def get_loop_reads_writes(loop: LoopInfo) -> Tuple[Set[str], Set[str]]:
    """
    Given a loop, get all the reads and writes of that loop
    """

    local_vars: Set[str] = set()

    # This is very inefficient. Should maybe do this in a node
    # visitor, where nodes will be visited in the right order,
    # and nodes in AnnAssign will be visited properly
    for node in ast.walk(loop.for_node):
        if isinstance(node, ast.AnnAssign):
            local_vars.add(node.target.id)

    loop_reads: Set[str] = set()
    loop_writes: Set[str] = set()
    arrays_seen: Set[str] = set()

    iterator_name: str = loop.iterator.id

    for node in ast.walk(loop.for_node):
        if isinstance(node, ast.Subscript):
            arrays_seen.add(node.value.id)
            array_name: str = node.value.id

            current_node = node

            while isinstance(current_node, ast.Subscript):
                index = current_node.slice
                index_name: str

                if isinstance(index, ast.Constant):
                    index_name = str(index.value)
                elif isinstance(index, ast.Name):
                    if index_name in local_vars:
                        raise RuntimeError("Can't fuse loops with local vars as indices")

                    # Still making a big assumption that differently
                    # named indices won't have the same value
                    index_name = index.id
                else:
                    raise RuntimeError("Can't analyze deps with complex indices (for now)")

                if index_name == iterator_name:
                    array_name += f"_pk_it"
                else:
                    array_name += f"_{index_name}"

                current_node = current_node.value

        elif isinstance(node, ast.Name):
            if node.id in local_vars:
                continue

            if isinstance(node.ctx, ast.Store):
                loop_writes.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loop_reads.add(node.id)
            else:
                raise RuntimeError("Del unsupported")

    return loop_reads, loop_writes


def split_negative_dependencies(loop_sets: List[List[LoopInfo]]) -> List[List[LoopInfo]]:
    """
    Split loops that have negative dependencies. According to
    https://llvm.org/devmtg/2018-10/slides/Barton-LoopFusion.pdf, a
    negative dependency occurs between L_j and L_k, such that L_j
    happens before L_k, when at iteration m L_k uses a value that is
    computed by L_j at a future iteration m + n (where n > 0). Also
    what if L_j reads from a memory location that L_k is writing to?

    1st: get all writes to variables/arrays in each loop
    2nd: get all reads to variables/arrays in each loop

    Note that there is no aliasing because pykokkos doesn't really let
    users take the address of different variables like C++ does.
    """

    for loop_list in loop_sets:
        if len(loop_list) == 1:
            continue

        loop_reads: Dict[int, Set[str]] = {}
        loop_writes: Dict[int, Set[str]] = {}

        for loop in loop_list:
            reads: Set[str]
            writes: Set[str]
            reads, writes = get_loop_reads_writes(loop)

            loop_reads[loop.lineno] = reads
            loop_writes[loop.lineno] = writes

        # Reverse and pop elements (more efficient for lists)
        loop_list.sort(key=lambda loop: loop.idx_in_parent, reverse=True)
        current_list: List[LoopInfo] = [loop_list.pop()]

        while len(loop_list) > 0:
            current_loop: LoopInfo = current_list[-1]
            next_loop: LoopInfo = loop_list.pop()

            current_reads: Set[str] = loop_reads[current_loop.lineno]
            next_reads: Set[str] = loop_reads[next_loop.lineno]

            current_writes: Set[str] = loop_writes[current_loop.lineno]
            next_writes: Set[str] = loop_writes[next_loop.lineno]

            for current_write in current_writes:
                for next_write in next_writes:
                    pass
                for next_read in next_reads:
                    pass

            for next_write in next_writes:
                for current_write in current_writes:
                    pass
                for current_read in current_reads:
                    pass


def split_unfusable_loops(loop_sets: List[List[LoopInfo]]) -> List[List[LoopInfo]]:
    """
    Split the loop sets if they contain statements that cannot be
    fused (such as print statements)

    :param loop_sets: lists of loops that are adjacent
    :returns: a further partitioned list of loop sets that do not
        contain prints
    """

    split_loops: List[List[LoopInfo]] = []

    for loop_list in loop_sets:
        if len(loop_list) == 1:
            continue

        loop_list.sort(key=lambda loop: loop.idx_in_parent, reverse=True)
        current_list: List[LoopInfo] = [loop_list.pop()]

        while len(loop_list) > 0:
            next_loop: LoopInfo = loop_list.pop()

            contains_print: bool = False
            for node in ast.walk(next_loop.for_node):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "printf":
                        contains_print = True
                        break

            if contains_print:
                split_loops.append(current_list)
                current_list = [next_loop]
            else:
                current_list.append(next_loop)

        split_loops.append(current_list)

    return split_loops


def identify_fusable_loops(loops_in_scope: Dict[int, List[LoopInfo]]) -> List[List[LoopInfo]]:
    """
    Partition the list of loops into sets that can be fused    

    :param loops_in_scope: a dict mapping from each scope to the list of
        loops it contains
    :returns: a list of sets of loops to be fused
    """

    fusable_loops: List[List[LoopInfo]] = []
    scopes: Deque[int] = deque(loops_in_scope.keys())

    # Start from the outermost scope. Fuse scopes that end up being
    # fused by updating the corresponding loops_in_scope entry.
    while len(scopes) > 0:
        scope: int = scopes.popleft()
        if scope not in loops_in_scope:
            continue
        loops: List[LoopInfo] = loops_in_scope[scope]

        # Group loops with the same iteration space
        equivalent_ranges: List[List[LoopInfo]] = group_equivalent_ranges(loops)
        adjacent_loops: List[List[LoopInfo]] = group_adjacent_loops(equivalent_ranges)
        split_loops: List[List[LoopInfo]] = split_unfusable_loops(adjacent_loops)

        for loop_list in split_loops:
            if len(loop_list) == 1:
                continue

            fusable_loops.append(loop_list)
            fuse_scopes(loop_list, loops_in_scope)

    return fusable_loops


def rename_variables(loop: LoopInfo, loop_idx: int, new_iterator: str) -> List[ast.stmt]:
    """
    Rename the variables in a loop's body according to the loop's
    index

    :param loop: the loop being renamed
    :param loop_idx: the unique index corresponding to this loop
    :param new_iterator: the new name of the loop's iterator
    """

    declarations = DeclarationsVisitor()
    declarations.visit(loop.for_node)

    name_map: Dict[Tuple[str, int], str] = {}
    for declaration in declarations.declarations:
        new_name: str = f"pk_fused_{declaration}_{loop_idx}"
        name_map[(declaration, loop_idx)] = new_name

    iterator: str = loop.iterator.id
    name_map[(iterator, loop_idx)] = new_iterator

    loop.iterator.id = new_iterator
    renamer = VariableRenamer(name_map, loop_idx)

    renamed_stmts: List[ast.stmt] = [renamer.visit(s) for s in loop.for_node.body]

    return renamed_stmts


def fuse_loops(fusable_loops: List[List[LoopInfo]]) -> None:
    """
    Perform loop fusion at the AST level in a given workunit

    :param fusable_loops: contains list of loops that can be fused into one
    """

    for idx, loops in enumerate(fusable_loops):
        assert len(loops) > 1
        main_loop: LoopInfo = loops[0]
        new_iterator: str = f"pk_fused_it_{idx}"
        main_loop_added: bool = False

        for loop_idx, loop in enumerate(loops):
            rename_variables(loop, loop_idx, new_iterator)
            if loop_idx == 0:
                continue

            make_adjacent(main_loop, loop)
            # Append renamed statements
            main_loop.for_node.body += loop.for_node.body
            # Remove old for loops
            new_body = []
            for n in loop.parent_node.body:
                if n.lineno != loop.lineno:
                    new_body.append(n)

                # This avoids an issue where a workunit is being fused
                # with itself and it contains a for loop. If we just
                # keep the above condition, no loops will be added
                # because all loops being fused have the same lineno
                if n.lineno == loop.lineno and loop.lineno == main_loop.lineno:
                    if not main_loop_added:
                        new_body.append(n)
                        main_loop_added = True

            loop.parent_node.body = new_body


def loop_fuse(AST: ast.FunctionDef) -> None:
    """
    Fuse loops within a workunit

    :param AST: the AST of the workunit to optimize
    """

    add_parent_refs(AST)

    loops_in_scope: Dict[int, List[LoopInfo]] = find_loops(AST)
    fusable_loops: List[List[LoopInfo]] = identify_fusable_loops(loops_in_scope)
    fuse_loops(fusable_loops)
