import ast
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

from pykokkos.core.fusion.util import DeclarationsVisitor, VariableRenamer
from pykokkos.core.translators import StaticTranslator


@dataclass
class LoopInfo:
    """
    Contains info on loops necessary to carry out optimizations
    """

    iterator: ast.Name
    start: ast.expr
    stop: ast.expr
    step: ast.expr
    for_node: ast.For
    scope: int
    parent_scope: int # will be changed as loops are fused
    parent_node: ast.stmt # will be changed as loops are fused
    idx_in_parent: int # will be changed as loops are fused
    original_parent_node: ast.stmt # needed to check if adjacent nodes can be moved
    original_idx_in_parent: int # needed to check if adjacent nodes can be moved
    lineno: int

    def get_range_key(self) -> Tuple[str, str, str]:
        """
        Get the key reprenting the range of this loop

        :returns: a tuple of start, stop, and end
        """

        return (ast.unparse(self.start), ast.unparse(self.stop), ast.unparse(self.step))

    def __repr__(self) -> str:
        return (f"LoopInfo(iterator={ast.unparse(self.iterator)}, "
                f"start={ast.unparse(self.start)}, "
                f"stop={ast.unparse(self.stop)}, "
                f"step={ast.unparse(self.step)}, "
                f"parent_scope={self.parent_scope}, "
                f"scope={self.scope})")


class LoopFinder(ast.NodeVisitor):
    """
    Node visitor that finds all for loops
    """

    def __init__(self):
        self.scope_id: int = 0
        self.scope_stack: List[int] = [self.scope_id]
        self.loops_in_scope: Dict[int, List[LoopInfo]] = {}

    def push_scope(self, id: int) -> None:
        """
        Push a scope onto the scope stack

        :param id: the scope to push
        """

        self.scope_stack.append(id)

    def pop_scope(self) -> None:
        """
        Pop a scope from the stack
        """

        self.scope_stack.pop()

    def get_scope(self) -> int:
        """
        Get the current scope

        :returns: the id of the scope
        """

        return self.scope_stack[-1]

    def is_range_call(self, node: ast.expr) -> bool:
        """
        Check if an ast expression is a range call

        :returns: True if range() is called
        """

        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"

    def get_loop_range(self, node: ast.Call) -> Tuple[ast.expr, ast.expr, ast.expr]:
        """
        Get a loop's range from a range call

        :param node: the node corresponding to the range call
        :returns: a tuple of the range's start, stop, and step
        """

        range_args: List = node.args

        start: ast.expr
        stop: ast.expr
        step: ast.expr

        if len(range_args) == 1:
            start = ast.Constant(0)
            stop = range_args[0]
            step = ast.Constant(1)
        elif len(range_args) == 2:
            start = range_args[0]
            stop = range_args[1]
            step = ast.Constant(1)
        elif len(range_args) == 3:
            start = range_args[0]
            stop = range_args[1]
            step = range_args[2]

        return start, stop, step

    def visit_If(self, node: ast.If) -> None:
        self.scope_id += 1
        self.push_scope(self.scope_id)
        for b in node.body:
            self.visit(b)
        self.pop_scope()

        self.scope_id += 1
        self.push_scope(self.scope_id)
        for b in node.orelse:
            self.visit(b)
        self.pop_scope()

    def visit_While(self, node: ast.While) -> None:
        self.scope_id += 1
        self.push_scope(self.scope_id)
        for b in node.body:
            self.visit(b)
        self.pop_scope()

    def visit_For(self, node: ast.For) -> None:
        self.scope_id += 1
        range_call: ast.Call = node.iter

        if self.is_range_call(range_call):
            start: ast.expr
            stop: ast.expr
            step: ast.expr
            start, stop, step = self.get_loop_range(range_call)

            parent_scope: int = self.get_scope()
            idx_in_parent_body: int = node.idx_in_parent - 2 # Subtract 2 for iterator and range call
            loop_info = LoopInfo(
                node.target, start, stop, step, node, self.scope_id, parent_scope,
                node.parent, idx_in_parent_body, node.parent, idx_in_parent_body, node.lineno)

            if parent_scope not in self.loops_in_scope:
                self.loops_in_scope[parent_scope] = []

            self.loops_in_scope[parent_scope].append(loop_info)

        self.push_scope(self.scope_id)
        for statement in node.body:
            self.visit(statement)
        
        self.pop_scope()


def find_loops(AST: ast.FunctionDef) -> Dict[int, List[LoopInfo]]:
    """
    Find all loops in a given AST

    :param AST: the AST containing the loops
    :returns: a dict mapping scope ids to lists of loops they contain
    """

    loop_finder = LoopFinder()
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
    a constant, as these could not possibly affect anything else.

    :param node: the node being considered
    :returns: True if the node can be moved
    """

    return isinstance(node, ast.AnnAssign) and isinstance(node.value, (ast.Constant, ast.Name))


def can_make_adjacent(current_loop: LoopInfo, next_loop: LoopInfo) -> bool:
    """
    This function checks if two loops that are not adjacent can be
    made adjacent by checking if the nodes between them can be moved
    prior to the first loop.

    :param current_loop: the loop that occurs first
    :param next_loop: the loop that occurs second
    :returns: True if the nodes separating them can be moved
    """

    nodes_current: List[ast.stmt] = current_loop.original_parent_node.body[current_loop.original_idx_in_parent + 1:]
    nodes_next: List[ast.stmt] = next_loop.original_parent_node.body[:next_loop.original_idx_in_parent]
    nodes_between: List[ast.stmt] = nodes_current + nodes_next

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

        for loop_list in adjacent_loops:
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

        for loop_idx, loop in enumerate(loops):
            rename_variables(loop, loop_idx, new_iterator)
            if loop_idx == 0:
                continue

            make_adjacent(main_loop, loop)
            # Append renamed statements
            main_loop.for_node.body += loop.for_node.body
            # Remove old for loops
            loop.parent_node.body = [n for n in loop.parent_node.body if n.lineno != loop.lineno]


def loop_fuse(AST: ast.FunctionDef):
    """
    Fuse loops within a workunit

    :param AST: the AST of the workunit to optimize
    """

    StaticTranslator.add_parent_refs(AST)

    loops_in_scope: Dict[int, List[LoopInfo]] = find_loops(AST)
    fusable_loops: List[List[LoopInfo]] = identify_fusable_loops(loops_in_scope)
    fuse_loops(fusable_loops)