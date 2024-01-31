import ast
import re
from typing import Any, Dict, List, Tuple, Union


class LoopInfo:
    """
    Contains info on loops necessary to carry out optimizations
    """

    def __init__(
        self,
        iterator: ast.Name,
        start: ast.expr,
        stop: ast.expr,
        step: ast.expr,
        for_node: ast.For,
        scope: int,
        parent_scope: int, #, will be changed as loops are fused
        parent_node: ast.stmt, # will be changed as loops are fused
        idx_in_parent: int, #, will be changed as loops are fused
        original_parent_node: ast.stmt, # needed to check if adjacent nodes can be moved
        original_idx_in_parent: int, # needed to check if adjacent nodes can be moved
        lineno: int #, using this as a unique id to the loop for now
    ):
        self.iterator = iterator
        self.start = start
        self.stop = stop
        self.step = step
        self.for_node = for_node
        self.scope = scope
        self.parent_scope = parent_scope
        self.parent_node = parent_node
        self.idx_in_parent = idx_in_parent
        self.original_parent_node = original_parent_node
        self.original_idx_in_parent = original_idx_in_parent
        self.lineno = lineno

        start = ast.unparse(self.start)
        stop = ast.unparse(self.stop)
        step = ast.unparse(self.step)

        r0 = re.search("fused_(.*)_[0-9]*", start)
        r1 = re.search("fused_(.*)_[0-9]*", stop)
        r2 = re.search("fused_(.*)_[0-9]*", step)

        self.start_key: str = r0.group(1) if r0 else start
        self.stop_key: str = r1.group(1) if r1 else stop
        self.step_key: str = r2.group(1) if r2 else step

    def get_range_key(self) -> Tuple[str, str, str]:
        """
        Get the key reprenting the range of this loop

        :returns: a tuple of start, stop, and end
        """

        return (self.start_key, self.stop_key, self.step_key)

    def __repr__(self) -> str:
        return (f"LoopInfo(iterator={ast.unparse(self.iterator)}, "
                f"start={self.start_key}, "
                f"stop={self.stop_key}, "
                f"step={self.step_key}, "
                f"parent_scope={self.parent_scope}, "
                f"scope={self.scope})")


class MemoryOpInfo:
    """
    Contains info on memory operations necessary to carry out optimizations
    """

    def __init__(self, array_name: str, memory_op: ast.Subscript, indices: List[ast.expr], context: ast.expr_context, parent_stmt: ast.stmt):
        self.array_name = array_name
        self.memory_op = memory_op
        self.indices = indices
        self.context = context
        self.parent_stmt = parent_stmt

        self.index_key: Tuple[str, ...] = tuple([ast.unparse(i) for i in self.indices])

    def __repr__(self) -> str:
        return (f"MemoryOpInfo(array={self.array_name}, "
                f"index={self.index_key}, "
                f"context={self.context})")


class ExpressionFinder(ast.NodeVisitor):
    """
    Node visitor that finds all for loops and memory operations
    """

    def __init__(self):
        self.scope_id: int = 0
        self.scope_stack: List[int] = [self.scope_id]
        self.loops_in_scope: Dict[int, List[LoopInfo]] = {}
        self.memory_ops_in_scope: Dict[int, List[MemoryOpInfo]] = {}

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
            idx_in_parent_body: int = node.idx_in_parent
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

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Skip Subscript nodes that are type annotations
        if isinstance(node.parent, (ast.arg)):
            return

        indices: List[ast.AST] = []
        array_node: Union[ast.Subscript, ast.Name] = node

        while not isinstance(array_node, ast.Name):
            indices.insert(0, array_node.slice)
            array_node = array_node.value

        # Get the first ancestor node that is of type ast.stmt. This
        # will be used to find the location that the memory load will
        # be inserted later on. The assumption is that this statement
        # will be contained in a list of statements in its parent, and
        # the fused loads will be inserted at the start of that
        # statement
        parent_stmt: ast.AST = node.parent
        while not isinstance(parent_stmt, ast.stmt):
            parent_stmt = parent_stmt.parent

        parent_scope: int = self.get_scope()
        if parent_scope not in self.memory_ops_in_scope:
            self.memory_ops_in_scope[parent_scope] = []

        array_name: str = array_node.id
        self.memory_ops_in_scope[parent_scope].append(MemoryOpInfo(array_name, node, indices, node.ctx, parent_stmt))
