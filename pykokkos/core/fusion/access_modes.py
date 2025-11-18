import ast
from enum import auto, Enum
from typing import Dict, List, Optional, Set, Tuple

from .util import add_parent_refs

class AccessMode(Enum):
    Read = auto()
    Write = auto()
    ReadWrite = auto()

class AccessIndex(Enum):
    Empty = 0
    Constant = 1
    TID = 2
    TIDFunc = 3
    Iter = 4
    All = 5

def get_view_access_modes(AST: ast.FunctionDef, view_args: Set[str]) -> Dict[str, AccessMode]:
    AST = add_parent_refs(AST)
    access_modes: Dict[str, AccessMode] = {}

    for node in ast.walk(AST):
        # For now, treat any view passed to a call as a RW access
        if isinstance(node, ast.Call):
            for arg in node.args:
                if not isinstance(arg, ast.Name):
                    continue
                if arg.id in view_args:
                    access_modes[arg.id] = AccessMode.ReadWrite
            continue

        if not isinstance(node, ast.Subscript): # We are only interested in view accesses
            continue

        # Skip type annotations
        if isinstance(node.parent, ast.arg):
            continue

        # Skip inner subscripts as they will be handled by the below while loop
        if isinstance(node.parent, ast.Subscript) and isinstance(node.parent.value, ast.Subscript):
            continue

        current_node: ast.Subscript = node
        while isinstance(current_node, ast.Subscript):
            current_node = current_node.value

        # Go back up one to the parent subscript
        if isinstance(current_node, ast.Name):
            current_node = current_node.parent

        # The subscript node that holds the load/store context is the
        # top level one.
        context_node: ast.Subscript = current_node
        while isinstance(context_node.parent, ast.Subscript):
            context_node = context_node.parent

        name: str = current_node.value.id
        if name not in view_args:
            continue

        existing_mode: Optional[AccessMode] = access_modes.get(name)
        new_mode: AccessMode

        if isinstance(context_node.ctx, ast.Load):
            if existing_mode is None:
                new_mode = AccessMode.Read
            elif existing_mode is AccessMode.Write:
                new_mode = AccessMode.ReadWrite
            else:
                new_mode = existing_mode

        if isinstance(context_node.ctx, ast.Store):
            if existing_mode is None:
                new_mode = AccessMode.Write
            elif existing_mode is AccessMode.Read:
                new_mode = AccessMode.ReadWrite
            else:
                new_mode = existing_mode

        if new_mode is AccessMode.Write and isinstance(node.parent, ast.AugAssign):
            new_mode = AccessMode.ReadWrite

        access_modes[name] = new_mode

    return access_modes

class WriteIndicesVisitor(ast.NodeVisitor):
    def __init__(self, tid_name: str, view_args: Dict[str, int]):
        self.tid_name = tid_name
        self.view_args = view_args

        # Map from each view (str) + dimension (int) to an AccessIndex
        self.access_indices: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]] = {}
        self.current_iters: List[Tuple[str, bool]] = []

    def visit_For(self, node: ast.For) -> None:
        index_node = node.target

        is_tid_iter: bool = False
        range_call: ast.Call = node.iter
        for arg in range_call.args:
            if isinstance(arg, ast.Name) and arg.id == self.tid_name:
                is_tid_iter = True

        self.current_iters.append((index_node.id, is_tid_iter))
        for b in node.body:
            self.visit(b)

        self.current_iters.pop()

    def visit_Call(self, node: ast.Call) -> None:
        # Treat function calls like a black box
        for arg in node.args:
            if not isinstance(arg, ast.Name):
                self.visit(arg)

            # If an entire view is passed to a function
            elif arg.id in self.view_args:
                rank: int = self.view_args[arg.id]
                for i in range(rank):
                    self.access_indices[(arg.id, i)] = (AccessIndex.All, AccessMode.ReadWrite, "")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        current_node: ast.Subscript = node
        slices: List = []

        while isinstance(current_node, ast.Subscript):
            index = current_node.slice

            slices.insert(0, index)
            current_node = current_node.value

        # The subscript itself could be indexing another view
        for s in slices:
            self.visit(s)

        # Avoid type annotations
        if isinstance(current_node, ast.Attribute):
            return

        assert isinstance(current_node, ast.Name)
        view_name: str = current_node.id

        if view_name not in self.view_args:
            return

        for i, index_node in enumerate(slices):
            index_node_str = ast.unparse(index_node)

            if isinstance(index_node, ast.Constant):
                new_index = AccessIndex.Constant
            elif isinstance(index_node, ast.Name) and index_node.id == self.tid_name:
                new_index = AccessIndex.TID
            elif self.tid_name in index_node_str:
                new_index = AccessIndex.TIDFunc
            elif (index_node_str, True) in self.current_iters:
                new_index = AccessIndex.TID
            elif (index_node_str, False) in self.current_iters:
                new_index = AccessIndex.Iter
            else:
                new_index = AccessIndex.All

            index_to_set: AccessIndex
            mode_to_set: AccessMode

            existing_access: Optional[Tuple[AccessIndex, AccessMode, str]] = self.access_indices.get((view_name, i))
            if existing_access is None:
                index_to_set = new_index
                mode_to_set = AccessMode.Read if isinstance(node.ctx, ast.Load) else AccessMode.Write
            else:
                existing_index: AccessIndex = existing_access[0]
                existing_mode: AccessMode = existing_access[1]

                # We will update the existing index if it is None or if
                # the new index's value (see enum above) is higher then
                # the existing value
                if new_index.value > existing_index.value:
                    index_to_set = new_index
                else:
                    index_to_set = existing_index

                if isinstance(current_node.ctx, ast.Load):
                    if existing_mode is AccessMode.Write:
                        mode_to_set = AccessMode.ReadWrite
                    else:
                        mode_to_set = existing_mode

                if isinstance(current_node.ctx, ast.Store):
                    if existing_mode is AccessMode.Read:
                        mode_to_set = AccessMode.ReadWrite
                    else:
                        mode_to_set = existing_mode

            if mode_to_set is AccessMode.Write and isinstance(node.parent, ast.AugAssign):
                mode_to_set = AccessMode.ReadWrite

            self.access_indices[(view_name, i)] = (index_to_set, mode_to_set, index_node_str)


def get_view_write_indices_and_modes(AST: ast.FunctionDef, view_args: Dict[str, int]) -> Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]]:
    """
    Get information from the AST needed for fusion safety

    :param AST: the AST of the workunit
    :param view_args: the set of view names and dimensionality
    :returns: the safety info
    """
    AST = add_parent_refs(AST)

    tid_name: str = AST.args.args[0].arg
    visitor = WriteIndicesVisitor(tid_name, view_args)
    visitor.visit(AST)
    access_indices: Dict[Tuple[str, int], Tuple[AccessIndex, AccessMode, str]] = visitor.access_indices

    return access_indices