import ast
from typing import Dict, Set, Tuple, Union


def get_node_name(node: Union[ast.Attribute, ast.Name]) -> str:
    """
    Copied from visitors_util.py due to circular import
    """

    name: str
    if isinstance(node, ast.Attribute):
        name = node.attr
    else:
        name = node.id

    return name


class DeclarationsVisitor(ast.NodeVisitor):
    """
    Get all variable declarations
    """

    def __init__(self) -> None:
        self.declarations: Set[str] = set()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.declarations.add(get_node_name(node.target))

    def visit_For(self, node: ast.For) -> None:
        self.declarations.add(get_node_name(node.target))
        for n in node.body:
            self.visit(n)

class VariableRenamer(ast.NodeTransformer):
    """
    Renames variables in a fused ast according to a map
    """

    def __init__(self, name_map: Dict[Tuple[str, int], str], workunit_idx: int):
        self.name_map = name_map
        self.workunit_idx = workunit_idx

    def visit_Name(self, node: ast.Name) -> None:
        key = (node.id, self.workunit_idx)
        # If the name is not mapped, keep the original name
        node.id = self.name_map.get(key, node.id)
        return node

    def visit_keyword(self, node: ast.keyword) -> None:
        key = (node.id, self.workunit_idx)
        # If the name is not mapped, keep the original name
        node.arg = self.name_map.get(key, node.arg)
        return node
