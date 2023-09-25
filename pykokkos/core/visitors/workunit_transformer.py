from _ast import FunctionDef
from ast import NodeTransformer
from typing import Any

class RemoveTransformer(NodeTransformer):

    def __init__(self, node:FunctionDef):
        self.remove_this = node

    def set_remove_this(self, node:FunctionDef):
        self.remove_this = node

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        
        if node.name == self.remove_this.name:
            return None
        
        return node