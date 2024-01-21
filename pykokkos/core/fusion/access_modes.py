import ast
from enum import auto, Enum
from typing import Dict, Optional, Set

from pykokkos.core.translators.static import StaticTranslator


class AccessMode(Enum):
    Read = auto()
    Write = auto()
    ReadWrite = auto()


def get_view_access_modes(AST: ast.FunctionDef, view_args: Set[str]) -> Dict[str, AccessMode]:
    AST = StaticTranslator.add_parent_refs(AST)
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

        if not isinstance(node.value, ast.Name): # Skip type annotations
            continue

        name: str = node.value.id
        if name not in view_args:
            continue

        existing_mode: Optional[AccessMode] = access_modes.get(name)
        new_mode: AccessMode

        if isinstance(node.ctx, ast.Load):
            if existing_mode is None:
                new_mode = AccessMode.Read
            elif existing_mode is AccessMode.Write:
                new_mode = AccessMode.ReadWrite
            else:
                new_mode = existing_mode

        if isinstance(node.ctx, ast.Store):
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
