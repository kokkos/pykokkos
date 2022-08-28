import ast
from enum import Enum
import sys
from typing import Dict, List, Optional, Set, Union

from pykokkos.core.keywords import Keywords
from pykokkos.core.visitors.visitors_util import (
    math_constants, math_functions, allowed_types, view_dtypes
)
from pykokkos.interface import (
    View, TeamMember, BinSort, ViewType, Timer,
    Iterate, Rank, ScratchView, TeamPolicy
)
from .members import PyKokkosMembers


class ErrorStatus(Enum):
    """
    The status of an error node
    """

    reserved = "reserved"
    undefined = "undefined"


class SymbolsPass:
    """
    Check symbols occurring in PyKokkos code
    """

    def __init__(self, members: PyKokkosMembers, pk_import: str, path: str):
        """
        SymbolsPass constructor

        :param members: the PyKokkos related members of the entity being checked
        :param pk_import: the name of the imported PyKokkos package
        :param path: the path to the file containing AST
        """

        self.path: str = path
        self.reserved_symbols: Set[str] = {keyword.value for keyword in Keywords}
        self.global_symbols: Set[str] = set(dir(sys.modules["pykokkos"]))
        self.global_symbols.update(dir(View))
        self.global_symbols.update(dir(ScratchView))
        self.global_symbols.update(dir(TeamMember))
        self.global_symbols.update(dir(TeamPolicy))
        self.global_symbols.update(dir(BinSort))
        self.global_symbols.update(dir(ViewType))
        self.global_symbols.update(dir(Timer))
        self.global_symbols.update(dir(Iterate))
        self.global_symbols.update(dir(Rank))

        self.global_symbols.update(math_constants)
        self.global_symbols.update(math_functions)
        self.global_symbols.update(allowed_types)
        self.global_symbols.update(view_dtypes)
        self.global_symbols.update(["self", "range", "math", "List", "abs"])
        self.global_symbols.add(pk_import)

        self.global_symbols.update([field.declname for field in members.fields])
        self.global_symbols.update([view.declname for view in members.views])
        self.global_symbols.update([workunit.declname for workunit in members.pk_workunits])
        self.global_symbols.update([function.declname for function in members.pk_functions])

        for classtype, methods in members.classtype_methods.items():
            self.global_symbols.add(classtype.declname)
            self.global_symbols.update([method.declname for method in methods])


    def check_symbols(self, AST: Union[ast.ClassDef, ast.FunctionDef]) -> List[str]:
        """
        Check all symbols in an AST of PyKokkos code

        :param AST: the parent node of AST
        :returns: a list of errors (if any)
        """

        error_nodes: Dict[ast.AST, ErrorStatus] = {}
        local_symbols = self.get_local_symbols(AST)

        for node in ast.walk(AST):
            symbol: Optional[str] = self.get_symbol(node)
            if symbol is None:
                continue

            if symbol not in local_symbols and symbol not in self.global_symbols:
                error_nodes[node] = ErrorStatus.undefined

            if symbol in self.reserved_symbols:
                error_nodes[node] = ErrorStatus.reserved


        return self.get_error_messages(error_nodes)


    def get_local_symbols(self, AST: Union[ast.ClassDef, ast.FunctionDef]) -> Set[str]:
        """
        Get a set of all symbols used in an AST locally
        
        :param AST: the parent node of the AST being checked
        :returns: a set of all local symbols defined in the AST 
        """

        symbols: Set[str] = set()

        for node in ast.walk(AST):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    symbols.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.ctx, ast.Store):
                    symbols.add(node.attr)
            elif isinstance(node, ast.arg):
                symbols.add(node.arg)
            elif isinstance(node, ast.FunctionDef):
                if self.is_nested_call(node):
                    symbols.add(node.name)

        return symbols


    def is_nested_call(self, node: ast.FunctionDef) -> bool:
        args = node.args.args
        if len(args) == 0 or (args[0].arg != "self" and len(node.decorator_list) == 0):
            return True

        return False


    def get_symbol(self, node: ast.AST) -> Optional[str]:
        """
        Get a symbol of a particular node

        :param node: the current node
        :returns: the name of the node if it is of type ast.Name or ast.Attribute
        """

        if self.is_classtype_member(node):
            return None

        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr

        return None


    def is_classtype_member(self, node: ast.AST) -> bool:
        """
        Check if an ast node is a member variable of a classtype

        :param node: the node being checked
        :returns: true or false
        """

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                name: str = node.value.id   
                if name not in self.global_symbols:
                    return True

        return False


    def get_error_messages(self, nodes: Dict[ast.AST, ErrorStatus]) -> List[str]:
        """
        Get the error messages from a list of error nodes

        :param nodes: a map from the nodes containing invalid symbols to the error status
        :returns: the list of error messages
        """

        messages: List[str] = []

        for node, error in nodes.items():
            symbol: Optional[str] = self.get_symbol(node)
            if symbol is None:
                sys.exit("Internal Error")

            message: str = f"File \"{self.path}\", line {node.lineno}:\n {error.value} symbol {symbol} used"
            messages.append(message)

        return messages