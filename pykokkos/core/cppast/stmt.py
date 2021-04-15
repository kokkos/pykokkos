from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from .node import Node
if TYPE_CHECKING:
    from .decl import Decl
    from .expr import CallExpr, Expr


class Stmt(Node):
    """Represents one statement"""

    ...


class BreakStmt(Stmt):
    """Represents the break statement"""

    def __init__(self):
        ...


class CallStmt(Stmt):
    """Represents a call to a function where the return value is unused"""

    def __init__(self, call: CallExpr):
        self._call: CallExpr = call

    @property
    def call(self) -> CallExpr:
        return self._call


class CompoundStmt(Stmt):
    """Represents a group of statements"""

    def __init__(self, statements: List[Stmt]):
        self._statements: List[Stmt] = statements

    @property
    def statements(self) -> List[Stmt]:
        return self._statements

    def add_statement(self, stmt: Stmt) -> None:
        self._statements.append(stmt)


class ContinueStmt(Stmt):
    """Represents the continue statement"""

    def __init__(self):
        ...


class DeclStmt(Stmt):
    """Represents a statement that holds a declaration"""

    def __init__(self, decl: Decl):
        self._decl: Decl = decl

    @property
    def decl(self) -> Decl:
        return self._decl


class EmptyStmt(Stmt):
    """Represents an empty statement (a blank line)"""

    def __init__(self):
        ...


class ForStmt(Stmt):
    """Represents a for statement"""

    def __init__(self, init: Stmt, condition: Expr, increment: Expr, body: Stmt):
        self._init: Stmt = init
        self._condition: Expr = condition
        self._increment: Expr = increment
        self._body: Stmt = body

    @property
    def init(self) -> Stmt:
        return self._init

    @property
    def condition(self) -> Expr:
        return self._condition

    @property
    def increment(self) -> Expr:
        return self._increment

    @property
    def body(self) -> Stmt:
        return self._body


class IfStmt(Stmt):
    """Represents an if/else statement"""

    def __init__(self, condition: Expr, then_body: Stmt, else_body: Optional[Stmt] = None):
        self._condition: Expr = condition
        self._then_body: Stmt = then_body
        self._else_body: Optional[Stmt] = else_body

    @property
    def condition(self) -> Expr:
        return self._condition

    @property
    def then_body(self) -> Stmt:
        return self._then_body

    @property
    def else_body(self) -> Optional[Stmt]:
        return self._else_body


class ReturnStmt(Stmt):
    """Represents a return statement"""

    def __init__(self, expr: Optional[Expr] = None):
        self._expr: Optional[Expr] = expr

    @property
    def expr(self) -> Optional[Expr]:
        return self._expr


class WhileStmt(Stmt):
    """Represents a while statement"""

    def __init__(self, condition: Expr, body: Stmt):
        self._condition: Expr = condition
        self._body: Stmt = body

    @property
    def condition(self) -> Expr:
        return self._condition

    @property
    def body(self) -> Stmt:
        return self._body


class ValueStmt(Stmt):
    """Represents a statement that could possibly have a value and type"""

    ...
