from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from typing import List, TYPE_CHECKING, Union

from .node import Node
from .stmt import ValueStmt
if TYPE_CHECKING:
    from .decl import ParmVarDecl, Type, ValueDecl
    from .stmt import Stmt


class BinaryOperatorKind(Node, Enum):
    """An Enum of binary operators in C++"""
    """From https://github.com/llvm-mirror/clang/blob/master/include/clang/AST/OperationKinds.def"""

    # Binary Operations
    Mul = "*"
    Div = "/"
    Rem = "%"
    Add = "+"
    Sub = "-"
    Shl = "<<"
    Shr = ">>"
    Cmp = "<=>"
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQ = "=="
    NE = "!="
    And = "&"
    Xor = "^"
    Or = "|"
    LAnd = "&&"
    LOr = "||"
    Assign = "="
    MulAssign = "*="
    DivAssign = "/="
    RemAssign = "%="
    AddAssign = "+="
    SubAssign = "-="
    ShlAssign = "<<="
    ShrAssign = ">>="
    AndAssign = "&="
    XorAssign = "^="
    OrAssign = "|="
    Comma = ","

    # Unary Operations
    AddrOf = "&"
    Plus = "+"
    Minus = "-"
    Not = "~"
    LNot = "!"


class Expr(ValueStmt):
    """Represents one expression"""

    ...


class DeclRefExpr(Expr):
    """Represents a reference to a declared variable, function, etc."""

    def __init__(self, declname: str):
        self._declname: str = declname

    @property
    def declname(self) -> str:
        return self._declname

    def add_length(self, length: int) -> None:
        """Used by array types"""

        self._declname += f"[{length}]"

    def __hash__(self) -> int:
        return hash(self._declname)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeclRefExpr):
            return NotImplemented

        return self._declname == other.declname


class ArraySubscriptExpr(Expr):
    """Represents an array subscript expression"""

    def __init__(self, array: DeclRefExpr, subscripts: List[Expr]):
        self._array: DeclRefExpr = array
        self._subscripts: List[Expr] = subscripts

    @property
    def array(self) -> DeclRefExpr:
        return self._array

    @property
    def subscripts(self) -> List[Expr]:
        return self._subscripts


class BinaryOperator(Expr):
    """Represents a binary operator"""

    def __init__(self, lhs: Expr, rhs: Expr, op: BinaryOperatorKind):
        self._lhs: Expr = lhs
        self._rhs: Expr = rhs
        self._op: BinaryOperatorKind = op

    @property
    def lhs(self) -> Expr:
        return self._lhs

    @property
    def rhs(self) -> Expr:
        return self._rhs

    @property
    def op(self) -> BinaryOperatorKind:
        return self._op


class BoolLiteral(Expr):
    """Represents a boolean literal (true/false)"""

    def __init__(self, value: bool):
        self._value: bool = value

    @property
    def value(self) -> bool:
        return self._value


class BoolOperator(Expr):
    """Represents a chained collection of boolean expressions"""

    def __init__(self, exprs: List[Expr], op: BinaryOperatorKind):
        self._exprs: List[Expr] = exprs
        self._op: BinaryOperatorKind = op

    @property
    def exprs(self) -> List[Expr]:
        return self._exprs

    @property
    def op(self) -> BinaryOperatorKind:
        return self._op


class CallExpr(Expr):
    """Represents a function call"""

    def __init__(self, function: DeclRefExpr, args: List[Expr]):
        self._function: DeclRefExpr = function
        self._args: List[Expr] = args

    @property
    def function(self) -> DeclRefExpr:
        return self._function

    @property
    def args(self) -> List[Expr]:
        return self._args

    def add_arg(self, arg: Expr) -> None:
        self._args.append(arg)


class CastExpr(Expr):
    """Represents a C-style cast expression"""

    def __init__(self, casttype: Type, expr: Expr):
        self._casttype: Type = casttype
        self._expr: Expr = expr

    @property
    def casttype(self) -> Type:
        return self._casttype

    @property
    def expr(self) -> Expr:
        return self._expr


class ConstructExpr(Expr):
    """Represents a call to a constructor"""

    def __init__(self, constructor: DeclRefExpr, args: List[Expr]):
        self._constructor: DeclRefExpr = constructor
        self._args: List[Expr] = args
        self._template_params: List[Node] = []

    @property
    def constructor(self) -> DeclRefExpr:
        return self._constructor

    @property
    def args(self) -> List[Expr]:
        return self._args

    def add_arg(self, arg: Expr) -> None:
        self._args.append(arg)

    @property
    def template_params(self) -> List[Node]:
        return self._template_params

    def add_template_param(self, param: Node) -> None:
        self._template_params.append(param)


class FloatingLiteral(Expr):
    """Represents a floating point literal"""

    def __init__(self, value: float):
        self._value: float = value

    @property
    def value(self) -> float:
        return self._value


class InitListExpr(Expr):
    """Represents a C++ initializer list"""

    def __init__(self, exprs: List[Expr]):
        self._exprs: List[Expr] = exprs

    @property
    def exprs(self) -> List[Expr]:
        return self._exprs

    def add_expr(self, expr: Expr) -> None:
        self._exprs.append(expr)


class IntegerLiteral(Expr):
    """Represents an integer literal"""

    def __init__(self, value: int):
        self._value: int = value

    @property
    def value(self) -> int:
        return self._value


class LambdaExpr(Expr):
    """Represents a C++ lambda expression"""

    def __init__(self, capture: str, params: List[ParmVarDecl], body: Stmt):
        self._capture: str = capture
        self._params: List[ParmVarDecl] = params
        self._body: Stmt = body

    @property
    def capture(self) -> str:
        return self._capture

    @property
    def params(self) -> List[ParmVarDecl]:
        return self._params

    @property
    def body(self) -> Stmt:
        return self._body

    def add_param(self, param: ParmVarDecl) -> None:
        self._params.append(param)


class ParenExpr(Expr):
    """Represents a parenthesized expression"""

    def __init__(self, expr: Expr):
        self._expr: Expr = expr

    @property
    def expr(self) -> Expr:
        return self._expr


class StringLiteral(Expr):
    """Represents a string literal"""

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value


class UnaryOperator(Expr):
    """Represents a unary operator"""

    def __init__(self, operand: Expr, op: BinaryOperatorKind):
        self._operand: Expr = operand
        self._op: BinaryOperatorKind = op

    @property
    def operand(self) -> Expr:
        return self._operand

    @property
    def op(self) -> BinaryOperatorKind:
        return self._op


class AssignOperator(BinaryOperator):
    """Represents an assignment to a variable"""

    def __init__(self, targets: List[Expr], value: Expr, op: BinaryOperatorKind):
        self._targets: List[Expr] = targets
        self._value: Expr = value
        self._op: BinaryOperatorKind = op

    @property
    def targets(self) -> List[Expr]:
        return self._targets

    @property
    def value(self) -> Expr:
        return self._value

    @property
    def op(self) -> BinaryOperatorKind:
        return self._op


class CompoundAssignOperator(BinaryOperator):
    """Represents compound assignments (e.g. +=, -=, etc.)"""

    def __init__(self, lhs: Expr, rhs: Expr, op: BinaryOperatorKind):
        self._lhs: Expr = lhs
        self._rhs: Expr = rhs
        self._op: BinaryOperatorKind = op

    @property
    def lhs(self) -> Expr:
        return self._lhs

    @property
    def rhs(self) -> Expr:
        return self._rhs

    @property
    def op(self) -> BinaryOperatorKind:
        return self._op


class MemberCallExpr(CallExpr):
    """Represents a call to a member function """

    def __init__(self, base: DeclRefExpr, function: DeclRefExpr, args: List[Expr]):
        self._base: DeclRefExpr = base
        self.is_pointer = False
        self.is_static = False
        super().__init__(function, args)

    @property
    def base(self) -> DeclRefExpr:
        return self._base


class MemberExpr(DeclRefExpr):
    """Represents the expression for accessing a member of a class"""

    def __init__(self, base: DeclRefExpr, declname: str):
        self._base: DeclRefExpr = base
        self.is_pointer = False
        super().__init__(declname)

    @property
    def base(self) -> DeclRefExpr:
        return self._base
