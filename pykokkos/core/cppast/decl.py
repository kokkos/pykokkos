from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from typing import List, Optional, TYPE_CHECKING, Union

from .node import Node
if TYPE_CHECKING:
    from .expr import DeclRefExpr, Expr
    from .stmt import Stmt


class BuiltinType(Enum):
    INT = "int32_t"
    DOUBLE = "double"
    BOOL = "bool"
    FLOAT = "float"

    INT8 = "int8_t"
    INT16 = "int16_t"
    INT32 = "int32_t"
    INT64 = "int64_t"

    UINT8 = "uint8_t"
    UINT16 = "uint16_t"
    UINT32 = "uint32_t"
    UINT64 = "uint64_t"


class Type(Node):
    """Holds all type information"""

    @property
    @abstractmethod
    def typename(self):
        ...

    @property  # type: ignore
    @abstractmethod
    def is_reference(self) -> bool:
        ...

    @is_reference.setter  # type: ignore
    @abstractmethod
    def is_reference(self, value: bool) -> None:
        ...


class PrimitiveType(Type):
    """Represents types in BuiltInType"""

    def __init__(self, built_in_type: Union[BuiltinType, str]):
        self._type: Union[BuiltinType, str] = built_in_type
        self._is_reference: bool = False

    @property
    def typename(self) -> Union[BuiltinType, str]:
        return self._type

    @property
    def is_reference(self) -> bool:
        return self._is_reference

    @is_reference.setter
    def is_reference(self, value: bool) -> None:
        self._is_reference = value


class ClassType(Type):
    """Represents C++ class types"""

    def __init__(self, typename: str):
        self._typename: str = typename
        self._is_reference: bool = False
        self._template_params: List[Node] = []

    @property
    def typename(self) -> str:
        return self._typename

    @typename.setter
    def typename(self, value: str) -> None:
        self._typename = value

    @property
    def is_reference(self) -> bool:
        return self._is_reference

    @is_reference.setter
    def is_reference(self, value: bool) -> None:
        self._is_reference = value

    @property
    def template_params(self) -> List[Node]:
        return self._template_params

    @template_params.setter
    def template_params(self, value: List[Node]) -> None:
        self._template_params = value

    def add_template_param(self, param: Node) -> None:
        self._template_params.append(param)


class Decl(Node):
    """Base class for all declarations"""

    ...


class TypeDecl(Decl):
    """Represents the declaration of a new type (e.g. a class)"""

    @property
    @abstractmethod
    def typename(self) -> Type:
        ...


class ValueDecl(Decl):
    """Represents the declaration of a value (e.g. variable, function, ...)"""

    @property
    @abstractmethod
    def decltype(self) -> Type:
        ...

    @property
    @abstractmethod
    def declname(self) -> str:
        ...


class RecordDecl(TypeDecl):
    """Represents a class or struct"""

    def __init__(self, typename: ClassType, decls: List[Decl]):
        self._typename: ClassType = typename
        self._decls: List[Decl] = decls
        self.is_definition: bool = False
        self._template_params: List[Node] = []

    @property
    def typename(self) -> ClassType:
        return self._typename

    @property
    def decls(self) -> List[Decl]:
        return self._decls

    def add_decl(self, decl: Decl) -> None:
        self._decls.append(decl)

    @property
    def template_params(self) -> List[Node]:
        return self._template_params

    @template_params.setter
    def template_params(self, value: List[Node]) -> None:
        self._template_params = value

    def add_template_param(self, param: Node) -> None:
        self._template_params.append(param)

class VarDecl(ValueDecl):
    """Represents a variable declaration or definition"""

    def __init__(self, decltype: Type, declname: DeclRefExpr, value: Optional[Expr]):
        self._decltype: Type = decltype
        self._declname: DeclRefExpr = declname
        self._value: Optional[Expr] = value

    @property
    def decltype(self) -> Type:
        return self._decltype

    @property
    def declname(self) -> DeclRefExpr:
        return self._declname

    @property
    def value(self) -> Optional[Expr]:
        return self._value


class FieldDecl(VarDecl):
    """Represents a member of a class or struct"""

    def __init__(self, decltype: Type, declname: DeclRefExpr):
        super().__init__(decltype, declname, None)


class ParmVarDecl(VarDecl):
    """Represents a function parameter declaration"""

    def __init__(self, decltype: Type, declname: DeclRefExpr):
        super().__init__(decltype, declname, None)


class FunctionDecl(ValueDecl):
    """Represents a function declaration or definition"""

    def __init__(self, attributes: str, decltype: Type, declname: str, params: List[ParmVarDecl], body: Optional[Stmt]):
        self._attributes: str = attributes
        self._decltype: Type = decltype
        self._declname: str = declname
        self._params: List[ParmVarDecl] = params
        self._body: Optional[Stmt] = body

    @property
    def decltype(self) -> Type:
        return self._decltype

    @property
    def declname(self) -> str:
        return self._declname

    @property
    def params(self) -> List[ParmVarDecl]:
        return self._params

    @property
    def body(self) -> Optional[Stmt]:
        return self._body

    def add_param(self, param: ParmVarDecl) -> None:
        self._params.append(param)


class MethodDecl(FunctionDecl):
    """Represents a method declaration or definition"""

    def __init__(self, attributes: str, decltype: Type, declname: str, params: List[ParmVarDecl], body: Optional[Stmt]):
        super().__init__(attributes, decltype, declname, params, body)
        self.is_const = False


class ConstructorDecl(MethodDecl):
    """Represents a class constructor declaration or definition"""

    def __init__(self, attributes: str, declname: str, params: List[ParmVarDecl], body: Optional[Stmt]):
        super().__init__(attributes, None, declname, params, body)

    @property
    def decltype(self) -> Type:
        raise NotImplementedError("ConstructorDecl cannot have a decltype")
