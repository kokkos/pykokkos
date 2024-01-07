from typing import List, Union

from .decl import (
    BuiltinType, ClassType, ConstructorDecl, FieldDecl, FunctionDecl, MethodDecl,
    ParmVarDecl, PrimitiveType, RecordDecl, TypeDecl, VarDecl
)
from .expr import (
    ArraySubscriptExpr, AssignOperator, BinaryOperator, BinaryOperatorKind,
    BoolLiteral, BoolOperator, CompoundAssignOperator, CallExpr, CastExpr,
    ConstructExpr, DeclRefExpr, FloatingLiteral, InitListExpr, IntegerLiteral,
    LambdaExpr, MemberCallExpr, MemberExpr, ParenExpr, StringLiteral, UnaryOperator
)
from .node import Node
from .stmt import (
    BreakStmt, CallStmt, CompoundStmt, ContinueStmt, DeclStmt, EmptyStmt, ForStmt,
    IfStmt, ReturnStmt, WhileStmt
)


class Serializer:
    def __init__(self):
        ...

    def serialize(self, node: Node) -> str:
        """Serialize a node"""
        method: str = f"serialize_{node.__class__.__name__}"

        try:
            serializer = getattr(self, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method} has not been implemented")

        return serializer(node)

    def serialize_str(self, node: str) -> str:
        return node

    ####### serialize_Type() #######

    def serialize_ClassType(self, node: ClassType) -> str:
        typename: str = node.typename
        typedecl: str = typename

        if node.template_params:
            template_params: List[str] = [
                self.serialize(t) for t in node.template_params]
            typedecl += "<" + ",".join(template_params) + ">"

        if node.is_reference:
            typedecl += "&"

        return typedecl

    def serialize_PrimitiveType(self, node: PrimitiveType) -> str:
        typename: Union[BuiltinType, str] = node.typename

        typedecl: str
        if isinstance(typename, BuiltinType):
            typedecl = typename.value
        else:
            typedecl = typename

        if node.is_reference:
            typedecl += "&"

        return typedecl

    ####### serialize_Expr() #######

    def serialize_ArraySubscriptExpr(self, node: ArraySubscriptExpr) -> str:
        array: str = self.serialize(node.array)
        subscripts: List[str] = [self.serialize(s) for s in node.subscripts]

        return f"{array}[" + "][".join(subscripts) + "]"

    def serialize_AssignOperator(self, node: AssignOperator) -> str:
        targets: List[str] = [self.serialize(t) for t in node.targets]
        op: str = self.serialize(node.op)
        value: str = self.serialize(node.value)

        return op.join(targets) + f"{op} {value};"

    def serialize_BinaryOperator(self, node: BinaryOperator) -> str:
        lhs: str = self.serialize(node.lhs)
        op: str = self.serialize(node.op)
        rhs: str = self.serialize(node.rhs)

        return f"({lhs} {op} {rhs})"

    def serialize_BoolLiteral(self, node: BoolLiteral) -> str:
        return "true" if node.value else "false"

    def serialize_BoolOperator(self, node: BoolOperator) -> str:
        exprs: List[str] = [self.serialize(e) for e in node.exprs]
        op: str = self.serialize(node.op)

        return op.join(exprs)

    def serialize_CallExpr(self, node: CallExpr) -> str:
        function: str = self.serialize(node.function)
        args: List[str] = [self.serialize(a) for a in node.args]

        return f"{function}(" + ",".join(args) + ")"

    def serialize_CastExpr(self, node: CastExpr) -> str:
        casttype: str = self.serialize(node.casttype)
        expr: str = self.serialize(node.expr)

        return f"({casttype})({expr})"

    def serialize_CompoundAssignOperator(self, node: CompoundAssignOperator) -> str:
        lhs: str = self.serialize(node.lhs)
        op: str = self.serialize(node.op)
        rhs: str = self.serialize(node.rhs)

        return f"{lhs} {op}= {rhs};"

    def serialize_ConstructExpr(self, node: ConstructExpr) -> str:
        constructor: str = self.serialize(node.constructor)
        template_args: List[str] = [
            self.serialize(t) for t in node.template_params]
        args: List[str] = [self.serialize(a) for a in node.args]

        expr: str = constructor
        if template_args:
            expr += "<" + ",".join(template_args) + ">"
        expr += "(" + ",".join(args) + ")"

        return expr

    def serialize_DeclRefExpr(self, node: DeclRefExpr) -> str:
        return node.declname

    def serialize_FloatingLiteral(self, node: FloatingLiteral) -> str:
        return str(node.value)

    def serialize_InitListExpr(self, node: InitListExpr) -> str:
        exprs: List[str] = [self.serialize(e) for e in node.exprs]

        return "{" + ", ".join(exprs) + "}"

    def serialize_IntegerLiteral(self, node: IntegerLiteral) -> str:
        return str(node.value)

    def serialize_LambdaExpr(self, node: LambdaExpr) -> str:
        capture: str = node.capture
        params: List[str] = [self.serialize(p) for p in node.params]
        body: str = self.serialize(node.body)

        expr: str = capture
        expr += "(" + ",".join(params) + ")"
        expr += f"{{ {body} }}"

        return expr

    def serialize_MemberCallExpr(self, node: MemberCallExpr) -> str:
        base: str = self.serialize(node.base)
        function: str = self.serialize(node.function)
        args: List[str] = [self.serialize(a) for a in node.args]

        access: str
        if node.is_pointer:
            access = "->"
        elif node.is_static:
            access = "::"
        else:
            access = "."

        return f"{base}{access}{function}(" + ",".join(args) + ")"

    def serialize_MemberExpr(self, node: MemberExpr) -> str:
        base: str = self.serialize(node.base)
        declname: str = node.declname

        access: str = "." if node.is_pointer is False else "->"

        return f"{base}{access}{declname}"

    def serialize_ParenExpr(self, node: ParenExpr) -> str:
        expr: str = self.serialize(node.expr)

        return f"({expr})"

    def serialize_StringLiteral(self, node: StringLiteral) -> str:
        value: str = str(node.value.encode("raw_unicode_escape"))[2:-1]

        return f"\"{value}\""

    def serialize_UnaryOperator(self, node: UnaryOperator) -> str:
        operand: str = self.serialize(node.operand)
        op: str = self.serialize(node.op)

        return f"{op} ({operand})"

    ####### serialize_Decl() #######

    def serialize_ConstructorDecl(self, node: ConstructorDecl) -> str:
        name: str = node.declname
        params: List[str] = [self.serialize(p) for p in node.params]
        body: str = "" if node.body is None else self.serialize(node.body)

        decl: str = f"{node._attributes} {name}"
        decl += "(" + ", ".join(params) + ")"
        decl += f"{{ {body} }}"
        decl += ";"

        return decl

    def serialize_FieldDecl(self, node: FieldDecl) -> str:
        return self.serialize_VarDecl(node)

    def serialize_FunctionDecl(self, node: FunctionDecl) -> str:
        typename: str = self.serialize(node.decltype)
        funcname: str = node.declname
        params: List[str] = [self.serialize(p) for p in node.params]
        body: str = "" if node.body is None else self.serialize(node.body)

        decl: str = f"{node._attributes} {typename} {funcname}"
        decl += "(" + ", ".join(params) + ")"

        if body:
            decl += f"{{ {body} }}"

        decl += ";"

        return decl

    def serialize_MethodDecl(self, node: MethodDecl) -> str:
        typename: str = self.serialize(node.decltype)
        funcname: str = node.declname
        params: List[str] = [self.serialize(p) for p in node.params]
        body: str = "" if node.body is None else self.serialize(node.body)

        decl: str = f"{node._attributes} {typename} {funcname}"
        decl += "(" + ", ".join(params) + ")"

        if node.is_const:
            decl += "const"

        if body:
            decl += f"{{ {body} }}"

        decl += ";"

        return decl

    def serialize_ParmVarDecl(self, node: ParmVarDecl) -> str:
        return self.serialize_VarDecl(node)

    def serialize_RecordDecl(self, node: RecordDecl) -> str:
        name: str = self.serialize(node.typename)
        body: List[str] = [self.serialize(d) for d in node.decls]

        decl: str = ""
        if node.template_params:
            decl += "template "
            template_params: List[str] = [f"class {self.serialize(t)}" for t in node.template_params]
            decl += "<" + ",".join(template_params) + ">"

        decl += f"struct {name}"

        if node.is_definition:
            decl += "{"
            if body:
                decl += "".join(body)
            decl += "}"

        decl += ";"

        return decl

    def serialize_VarDecl(self, node: VarDecl) -> str:
        typename: str = self.serialize(node.decltype)
        varname: str = self.serialize(node.declname)
        value: str = "" if node.value is None else self.serialize(node.value)

        decl: str = f"{typename} {varname}"
        decl += "" if value == "" else f"= {value}"

        return decl

    ####### serialize_Stmt() #######

    def serialize_BreakStmt(self, node: BreakStmt) -> str:
        return "break;"

    def serialize_CallStmt(self, node: CallStmt) -> str:
        return self.serialize(node.call) + ";"

    def serialize_CompoundStmt(self, node: CompoundStmt) -> str:
        statements: List[str] = [self.serialize(s) for s in node.statements]

        return "".join(statements)

    def serialize_ContinueStmt(self, node: ContinueStmt) -> str:
        return "continue;"

    def serialize_DeclStmt(self, node: DeclStmt) -> str:
        return self.serialize(node.decl) + ";"

    def serialize_EmptyStmt(self, node: EmptyStmt) -> str:
        return "{}"

    def serialize_ForStmt(self, node: ForStmt) -> str:
        init: str = self.serialize(node.init)
        condition: str = self.serialize(node.condition)
        increment: str = self.serialize(node.increment)
        body: str = self.serialize(node.body)

        stmt: str = f"for ({init} {condition}; {increment})"
        stmt += f"{{ {body} }}"

        return stmt

    def serialize_IfStmt(self, node: IfStmt) -> str:
        condition: str = self.serialize(node.condition)
        then_body: str = self.serialize(node.then_body)
        else_body: str = "" if node.else_body is None else self.serialize(
            node.else_body)

        stmt: str = f"if ({condition})"
        stmt += f"{{ {then_body} }}"

        if else_body:
            stmt += f"else {{ {else_body} }}"

        return stmt

    def serialize_ReturnStmt(self, node: ReturnStmt) -> str:
        expr: str = self.serialize(node.expr) if node.expr is not None else ""

        return f"return {expr};"

    def serialize_WhileStmt(self, node: WhileStmt) -> str:
        condition: str = self.serialize(node.condition)
        body: str = self.serialize(node.body)

        stmt: str = f"while({condition}) {{ {body} }}"

        return stmt

    ####### serialize_BinaryOperatorKind() #######

    def serialize_BinaryOperatorKind(self, node: BinaryOperatorKind) -> str:
        return node.value
