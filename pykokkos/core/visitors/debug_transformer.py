import ast

from pykokkos.interface import Decorator


# in the Debug ExecSpace we need to wrap all instances of the accumulator
# variable in reduction workloads to work around python's lack of reference
# types for primative numbers
class DebugTransformer(ast.NodeTransformer):
    def __init__(self):
        self.inside_reduction = False

    def visit_FunctionDef(self, node):
        if (
            node.decorator_list
            and Decorator.is_work_unit(node.decorator_list[0].id)
            and len(node.args.args) == 3
        ):
            self.inside_reduction = True
            self.acc = node.args.args[-1].arg
            node.body = list(map(self.visit, node.body))
            self.inside_reduction = False
        return node

    def visit_Name(self, node):
        if self.inside_reduction and node.id == self.acc:
            return ast.Subscript(
                ast.Name(self.acc, ast.Load()),
                ast.Index(ast.Constant(0, None)),
                node.ctx,
            )
        return node
