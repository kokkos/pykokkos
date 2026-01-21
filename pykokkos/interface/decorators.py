from enum import Enum
from functools import partial


class Decorator(Enum):
    Workload = "workload"
    Functor = "functor"
    WorkUnit = "workunit"
    KokkosClasstype = "classtype"
    KokkosFunction = "function"
    KokkosMain = "main"
    KokkosCallback = "callback"
    Space = "space"

    @staticmethod
    def is_pykokkos_decorator(decorator) -> bool:
        return isinstance(decorator, Decorator)

    @staticmethod
    def is_work_unit(decorator: str) -> bool:
        return decorator == Decorator.WorkUnit.value

    @staticmethod
    def is_kokkos_classtype(decorator: str) -> bool:
        return decorator == Decorator.KokkosClasstype.value

    @staticmethod
    def is_kokkos_function(decorator: str) -> bool:
        return decorator == Decorator.KokkosFunction.value

    @staticmethod
    def is_kokkos_main(decorator: str) -> bool:
        return decorator == Decorator.KokkosMain.value

    @staticmethod
    def is_kokkos_callback(decorator: str) -> bool:
        return decorator == Decorator.KokkosCallback.value

    @staticmethod
    def is_space(decorator: str) -> bool:
        return decorator == Decorator.Space.value

    @staticmethod
    def is_functor(decorator: str) -> bool:
        return decorator == Decorator.Functor.value

    @staticmethod
    def is_workload(decorator: str) -> bool:
        return decorator == Decorator.Workload.value


def functor(func=None, **kwargs):
    if func is None:
        return partial(functor)

    return func


def workunit(func=None, **kwargs):
    if func is None:
        return partial(functor)

    return func


def workload(func=None, **kwargs):
    """
    DEPRECATED: The @workload decorator is no longer supported.

    Please use @workunit decorator instead and refactor your code accordingly.
    The workunit style provides better performance and more flexible code structure.
    """
    raise RuntimeError(
        "The @workload decorator is deprecated and no longer supported. "
        "Please refactor your code to use @workunit decorator instead. "
        "See documentation for migration guide."
    )


def classtype(func):
    return func


def function(func):
    return func


def main(func):
    return func


def callback(func):
    return func
