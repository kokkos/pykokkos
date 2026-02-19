from enum import Enum
from functools import partial


class Decorator(Enum):
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


def functor(func=None, **kwargs):
    if func is None:
        return partial(functor)

    return func


def workunit(func=None, *, scratch=None, **kwargs):
    """
    Decorator for PyKokkos workunits.

    :param func: the function being decorated
    :param scratch: optional list of tuples specifying scratch memory allocation
    """
    if func is None:
        return partial(workunit, scratch=scratch, **kwargs)

    # Store scratch specification as function attribute
    if scratch is not None:
        func._pk_scratch = scratch

    return func


def classtype(func):
    return func


def function(func):
    return func


def main(func):
    return func


def callback(func):
    return func
