from abc import ABCMeta, abstractmethod

from system import System


class Integrator(metaclass=ABCMeta):
    def __init__(self, s: System):
        self.system = s

    @abstractmethod
    def initial_integrate(self) -> None:
        pass

    @abstractmethod
    def final_integrate(self) -> None:
        pass
