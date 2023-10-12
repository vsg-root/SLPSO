from abc import ABC, abstractmethod
from typing import Tuple

class PsoAlgorithm(ABC):
    """
    Abstract base class for Particle Swarm Optimization (PSO) algorithms.
    """

    @abstractmethod
    def optimize(self) -> Tuple:
        """
        Abstract method to run the optimization process.

        This method should execute the optimization algorithm and return the best solution found along with its fitness value.

        Returns:
            Tuple: A tuple containing the best solution found and its fitness value.
        """
        pass
