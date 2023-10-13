from typing import Callable, Optional, Tuple
from .algorithm import PsoAlgorithm
import numpy as np


class SLPSO(PsoAlgorithm):
    def __init__(self, 
                 objective_function: Callable, 
                 M: int = 100, 
                 alpha: float = 0.5, 
                 beta: float = 0.01, 
                 n: int = 30, 
                 max_iterations: int = 20000, 
                 show_progress: bool = True, 
                 rng: Optional[np.random.Generator] = None, 
                 lower_bound: float = -1.0, 
                 upper_bound: float = 1.0
                 ):
        """
        Initialize the SLPSO optimizer.

        Args:
            objective_function (Callable): The objective function to be minimized.
            M (int, optional): The number of particles in the swarm. Default is 100.
            alpha (float, optional): A constant used in learning probability calculation. Default is 0.5.
            beta (float, optional): A constant used in epsilon calculation. Default is 0.01.
            n (int, optional): The dimensionality of the problem. Default is 30.
            max_iterations (int, optional): The maximum number of iterations. Default is 20000.
            show_progress (bool, optional): Whether to show progress during optimization. Default is True.
            rng (Optional[np.random.Generator], optional): A custom random number generator. Default is None.
            lower_bound (float, optional): The lower bound for particle positions. Default is -1.0.
            upper_bound (float, optional): The upper bound for particle positions. Default is 1.0.
        """
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = int(M + np.floor(n / 10))
        self.epsilon = beta * n / M
        self.max_iterations = max_iterations
        self.objective_function = objective_function
        self.show_progress = show_progress
        self.rng = rng if rng is not None else np.random.default_rng()  # Use a custom RNG if provided, otherwise use the default RNG
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self._initialize()

    def _initialize(self):
        """
        Initialize the positions, fitness values, and global best information for the swarm.
        """
        self.positions = self.rng.uniform(self.lower_bound, self.upper_bound, size=(self.m, self.n))
        self.fitness_values = self.objective_function(self.positions)
        self.global_best_index = np.argmin(self.fitness_values)
        self.global_best_position = self.positions[self.global_best_index]
        self.global_best_value = self.fitness_values[self.global_best_index]
        self.previous_deltas = np.ones((self.m, self.n))

    def learning_probability(self, 
                             indices: np.ndarray
                             ) -> np.ndarray:
        """
        Calculate the learning probability for each particle.

        Args:
            indices (np.ndarray): An array of particle indices.

        Returns:
            np.ndarray: An array of learning probabilities.
        """
        power = self.alpha - np.log10(np.ceil(self.n / self.M))
        div = indices / self.m
        return (1 - div) ** power

    def delta_x(self, 
                r1: float, 
                r2: float, 
                r3: float, 
                mean_individual: np.ndarray, 
                demonstrators: np.ndarray
                ) -> np.ndarray:
        """
        Calculate the position update delta for each particle.

        Args:
            r1 (float): Random number.
            r2 (float): Random number.
            r3 (float): Random number.
            mean_individual (np.ndarray): Mean position of the swarm.
            demonstrators (np.ndarray): Demonstrator positions.

        Returns:
            np.ndarray: Position update delta for each particle.
        """
        term1 = r1 * self.previous_deltas
        term2 = r2 * (demonstrators - self.positions)
        term3 = r3 * self.epsilon * (mean_individual - self.positions)
        return term1 + term2 + term3

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Optimize the objective function using SLPSO.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the global best position and its value.
        """
        count = self.m

        while count < self.max_iterations:
            sort_swarm_index = np.argsort(self.fitness_values)
            self.fitness_values = self.fitness_values[sort_swarm_index]
            self.positions = self.positions[sort_swarm_index]
            self.previous_deltas = self.previous_deltas[sort_swarm_index]

            r1, r2, r3 = self.rng.random(), self.rng.random(), self.rng.random()
            probabilities = self.learning_probability(np.arange(self.m))

            demonstrators = self.rng.random(size=(self.m, self.n))  # Use the RNG for generating demonstrators
            for i in range(self.m - 1):
                demonstrator_index = self.rng.integers(i + 1, self.m)  # Use RNG for demonstrator selection
                demonstrators[i] = self.positions[demonstrator_index]

            mean_positions = np.mean(self.positions, axis=0)
            deltas = self.delta_x(r1, r2, r3, mean_positions, demonstrators)
            self.previous_deltas = deltas

            mask = self.rng.random(size=self.m) < probabilities

            self.positions += mask[:, np.newaxis] * deltas
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            self.fitness_values = self.objective_function(self.positions)
            improved_global_indices = np.argmin(self.fitness_values)

            if self.fitness_values[improved_global_indices] < self.global_best_value:
                self.global_best_value = self.fitness_values[improved_global_indices]
                self.global_best_position = self.positions[improved_global_indices]
                self.global_best_index = improved_global_indices

            self.positions[-1] = self.global_best_position
            count += self.m

            if self.show_progress:
                print(f"Iteration {count + 1}: Global Best Value = {self.global_best_value}")

        if self.show_progress:
            print("Global Best Position:", self.global_best_position)
            print("Global Best Value:", self.global_best_value)

        return self.global_best_position, self.global_best_value

if __name__ == "__main__":
    def custom_objective_function(positions: np.ndarray) -> np.ndarray:
        """
        The custom objective function to be minimized.

        Args:
            positions (np.ndarray): An array of particle positions.

        Returns:
            np.ndarray: An array of fitness values.
        """
        return np.sum(positions ** 2, axis=1)

    # Create a custom random number generator
    rng = np.random.default_rng(seed=50)  # Replace 40 with the desired seed

    lower_bound = -5.0  # Set the lower bound
    upper_bound = 5.0   # Set the upper bound

    slpso_optimizer = SLPSO(custom_objective_function, rng=rng, lower_bound=lower_bound, upper_bound=upper_bound, show_progress=False)
    global_best_position, global_best_value = slpso_optimizer.optimize()
    print("Global Best Position:", global_best_position)
    print("Global Best Value:", global_best_value)
