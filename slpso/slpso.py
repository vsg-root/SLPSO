"""Social Learning PSO optimizer.

Reference
A Social Learning Particle Swarm Optimization Algorithm for Scalable Optimization Authors: Ran Cheng and Yaochu Jin Journal: Information Sciences, Volume 291, Pages 43-60, Year 2015 DOI: 10.1016/j.ins.2014.08.039
---------

"""
from typing import Callable, Tuple
from .algorithm import Algorithm
import numpy as np


class SLPSO(Algorithm):
    def __init__(self,
                 fn: Callable,
                 M: int = 100,
                 alpha: float = 0.5,
                 beta: float = 0.01,
                 n: int = 30,
                 max_fn: int = 20000,
                 show_progress: bool = True,
                 seed: int = 42,
                 lower_bound: float = -1.0,
                 upper_bound: float = 1.0):
        """
        Initialize the SLPSO optimizer.

        Args:
            fn (Callable): The objective function to be minimized.
            M (int, optional): The number of particles in the swarm. Default is 100.
            alpha (float, optional): A constant used in learning probability calculation. Default is 0.5.
            beta (float, optional): A constant used in epsilon calculation. Default is 0.01.
            n (int, optional): The dimensionality of the problem. Default is 30.
            max_fn (int, optional): The maximum number of function evaluations. Default is 20000.
            show_progress (bool, optional): Whether to show progress during optimization. Default is True.
            seed (int, optional): random seed. Default is 42.
            lower_bound (float, optional): The lower bound for particle positions. Default is -1.0.
            upper_bound (float, optional): The upper bound for particle positions. Default is 1.0.
        """
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.m = M + int(np.floor(n / 10))
        self.epsilon = beta * (n / M)
        self.max_evaluations = max_fn
        self.fn = fn
        self.show_progress = show_progress
        self.rng = np.random.default_rng(seed)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self._initialize()

    def _initialize(self):
        """
        Initialize the positions, fitness values, and global best information for the swarm.
        """
        self.positions = self.rng.uniform(self.lower_bound,
                                          self.upper_bound,
                                          size=(self.m, self.n))
        self.fitness_values = self.fn(self.positions)
        self.global_best_index = np.argmin(self.fitness_values)
        self.global_best_position = self.positions[self.global_best_index]
        self.global_best_value = self.fitness_values[self.global_best_index]
        self.previous_deltas = np.zeros((self.m, self.n))

    def learning_probability(self, indices: np.ndarray) -> np.ndarray:
        """
        Calculate the learning probability for each particle.

        Args:
            indices (np.ndarray): An array of particle indices.

        Returns:
            np.ndarray: An array of learning probabilities.
        """
        ceil = np.ceil(self.n / self.M)
        power = self.alpha * np.log10(ceil)
        div = indices / self.m
        return (1 - div) ** power

    def delta_x(self,
                r1: float,
                r2: float,
                r3: float,
                mean_individual: np.ndarray,
                demonstrators: np.ndarray) -> np.ndarray:
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
        n_evaluations = self.m

        # Obtain learning probabilities for each individual
        learning_probabilities = self.learning_probability(np.arange(self.m))

        # Dimensions indices
        dimensions_indices = np.arange(self.n)

        while n_evaluations < self.max_evaluations:
            # Swarm sorting from worst (idx 0) to best (idx -1)
            #   We assume a minimization problem
            sort_indices = np.argsort(self.fitness_values)[::-1]
            self.fitness_values = self.fitness_values[sort_indices]
            self.positions = self.positions[sort_indices]
            self.previous_deltas = self.previous_deltas[sort_indices]

            # Generate random r-values for this iteration
            r1, r2, r3 = self.rng.random(), self.rng.random(), self.rng.random()

            # Initialize with random demonstrators
            demonstrators = self.positions.copy()

            # Use RNG for demonstrator selection in each dimension
            for i in range(self.m - 1):
                demonstrator_indices = self.rng.integers(i + 1,
                                                         self.m,
                                                         size=(self.n,))
                demonstrators[i] = self.positions[demonstrator_indices,
                                                  dimensions_indices]

            # Create a mask for each particle
            mask = self.rng.random(size=self.m) <= learning_probabilities
            mask[-1] = False

            # Obtain the average position of each dimension
            mean_positions = np.mean(self.positions, axis=0)

            # Obtain the delta according (4), (5) and (6)
            deltas = self.delta_x(r1, r2, r3, mean_positions, demonstrators)

            # Update positions
            self.positions = self.positions + mask[:, np.newaxis] * deltas
            self.positions = np.clip(self.positions,
                                     self.lower_bound,
                                     self.upper_bound)

            # Update values of previous deltas to new ones
            self.previous_deltas[mask] = deltas[mask]

            # Obtain new fitness values
            self.fitness_values = self.fn(self.positions)
            n_evaluations += self.m

            # Obtain the index of the best particle
            best_idx = np.argmin(self.fitness_values)

            # Update best known position
            if self.fitness_values[best_idx] < self.global_best_value:
                self.global_best_value = self.fitness_values[best_idx]
                self.global_best_position = self.positions[best_idx]
                self.global_best_index = best_idx

            if self.show_progress:
                print(f"Iteration {n_evaluations + 1}: "
                      f"Global Best Value = {self.global_best_value}")

        if self.show_progress:
            print("Global Best Position:", self.global_best_position)
            print("Global Best Value:", self.global_best_value)

        return self.global_best_position, self.global_best_value
