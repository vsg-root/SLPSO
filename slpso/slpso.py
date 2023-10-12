import numpy as np
from .Algorithm import PsoAlgorithm


class SLPSO(PsoAlgorithm):
    def __init__(self, 
                 objective_function, 
                 M: int = 100, 
                 alpha: float = 0.5, 
                 beta: float = 0.01, 
                 n: int = 30, 
                 max_iterations: int = 20000, 
                 show_progress: bool = True, 
                 random_seed: int = None
                 ):
        """
        Initialize the SLPSO optimizer.

        Parameters:
        - objective_function: The objective function to be minimized.
        - M: Number of swarm particles.
        - alpha: Learning power parameter.
        - beta: Learning power parameter.
        - n: Number of dimensions.
        - max_iterations: Maximum number of iterations.
        - show_progress: If True, display progress during optimization.
        - random_seed: Seed for random number generation (for reproducibility).
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
        self.random_seed = random_seed
        self._initialize()

    def _initialize(self):
        """
        Initialize the optimizer, including setting the random seed if provided.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.positions = np.random.rand(self.m, self.n)
        self.fitness_values = self.objective_function(self.positions)
        self.global_best_index = np.argmin(self.fitness_values)
        self.global_best_position = self.positions[self.global_best_index]
        self.global_best_value = self.fitness_values[self.global_best_index]
        self.previous_deltas = np.ones((self.m, self.n))

    def learning_probability(self, indices: np.ndarray) -> np.ndarray:
        """
        Calculate learning probabilities for particles.

        Parameters:
        - indices: Particle indices.

        Returns:
        - Learning probabilities for each particle.
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
        Calculate delta_x values for particles.

        Parameters:
        - r1: Random value for term1.
        - r2: Random value for term2.
        - r3: Random value for term3.
        - mean_individual: Mean of particle positions.
        - demonstrators: Positions of demonstrators.

        Returns:
        - Delta_x values for particles.
        """
        term1 = r1 * self.previous_deltas
        term2 = r2 * (demonstrators - self.positions)
        term3 = r3 * self.epsilon * (mean_individual - self.positions)
        return term1 + term2 + term3

    def optimize(self) -> tuple:
        """
        Optimize the objective function using SLPSO.

        Returns:
        - A tuple containing the global best position and its value.
        """
        count = 0

        while count < self.max_iterations:
            # Sort the swarm
            sort_swarm_index = np.argsort(self.fitness_values)
            self.fitness_values = self.fitness_values[sort_swarm_index]
            self.positions = self.positions[sort_swarm_index]
            self.previous_deltas = self.previous_deltas[sort_swarm_index]

            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
            probabilities = self.learning_probability(np.arange(self.m))

            # Assign demonstrators
            demonstrators = np.full((self.m, self.n), -1, dtype=float)
            for i in range(self.m - 1):
                demonstrator_index = np.random.randint(i + 1, self.m)
                demonstrators[i] = self.positions[demonstrator_index]

            mean_positions = np.mean(self.positions, axis=0)
            deltas = self.delta_x(r1, r2, r3, mean_positions, demonstrators)
            self.previous_deltas = deltas

            mask = np.random.rand(self.m) < probabilities

            self.positions += mask[:, np.newaxis] * deltas

            self.fitness_values = self.objective_function(self.positions)
            improved_global_indices = np.argmin(self.fitness_values)

            if self.fitness_values[improved_global_indices] < self.global_best_value:
                self.global_best_value = self.fitness_values[improved_global_indices]
                self.global_best_position = self.positions[improved_global_indices]
                self.global_best_index = improved_global_indices


            self.positions[-1] = self.global_best_position # Does not change the best particle fitness value

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
        Example objective function to be minimized.

        Parameters:
        - positions: Particle positions.

        Returns:
        - Array of objective function values for each particle.
        """
        return np.sum(positions ** 2, axis=1)

    # Define a desired random seed to control randomness
    random_seed = 40

    slpso_optimizer = SLPSO(custom_objective_function, random_seed=random_seed)
    global_best_position, global_best_value = slpso_optimizer.optimize()
    print("Global Best Position:", global_best_position)
    print("Global Best Value:", global_best_value)

