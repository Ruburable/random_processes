# regression.py

import math
import random
from typing import List, Dict, Callable, Optional, Tuple


class RegressionDataGenerator:
    """
    Generate synthetic data for regression analysis.
    Supports linear, nonlinear, heteroskedastic, and correlated designs.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def _box_muller(self) -> float:
        """Generate standard normal random variable."""
        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def linear_regression(self,
                          n_samples: int,
                          coefficients: List[float],
                          intercept: float = 0.0,
                          noise_std: float = 1.0,
                          x_range: Tuple[float, float] = (0.0, 10.0)) -> Dict[str, List]:
        """
        Generate linear regression data: y = β₀ + β₁x₁ + ... + βₚxₚ + ε

        Parameters
        ----------
        n_samples : int
            Number of observations
        coefficients : list
            Regression coefficients [β₁, β₂, ..., βₚ]
        intercept : float
            Intercept term β₀
        noise_std : float
            Standard deviation of noise
        x_range : tuple
            Range for generating X values

        Returns
        -------
        Dict with keys 'X' (list of lists), 'y' (list)
        """
        p = len(coefficients)
        X = []
        y = []

        for _ in range(n_samples):
            # Generate predictors
            x_i = [x_range[0] + (x_range[1] - x_range[0]) * random.random()
                   for _ in range(p)]

            # Compute y
            y_i = intercept + sum(coefficients[j] * x_i[j] for j in range(p))
            y_i += noise_std * self._box_muller()

            X.append(x_i)
            y.append(y_i)

        return {'X': X, 'y': y}

    def nonlinear_regression(self,
                             n_samples: int,
                             func: Callable,
                             n_features: int = 1,
                             noise_std: float = 1.0,
                             x_range: Tuple[float, float] = (0.0, 10.0)) -> Dict[str, List]:
        """
        Generate nonlinear regression data: y = f(x) + ε

        Parameters
        ----------
        n_samples : int
            Number of observations
        func : callable
            Function taking list of features and returning scalar
        n_features : int
            Number of input features
        noise_std : float
            Standard deviation of noise
        x_range : tuple
            Range for generating X values

        Returns
        -------
        Dict with keys 'X', 'y'
        """
        X = []
        y = []

        for _ in range(n_samples):
            x_i = [x_range[0] + (x_range[1] - x_range[0]) * random.random()
                   for _ in range(n_features)]

            y_i = func(x_i) + noise_std * self._box_muller()

            X.append(x_i)
            y.append(y_i)

        return {'X': X, 'y': y}

    def heteroskedastic_regression(self,
                                   n_samples: int,
                                   coefficients: List[float],
                                   intercept: float = 0.0,
                                   noise_func: Optional[Callable] = None,
                                   x_range: Tuple[float, float] = (0.0, 10.0)) -> Dict[str, List]:
        """
        Generate regression data with heteroskedastic errors.
        Variance of error depends on X values.

        Parameters
        ----------
        n_samples : int
            Number of observations
        coefficients : list
            Regression coefficients
        intercept : float
            Intercept
        noise_func : callable, optional
            Function that takes x and returns std of noise
            Default: std proportional to sqrt(x[0])
        x_range : tuple
            Range for X values

        Returns
        -------
        Dict with keys 'X', 'y'
        """
        if noise_func is None:
            noise_func = lambda x: 0.5 + 0.5 * math.sqrt(abs(x[0]))

        p = len(coefficients)
        X = []
        y = []

        for _ in range(n_samples):
            x_i = [x_range[0] + (x_range[1] - x_range[0]) * random.random()
                   for _ in range(p)]

            y_i = intercept + sum(coefficients[j] * x_i[j] for j in range(p))
            y_i += noise_func(x_i) * self._box_muller()

            X.append(x_i)
            y.append(y_i)

        return {'X': X, 'y': y}

    def logistic_regression(self,
                            n_samples: int,
                            coefficients: List[float],
                            intercept: float = 0.0,
                            x_range: Tuple[float, float] = (-3.0, 3.0)) -> Dict[str, List]:
        """
        Generate binary classification data using logistic model.
        P(y=1|x) = 1 / (1 + exp(-(β₀ + β'x)))

        Parameters
        ----------
        n_samples : int
            Number of observations
        coefficients : list
            Logistic coefficients
        intercept : float
            Intercept
        x_range : tuple
            Range for X values

        Returns
        -------
        Dict with keys 'X', 'y' (y is binary 0/1)
        """
        p = len(coefficients)
        X = []
        y = []

        for _ in range(n_samples):
            x_i = [x_range[0] + (x_range[1] - x_range[0]) * random.random()
                   for _ in range(p)]

            # Compute probability
            logit = intercept + sum(coefficients[j] * x_i[j] for j in range(p))
            prob = 1.0 / (1.0 + math.exp(-logit))

            # Generate binary outcome
            y_i = 1 if random.random() < prob else 0

            X.append(x_i)
            y.append(y_i)

        return {'X': X, 'y': y}

    def correlated_features(self,
                            n_samples: int,
                            n_features: int,
                            correlation: float = 0.5) -> List[List[float]]:
        """
        Generate correlated feature matrix.

        Parameters
        ----------
        n_samples : int
            Number of observations
        n_features : int
            Number of features
        correlation : float
            Pairwise correlation between features

        Returns
        -------
        Feature matrix (list of lists)
        """
        X = []

        for _ in range(n_samples):
            # First feature
            z0 = self._box_muller()
            x_i = [z0]

            # Subsequent features correlated with first
            for _ in range(n_features - 1):
                z = self._box_muller()
                x_j = correlation * z0 + math.sqrt(1 - correlation ** 2) * z
                x_i.append(x_j)

            X.append(x_i)

        return X

    def polynomial_regression(self,
                              n_samples: int,
                              degree: int,
                              coefficients: List[float],
                              noise_std: float = 1.0,
                              x_range: Tuple[float, float] = (-3.0, 3.0)) -> Dict[str, List]:
        """
        Generate polynomial regression data.
        y = β₀ + β₁x + β₂x² + ... + βₚx^p + ε

        Parameters
        ----------
        n_samples : int
            Number of observations
        degree : int
            Polynomial degree
        coefficients : list
            Coefficients [β₀, β₁, ..., βₚ]
        noise_std : float
            Noise standard deviation
        x_range : tuple
            Range for x

        Returns
        -------
        Dict with keys 'x', 'y'
        """
        x = []
        y = []

        for _ in range(n_samples):
            x_i = x_range[0] + (x_range[1] - x_range[0]) * random.random()

            # Compute polynomial
            y_i = sum(coefficients[d] * (x_i ** d) for d in range(degree + 1))
            y_i += noise_std * self._box_muller()

            x.append(x_i)
            y.append(y_i)

        return {'x': x, 'y': y}

    def interaction_effects(self,
                            n_samples: int,
                            main_effects: List[float],
                            interaction_effects: List[Tuple[int, int, float]],
                            intercept: float = 0.0,
                            noise_std: float = 1.0) -> Dict[str, List]:
        """
        Generate data with interaction effects.
        y = β₀ + Σβᵢxᵢ + Σγᵢⱼxᵢxⱼ + ε

        Parameters
        ----------
        n_samples : int
            Number of observations
        main_effects : list
            Main effect coefficients
        interaction_effects : list of tuples
            Each tuple: (feature_i, feature_j, coefficient)
        intercept : float
            Intercept
        noise_std : float
            Noise standard deviation

        Returns
        -------
        Dict with keys 'X', 'y'
        """
        p = len(main_effects)
        X = []
        y = []

        for _ in range(n_samples):
            x_i = [random.gauss(0, 1) for _ in range(p)]

            # Main effects
            y_i = intercept + sum(main_effects[j] * x_i[j] for j in range(p))

            # Interaction effects
            for i, j, coef in interaction_effects:
                y_i += coef * x_i[i] * x_i[j]

            y_i += noise_std * self._box_muller()

            X.append(x_i)
            y.append(y_i)

        return {'X': X, 'y': y}


# Example usage
if __name__ == "__main__":
    gen = RegressionDataGenerator(seed=42)

    # Example 1: Simple linear regression
    print("1. Linear regression (y = 2 + 3x + ε):")
    data = gen.linear_regression(
        n_samples=10,
        coefficients=[3.0],
        intercept=2.0,
        noise_std=0.5
    )
    for i in range(5):
        print(f"  x={data['X'][i][0]:.2f}, y={data['y'][i]:.2f}")

    # Example 2: Multiple regression
    print("\n2. Multiple regression (y = 1 + 2x₁ - 3x₂ + ε):")
    data = gen.linear_regression(
        n_samples=10,
        coefficients=[2.0, -3.0],
        intercept=1.0,
        noise_std=1.0
    )
    for i in range(5):
        print(f"  x=[{data['X'][i][0]:.2f}, {data['X'][i][1]:.2f}], y={data['y'][i]:.2f}")

    # Example 3: Nonlinear (quadratic)
    print("\n3. Nonlinear regression (y = x² + ε):")
    data = gen.nonlinear_regression(
        n_samples=10,
        func=lambda x: x[0] ** 2,
        noise_std=0.5,
        x_range=(-3, 3)
    )
    for i in range(5):
        print(f"  x={data['X'][i][0]:.2f}, y={data['y'][i]:.2f}")

    # Example 4: Logistic regression
    print("\n4. Logistic regression (binary y):")
    data = gen.logistic_regression(
        n_samples=10,
        coefficients=[2.0],
        intercept=-1.0
    )
    for i in range(10):
        print(f"  x={data['X'][i][0]:.2f}, y={data['y'][i]}")

    # Example 5: Polynomial regression
    print("\n5. Polynomial regression (y = 1 - 2x + 3x²):")
    data = gen.polynomial_regression(
        n_samples=8,
        degree=2,
        coefficients=[1.0, -2.0, 3.0],
        noise_std=0.5
    )
    for i in range(8):
        print(f"  x={data['x'][i]:.2f}, y={data['y'][i]:.2f}")