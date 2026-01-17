# distributions.py

import math
import random
from typing import List, Optional


class DistributionGenerator:
    """
    Generate samples from various statistical distributions.
    No external packages required - uses only built-in math and random.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    # ========================
    # Continuous Distributions
    # ========================

    def normal(self, n_samples: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
        """Generate from Normal/Gaussian distribution."""
        samples = []
        for _ in range(n_samples):
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            samples.append(mu + sigma * z)
        return samples

    def lognormal(self, n_samples: int, mu: float = 0.0, sigma: float = 1.0) -> List[float]:
        """Generate from Lognormal distribution."""
        return [math.exp(x) for x in self.normal(n_samples, mu, sigma)]

    def exponential(self, n_samples: int, rate: float = 1.0) -> List[float]:
        """Generate from Exponential distribution."""
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(-math.log(u) / rate)
        return samples

    def uniform(self, n_samples: int, a: float = 0.0, b: float = 1.0) -> List[float]:
        """Generate from Uniform distribution."""
        return [a + (b - a) * random.random() for _ in range(n_samples)]

    def gamma(self, n_samples: int, shape: float, scale: float = 1.0) -> List[float]:
        """
        Generate from Gamma distribution using Marsaglia-Tsang method.
        Works well for shape >= 1.
        """
        samples = []

        for _ in range(n_samples):
            if shape < 1:
                # For shape < 1, use shape + 1 and transform
                sample = self._gamma_marsaglia(shape + 1) * (random.random() ** (1.0 / shape))
            else:
                sample = self._gamma_marsaglia(shape)

            samples.append(sample * scale)

        return samples

    def _gamma_marsaglia(self, shape: float) -> float:
        """Marsaglia-Tsang method for Gamma distribution (shape >= 1)."""
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)

        while True:
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

            v = (1.0 + c * z) ** 3

            if z > -1.0 / c and math.log(random.random()) < 0.5 * z ** 2 + d - d * v + d * math.log(v):
                return d * v

    def beta(self, n_samples: int, alpha: float, beta_param: float) -> List[float]:
        """Generate from Beta distribution using ratio of Gammas."""
        samples = []
        for _ in range(n_samples):
            x = self._gamma_marsaglia(alpha if alpha >= 1 else alpha + 1)
            y = self._gamma_marsaglia(beta_param if beta_param >= 1 else beta_param + 1)

            if alpha < 1:
                x *= random.random() ** (1.0 / alpha)
            if beta_param < 1:
                y *= random.random() ** (1.0 / beta_param)

            samples.append(x / (x + y))

        return samples

    def cauchy(self, n_samples: int, location: float = 0.0, scale: float = 1.0) -> List[float]:
        """Generate from Cauchy distribution."""
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(location + scale * math.tan(math.pi * (u - 0.5)))
        return samples

    def pareto(self, n_samples: int, alpha: float = 1.0, xm: float = 1.0) -> List[float]:
        """Generate from Pareto distribution."""
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(xm / (u ** (1.0 / alpha)))
        return samples

    def weibull(self, n_samples: int, shape: float, scale: float = 1.0) -> List[float]:
        """Generate from Weibull distribution."""
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(scale * (-math.log(u)) ** (1.0 / shape))
        return samples

    def student_t(self, n_samples: int, df: float) -> List[float]:
        """Generate from Student's t-distribution."""
        samples = []
        for _ in range(n_samples):
            # Normal
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

            # Chi-square with df degrees of freedom
            chi_sq = sum(self.normal(int(df), 0, 1)[i] ** 2 for i in range(int(df)))

            t = z / math.sqrt(chi_sq / df)
            samples.append(t)

        return samples

    # ========================
    # Discrete Distributions
    # ========================

    def poisson(self, n_samples: int, lam: float) -> List[int]:
        """Generate from Poisson distribution."""
        samples = []
        for _ in range(n_samples):
            L = math.exp(-lam)
            k = 0
            p = 1.0

            while p > L:
                k += 1
                p *= random.random()

            samples.append(k - 1)

        return samples

    def binomial(self, n_samples: int, n: int, p: float) -> List[int]:
        """Generate from Binomial distribution."""
        samples = []
        for _ in range(n_samples):
            successes = sum(1 for _ in range(n) if random.random() < p)
            samples.append(successes)
        return samples

    def geometric(self, n_samples: int, p: float) -> List[int]:
        """Generate from Geometric distribution (number of trials until first success)."""
        samples = []
        for _ in range(n_samples):
            u = random.random()
            samples.append(int(math.ceil(math.log(u) / math.log(1 - p))))
        return samples

    def negative_binomial(self, n_samples: int, r: int, p: float) -> List[int]:
        """Generate from Negative Binomial distribution."""
        samples = []
        for _ in range(n_samples):
            # Sum of r geometric random variables
            value = sum(self.geometric(1, p)[0] for _ in range(r))
            samples.append(value)
        return samples

    # ========================
    # Multivariate
    # ========================

    def multivariate_normal(self, n_samples: int, mean: List[float],
                            cov: List[List[float]]) -> List[List[float]]:
        """
        Generate from Multivariate Normal using Cholesky decomposition.

        Parameters
        ----------
        n_samples : int
            Number of samples
        mean : list
            Mean vector
        cov : list of lists
            Covariance matrix

        Returns
        -------
        List of multivariate normal samples
        """
        dim = len(mean)

        # Cholesky decomposition
        L = self._cholesky(cov)

        samples = []
        for _ in range(n_samples):
            # Generate independent standard normals
            z = self.normal(dim, 0, 1)

            # Transform: x = mean + L @ z
            x = mean.copy()
            for i in range(dim):
                for j in range(i + 1):
                    x[i] += L[i][j] * z[j]

            samples.append(x)

        return samples

    def _cholesky(self, A: List[List[float]]) -> List[List[float]]:
        """Cholesky decomposition of positive definite matrix."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))

                if i == j:
                    L[i][j] = math.sqrt(A[i][i] - s)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]

        return L


# Example usage
if __name__ == "__main__":
    gen = DistributionGenerator(seed=42)

    print("1. Normal(0, 1):")
    print([f"{x:.3f}" for x in gen.normal(10)])

    print("\n2. Exponential(rate=2):")
    print([f"{x:.3f}" for x in gen.exponential(10, rate=2.0)])

    print("\n3. Gamma(shape=2, scale=2):")
    print([f"{x:.3f}" for x in gen.gamma(10, shape=2.0, scale=2.0)])

    print("\n4. Poisson(λ=5):")
    print(gen.poisson(15, lam=5.0))

    print("\n5. Binomial(n=10, p=0.3):")
    print(gen.binomial(15, n=10, p=0.3))

    print("\n6. Multivariate Normal:")
    mean = [0.0, 1.0]
    cov = [[1.0, 0.5], [0.5, 2.0]]
    samples = gen.multivariate_normal(5, mean, cov)
    for s in samples:
        print(f"  [{s[0]:.3f}, {s[1]:.3f}]")