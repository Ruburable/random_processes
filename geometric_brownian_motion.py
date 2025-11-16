# geometric_brownian_motion.py

import math
import random
import time

class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM)
    dS = μ S dt + σ S dW
    """

    def __init__(self, mu=0.1, sigma=0.2, S0=1.0, dt=0.01, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.current_value = S0

        if seed is not None:
            random.seed(seed)

    def _increment(self):
        """Generate GBM increment."""

        # Standard normal (Box-Muller)
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        dW = math.sqrt(self.dt) * z

        # Exact discretization (lognormal)
        dS = self.current_value * (
            self.mu * self.dt + self.sigma * dW
        )
        return dS

    def generate_path(self, n_steps):
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        """Stream GBM live values."""
        while True:
            self.current_value += self._increment()
            yield self.current_value
            time.sleep(interval)


if __name__ == "__main__":
        gbm = GeometricBrownianMotion(mu=0.1, sigma=0.3, S0=1.0, dt=0.01, seed=123)
        print(gbm.generate_path(5))

        for val in gbm.stream(interval=1.0):
            print(val)
