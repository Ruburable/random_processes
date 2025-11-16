# cir_process.py

import math
import random
import time

class CIRProcess:
    """
    Cox–Ingersoll–Ross (CIR) process
    dX = θ(μ - X) dt + σ sqrt(X) dW
    Keeps X >= 0 (if Feller condition holds)
    """

    def __init__(self, theta=1.0, mu=0.5, sigma=0.3, X0=0.5, dt=0.01, seed=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.current_value = X0

        if seed is not None:
            random.seed(seed)

    def _increment(self):
        """One CIR step with sqrt diffusion."""

        # Standard normal
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        dW = z * math.sqrt(self.dt)

        # Euler discretization (simple + common)
        x = self.current_value
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * math.sqrt(max(x, 0)) * dW

        return dx

    def generate_path(self, n_steps):
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            self.current_value = max(self.current_value, 0.0)  # enforce positivity
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        while True:
            self.current_value += self._increment()
            self.current_value = max(self.current_value, 0.0)
            yield self.current_value
            time.sleep(interval)


if __name__ == "__main__":
    cir = CIRProcess(theta=1.0, mu=0.5, sigma=0.3, X0=0.2, dt=0.01, seed=42)
    print(cir.generate_path(5))

    for val in cir.stream(interval=1.0):
        print(val)
