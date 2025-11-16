# ornstein_uhlenbeck.py

import math
import random
import time


class OrnsteinUhlenbeck:
    """
    Ornstein–Uhlenbeck (OU) process generator.

    dX = θ(μ - X) dt + σ dW
    """

    def __init__(self, theta=1.0, mu=0.0, sigma=1.0, dt=0.01, seed=None):
        """
        Parameters
        ----------
        theta : float
            Mean-reversion speed.
        mu : float
            Long-term mean.
        sigma : float
            Volatility term.
        dt : float
            Time step.
        seed : optional
            Random seed.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.current_value = mu  # starting at long-term mean

        if seed is not None:
            random.seed(seed)

    def _increment(self):
        """Generate one OU increment with Gaussian noise."""
        # Standard normal using Box-Muller
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        # Brownian increment
        dW = math.sqrt(self.dt) * z

        # OU update
        dx = self.theta * (self.mu - self.current_value) * self.dt \
             + self.sigma * dW
        return dx

    def generate_path(self, n_steps):
        """Generate a full OU path."""
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        """
        Generator that yields a new OU value every `interval` seconds.
        """
        while True:
            self.current_value += self._increment()
            yield self.current_value
            time.sleep(interval)


# Example usage
if __name__ == "__main__":
    ou = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=0.3, dt=0.01, seed=42)

    path = ou.generate_path(10)
    print("First 10 OU values:")
    for i, val in enumerate(path):
        print(f"{i}: {val:.5f}")

    print("\nStreaming OU values (Ctrl+C to stop):")
    for val in ou.stream(interval=1.0):
        print(f"{val:.5f}")
