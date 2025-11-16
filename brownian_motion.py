import math
import random
import time


class BrownianMotion:
    """
    Object-oriented Brownian motion (Wiener process) generator.
    No external packages required.

    dX = drift * dt + volatility * dW
    """

    def __init__(self, drift=0.0, volatility=1.0, dt=0.01, seed=None):
        self.drift = drift
        self.volatility = volatility
        self.dt = dt
        self.current_value = 0.0

        if seed is not None:
            random.seed(seed)

    # -----------------------------
    # Generate a single Brownian increment
    # -----------------------------
    def _increment(self):
        """Generate one normally distributed increment using Box-Muller."""
        u1 = random.random()
        u2 = random.random()

        # Standard normal (Box-Muller)
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        dW = math.sqrt(self.dt) * z
        return self.drift * self.dt + self.volatility * dW

    # -----------------------------
    # Generate an entire path
    # -----------------------------
    def generate_path(self, n_steps):
        """Generate a full Brownian motion path."""
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            path.append(self.current_value)
        return path

    # -----------------------------
    # Generate live streaming values
    # -----------------------------
    def stream(self, interval=1.0):
        """
        Infinite generator that yields a new Brownian value
        every `interval` seconds in real time.
        """
        while True:
            self.current_value += self._increment()
            yield self.current_value
            time.sleep(interval)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    bm = BrownianMotion(drift=0.1, volatility=1.0, dt=0.01, seed=123)

    # Example 1: Generate a static path
    path = bm.generate_path(10)
    print("Static path:")
    for i, val in enumerate(path):
        print(f"{i}: {val:.5f}")

    # Example 2: Stream live values (press Ctrl+C to stop)
    print("\nStreaming live Brownian values:")
    for value in bm.stream(interval=1.0):
        print(f"{value:.5f}")
