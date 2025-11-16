# heston_process.py

import math
import random
import time


class HestonProcess:
    """
    Heston Stochastic Volatility Model

    dS = μ S dt + sqrt(V) S dW1
    dV = κ(θ - V) dt + ξ sqrt(V) dW2

    Corr(dW1, dW2) = ρ
    """

    def __init__(self,
                 mu=0.1,
                 kappa=1.0,
                 theta=0.2,
                 xi=0.3,
                 rho=-0.5,
                 S0=1.0,
                 V0=0.2,
                 dt=0.01,
                 seed=None):

        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt

        self.S = S0
        self.V = V0

        if seed is not None:
            random.seed(seed)

    def _correlated_normals(self):
        """Generate two correlated standard normal variables."""

        # Independent normals
        u1 = random.random()
        u2 = random.random()
        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        u3 = random.random()
        u4 = random.random()
        z2_indep = math.sqrt(-2 * math.log(u3)) * math.cos(2 * math.pi * u4)

        # Correlation structure
        z2 = self.rho * z1 + math.sqrt(1 - self.rho ** 2) * z2_indep

        return z1, z2

    def _increment(self):
        """Perform one Euler step for S and V."""

        z1, z2 = self._correlated_normals()
        dW1 = z1 * math.sqrt(self.dt)
        dW2 = z2 * math.sqrt(self.dt)

        # Update V (variance)
        dV = self.kappa * (self.theta - self.V) * self.dt \
             + self.xi * math.sqrt(max(self.V, 0)) * dW2

        # Update S (price)
        dS = self.S * (self.mu * self.dt + math.sqrt(max(self.V, 0)) * dW1)

        return dS, dV

    def generate_path(self, n_steps):
        S_path = [self.S]
        V_path = [self.V]

        for _ in range(n_steps):
            dS, dV = self._increment()

            self.S += dS
            self.V = max(self.V + dV, 0.0)  # enforce V >= 0

            S_path.append(self.S)
            V_path.append(self.V)

        return S_path, V_path

    def stream(self, interval=1.0):
        while True:
            dS, dV = self._increment()
            self.S += dS
            self.V = max(self.V + dV, 0.0)
            yield (self.S, self.V)
            time.sleep(interval)


if __name__ == "__main__":
    heston = HestonProcess(seed=1)
    S_path, V_path = heston.generate_path(5)
    print("S:", S_path)
    print("V:", V_path)

    for S, V in heston.stream(interval=1.0):
        print(f"S={S:.4f}, V={V:.4f}")
