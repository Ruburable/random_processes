# datagen/stochastic_processes.py
"""All stochastic processes in one module."""

import math
import random
import time


class BrownianMotion:
    """
    Brownian motion (Wiener process): dX = drift * dt + volatility * dW
    """
    def __init__(self, drift=0.0, volatility=1.0, dt=0.01, seed=None):
        self.drift = drift
        self.volatility = volatility
        self.dt = dt
        self.current_value = 0.0
        if seed is not None:
            random.seed(seed)

    def _increment(self):
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        dW = math.sqrt(self.dt) * z
        return self.drift * self.dt + self.volatility * dW

    def generate_path(self, n_steps):
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        while True:
            self.current_value += self._increment()
            yield self.current_value
            time.sleep(interval)


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion: dS = μ S dt + σ S dW
    """
    def __init__(self, mu=0.1, sigma=0.2, S0=1.0, dt=0.01, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.current_value = S0
        if seed is not None:
            random.seed(seed)

    def _increment(self):
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        dW = math.sqrt(self.dt) * z
        return self.current_value * (self.mu * self.dt + self.sigma * dW)

    def generate_path(self, n_steps):
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        while True:
            self.current_value += self._increment()
            yield self.current_value
            time.sleep(interval)


class OrnsteinUhlenbeck:
    """
    Ornstein–Uhlenbeck process: dX = θ(μ - X) dt + σ dW
    """
    def __init__(self, theta=1.0, mu=0.0, sigma=1.0, dt=0.01, seed=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.current_value = mu
        if seed is not None:
            random.seed(seed)

    def _increment(self):
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        dW = math.sqrt(self.dt) * z
        return self.theta * (self.mu - self.current_value) * self.dt + self.sigma * dW

    def generate_path(self, n_steps):
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        while True:
            self.current_value += self._increment()
            yield self.current_value
            time.sleep(interval)


class CIRProcess:
    """
    Cox–Ingersoll–Ross process: dX = θ(μ - X) dt + σ sqrt(X) dW
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
        u1, u2 = random.random(), random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        dW = z * math.sqrt(self.dt)
        x = self.current_value
        return self.theta * (self.mu - x) * self.dt + self.sigma * math.sqrt(max(x, 0)) * dW

    def generate_path(self, n_steps):
        path = [self.current_value]
        for _ in range(n_steps):
            self.current_value += self._increment()
            self.current_value = max(self.current_value, 0.0)
            path.append(self.current_value)
        return path

    def stream(self, interval=1.0):
        while True:
            self.current_value += self._increment()
            self.current_value = max(self.current_value, 0.0)
            yield self.current_value
            time.sleep(interval)


class HestonProcess:
    """
    Heston Stochastic Volatility Model:
    dS = μ S dt + sqrt(V) S dW1
    dV = κ(θ - V) dt + ξ sqrt(V) dW2
    """
    def __init__(self, mu=0.1, kappa=1.0, theta=0.2, xi=0.3, rho=-0.5,
                 S0=1.0, V0=0.2, dt=0.01, seed=None):
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
        u1, u2 = random.random(), random.random()
        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        u3, u4 = random.random(), random.random()
        z2_indep = math.sqrt(-2 * math.log(u3)) * math.cos(2 * math.pi * u4)
        z2 = self.rho * z1 + math.sqrt(1 - self.rho ** 2) * z2_indep
        return z1, z2

    def _increment(self):
        z1, z2 = self._correlated_normals()
        dW1 = z1 * math.sqrt(self.dt)
        dW2 = z2 * math.sqrt(self.dt)
        dV = self.kappa * (self.theta - self.V) * self.dt + self.xi * math.sqrt(max(self.V, 0)) * dW2
        dS = self.S * (self.mu * self.dt + math.sqrt(max(self.V, 0)) * dW1)
        return dS, dV

    def generate_path(self, n_steps):
        S_path = [self.S]
        V_path = [self.V]
        for _ in range(n_steps):
            dS, dV = self._increment()
            self.S += dS
            self.V = max(self.V + dV, 0.0)
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