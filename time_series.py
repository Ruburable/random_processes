# time_series.py

import math
import random
from typing import List, Optional, Tuple


class TimeSeriesGenerator:
    """
    Generate various types of time series data patterns.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def _box_muller(self) -> float:
        """Generate standard normal random variable."""
        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def generate_ar(self,
                    n_samples: int,
                    coefficients: List[float],
                    mean: float = 0.0,
                    std: float = 1.0) -> List[float]:
        """
        Generate Autoregressive AR(p) process.
        X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φₚX_{t-p} + ε_t

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        coefficients : list
            AR coefficients [φ₁, φ₂, ..., φₚ]
        mean : float
            Process mean (c)
        std : float
            Innovation standard deviation

        Returns
        -------
        List of AR process values
        """
        p = len(coefficients)
        series = [mean] * p  # Initialize with mean

        for _ in range(n_samples):
            # AR component
            ar_term = sum(coefficients[i] * series[-(i + 1)]
                          for i in range(p))
            # Innovation
            innovation = std * self._box_muller()
            # New value
            value = mean + ar_term + innovation
            series.append(value)

        return series[p:]  # Return only n_samples values

    def generate_ma(self,
                    n_samples: int,
                    coefficients: List[float],
                    mean: float = 0.0,
                    std: float = 1.0) -> List[float]:
        """
        Generate Moving Average MA(q) process.
        X_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θqε_{t-q}

        Parameters
        ----------
        n_samples : int
            Number of samples
        coefficients : list
            MA coefficients [θ₁, θ₂, ..., θq]
        mean : float
            Process mean
        std : float
            Innovation standard deviation

        Returns
        -------
        List of MA process values
        """
        q = len(coefficients)
        innovations = [std * self._box_muller()
                       for _ in range(n_samples + q)]

        series = []
        for t in range(q, n_samples + q):
            ma_term = sum(coefficients[i] * innovations[t - (i + 1)]
                          for i in range(q))
            value = mean + innovations[t] + ma_term
            series.append(value)

        return series

    def generate_arma(self,
                      n_samples: int,
                      ar_coefficients: List[float],
                      ma_coefficients: List[float],
                      mean: float = 0.0,
                      std: float = 1.0) -> List[float]:
        """
        Generate ARMA(p,q) process.

        Parameters
        ----------
        n_samples : int
            Number of samples
        ar_coefficients : list
            AR coefficients
        ma_coefficients : list
            MA coefficients
        mean : float
            Process mean
        std : float
            Innovation standard deviation

        Returns
        -------
        List of ARMA process values
        """
        p = len(ar_coefficients)
        q = len(ma_coefficients)
        warmup = max(p, q) + 50  # Warmup period

        # Generate innovations
        innovations = [std * self._box_muller()
                       for _ in range(n_samples + warmup)]

        series = [mean] * warmup

        for t in range(warmup, n_samples + warmup):
            # AR component
            ar_term = sum(ar_coefficients[i] * (series[t - (i + 1)] - mean)
                          for i in range(p))
            # MA component
            ma_term = sum(ma_coefficients[i] * innovations[t - (i + 1)]
                          for i in range(q))

            value = mean + ar_term + innovations[t] + ma_term
            series.append(value)

        return series[-n_samples:]

    def generate_seasonal(self,
                          n_samples: int,
                          seasonal_period: int,
                          seasonal_strength: float = 1.0,
                          trend: float = 0.0,
                          noise_std: float = 0.1) -> List[float]:
        """
        Generate time series with seasonal pattern.

        Parameters
        ----------
        n_samples : int
            Number of samples
        seasonal_period : int
            Period of seasonality (e.g., 12 for monthly)
        seasonal_strength : float
            Amplitude of seasonal component
        trend : float
            Linear trend coefficient
        noise_std : float
            Noise standard deviation

        Returns
        -------
        List of values with seasonal pattern
        """
        series = []
        for t in range(n_samples):
            # Seasonal component (sinusoidal)
            seasonal = seasonal_strength * math.sin(
                2 * math.pi * t / seasonal_period
            )
            # Trend component
            trend_component = trend * t
            # Noise
            noise = noise_std * self._box_muller()

            value = seasonal + trend_component + noise
            series.append(value)

        return series

    def generate_level_shift(self,
                             n_samples: int,
                             shift_point: int,
                             shift_magnitude: float,
                             noise_std: float = 1.0) -> List[float]:
        """
        Generate time series with level shift.

        Parameters
        ----------
        n_samples : int
            Number of samples
        shift_point : int
            Time point where level shift occurs
        shift_magnitude : float
            Magnitude of shift
        noise_std : float
            Noise standard deviation

        Returns
        -------
        List of values with level shift
        """
        series = []
        for t in range(n_samples):
            level = shift_magnitude if t >= shift_point else 0.0
            noise = noise_std * self._box_muller()
            series.append(level + noise)

        return series

    def generate_garch(self,
                       n_samples: int,
                       omega: float = 0.1,
                       alpha: float = 0.1,
                       beta: float = 0.8,
                       mean: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Generate GARCH(1,1) process with time-varying volatility.

        r_t = μ + σ_t * ε_t
        σ_t² = ω + α * r²_{t-1} + β * σ²_{t-1}

        Parameters
        ----------
        n_samples : int
            Number of samples
        omega : float
            Constant term
        alpha : float
            ARCH coefficient
        beta : float
            GARCH coefficient
        mean : float
            Return mean

        Returns
        -------
        Tuple of (returns, volatilities)
        """
        returns = []
        volatilities = []

        # Initialize
        sigma_sq = omega / (1 - alpha - beta)

        for _ in range(n_samples):
            # Current volatility
            sigma = math.sqrt(sigma_sq)
            volatilities.append(sigma)

            # Generate return
            epsilon = self._box_muller()
            r = mean + sigma * epsilon
            returns.append(r)

            # Update volatility for next period
            sigma_sq = omega + alpha * r ** 2 + beta * sigma_sq

        return returns, volatilities

    def generate_random_walk(self,
                             n_samples: int,
                             drift: float = 0.0,
                             std: float = 1.0,
                             start_value: float = 0.0) -> List[float]:
        """
        Generate random walk with drift.
        X_t = X_{t-1} + μ + ε_t

        Parameters
        ----------
        n_samples : int
            Number of samples
        drift : float
            Drift parameter
        std : float
            Step size standard deviation
        start_value : float
            Starting value

        Returns
        -------
        List of random walk values
        """
        series = [start_value]
        current = start_value

        for _ in range(n_samples - 1):
            step = drift + std * self._box_muller()
            current += step
            series.append(current)

        return series


# Example usage
if __name__ == "__main__":
    ts = TimeSeriesGenerator(seed=42)

    # Example 1: AR(2) process
    print("1. AR(2) process:")
    ar2 = ts.generate_ar(
        n_samples=10,
        coefficients=[0.6, -0.2],
        mean=5.0,
        std=1.0
    )
    print([f"{x:.3f}" for x in ar2])

    # Example 2: Seasonal pattern
    print("\n2. Seasonal time series:")
    seasonal = ts.generate_seasonal(
        n_samples=24,
        seasonal_period=12,
        seasonal_strength=2.0,
        trend=0.1,
        noise_std=0.3
    )
    print([f"{x:.3f}" for x in seasonal[:12]])

    # Example 3: GARCH process
    print("\n3. GARCH(1,1) returns and volatility:")
    returns, vols = ts.generate_garch(
        n_samples=10,
        omega=0.1,
        alpha=0.15,
        beta=0.75
    )
    print("Returns:", [f"{r:.4f}" for r in returns])
    print("Volatility:", [f"{v:.4f}" for v in vols])

    # Example 4: Random walk
    print("\n4. Random walk with drift:")
    rw = ts.generate_random_walk(
        n_samples=15,
        drift=0.1,
        std=1.0,
        start_value=100.0
    )
    print([f"{x:.2f}" for x in rw])