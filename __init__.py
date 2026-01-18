# datagen/__init__.py
"""
Data Generation Package
A comprehensive toolkit for generating synthetic data for testing and development.
"""

from .distributions import DistributionGenerator
from .stochastic_processes import (
    BrownianMotion,
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    CIRProcess,
    HestonProcess
)
from .time_series import TimeSeriesGenerator
from .regression import RegressionDataGenerator
from .panel import PanelDataGenerator
from .factory import DataFactory

__version__ = "1.0.0"

__all__ = [
    # Main factory
    'DataFactory',

    # Generators
    'DistributionGenerator',
    'TimeSeriesGenerator',
    'RegressionDataGenerator',
    'PanelDataGenerator',

    # Stochastic processes
    'BrownianMotion',
    'GeometricBrownianMotion',
    'OrnsteinUhlenbeck',
    'CIRProcess',
    'HestonProcess',
]