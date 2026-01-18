# DataGen - Synthetic Data Generation Toolkit

A zero-dependency Python toolkit for generating synthetic data. Built for dashboard prototyping, testing, and development.

## Features

- **Dashboard-Ready Data**: Pre-built generators for common business use cases
- **Statistical Distributions**: Normal, gamma, beta, Poisson, and 12 more
- **Time Series**: AR, MA, ARMA, GARCH, seasonal patterns
- **Stochastic Processes**: Brownian motion, GBM, Ornstein-Uhlenbeck, CIR, Heston
- **Structured Data**: Regression, panel/longitudinal, custom schemas
- **Financial Data**: Stock prices, portfolios, OHLCV data
- **IoT/Sensor Data**: Realistic sensor readings with anomaly detection
- **Business Metrics**: Sales, revenue, user analytics, conversion funnels
- **Zero Dependencies**: Pure Python standard library

## Installation

### Installation

```bash
# Option 1: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/datagen"

# Option 2: Install as editable package
cd /path/to/datagen
pip install -e .

# Option 3: Symlink to site-packages
ln -s /path/to/datagen ~/.local/lib/python3.x/site-packages/datagen
```

### Basic Example

```python
from datagen import DataFactory

# Create factory with seed for reproducibility
factory = DataFactory(seed=42)

# Generate 100 days of stock prices
stocks = factory.stock_prices(n_days=100, ticker='AAPL', start_price=150.0)

# Generate sales transactions
sales = factory.sales_data(n_records=1000, n_products=20, n_regions=5)

# Generate user engagement metrics
users = factory.user_metrics(n_users=500, n_days=30)

# That's it! Data is ready for your dashboard
```

## Usage Examples

### 1. Financial Dashboard

```python
from datagen import DataFactory

factory = DataFactory(seed=42)

# Stock OHLCV data
stock_data = factory.stock_prices(
    n_days=252,
    ticker='AAPL',
    start_price=150.0,
    volatility=0.25,
    drift=0.0005
)
# Returns: [{day, ticker, open, high, low, close, volume}, ...]

# Multi-asset portfolio
portfolio = factory.portfolio_data(n_assets=10, n_days=252)
# Returns: [{day, asset, price, weight, return}, ...]
```

### 2. Sales & Revenue Dashboard

```python
# Transaction-level sales data
sales = factory.sales_data(
    n_records=10000,
    n_products=50,
    n_regions=10
)
# Returns: [{transaction_id, day, product, region, quantity, price, revenue}, ...]

# Daily revenue time series with seasonality
revenue = factory.revenue_timeseries(
    n_days=365,
    base_revenue=50000,
    growth_rate=0.001,
    seasonality=True
)
# Returns: [{day, revenue, cumulative_revenue}, ...]
```

### 3. User Analytics Dashboard

```python
# User engagement metrics
users = factory.user_metrics(n_users=1000, n_days=30)
# Returns: [{user_id, day, sessions, page_views, time_spent, converted, engagement_level}, ...]

# Conversion funnel data
funnel = factory.funnel_data(
    n_visitors=10000,
    conversion_rates=[1.0, 0.3, 0.6, 0.4]  # visit, signup, activate, purchase
)
# Returns: [{visitor_id, stage, stage_order, reached}, ...]
```

### 4. IoT/Sensor Dashboard

```python
# Sensor readings with anomaly detection
sensors = factory.sensor_data(
    n_sensors=50,
    n_readings=10000,
    sensor_type='temperature'  # or 'pressure', 'humidity'
)
# Returns: [{sensor_id, timestamp, value, anomaly, sensor_type}, ...]
```

### 5. Custom Table Schema

```python
# Define your schema
schema = {
    'employee_id': 'int',
    'department': 'category',
    'salary': 'normal',
    'performance_score': 'uniform',
    'years_experience': 'int',
    'is_remote': 'bool'
}

# Generate data matching schema
employees = factory.table_data(n_rows=1000, schema=schema)
# Returns: [{employee_id, department, salary, performance_score, ...}, ...]
```

### 6. Machine Learning Datasets

```python
# Classification dataset with controllable separability
ml_data = factory.classification_data(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    separability=0.85  # Higher = easier to classify
)
# Returns: {'X': [[...], [...]], 'y': [0, 1, 2, ...]}
```

## Advanced Usage

### Using Individual Generators

```python
from datagen import (
    GeometricBrownianMotion,
    TimeSeriesGenerator,
    DistributionGenerator,
    RegressionDataGenerator
)

# Stochastic process
gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2, S0=100.0, dt=0.01, seed=42)
prices = gbm.generate_path(n_steps=1000)

# Time series
ts = TimeSeriesGenerator(seed=42)
ar_process = ts.generate_ar(
    n_samples=500,
    coefficients=[0.8, -0.2],  # AR(2)
    mean=0.0,
    std=1.0
)

# Statistical distributions
dist = DistributionGenerator(seed=42)
normal_samples = dist.normal(n_samples=1000, mu=0, sigma=1)
gamma_samples = dist.gamma(n_samples=1000, shape=2.0, scale=2.0)

# Regression data
reg = RegressionDataGenerator(seed=42)
data = reg.linear_regression(
    n_samples=100,
    coefficients=[2.5, -1.3, 0.8],
    intercept=5.0,
    noise_std=1.0
)
```

### Streaming Data

```python
# Real-time streaming (useful for live demos)
from datagen import OrnsteinUhlenbeck

ou = OrnsteinUhlenbeck(theta=1.0, mu=0.0, sigma=0.3, dt=0.01)

# Stream values every second
for value in ou.stream(interval=1.0):
    print(f"Current value: {value:.4f}")
    # Press Ctrl+C to stop
```

## Module Reference

### High-Level API
- **`DataFactory`**: One-stop shop for dashboard-ready data generation

### Generators
- **`DistributionGenerator`**: 15+ statistical distributions (normal, gamma, beta, Poisson, etc.)
- **`TimeSeriesGenerator`**: AR, MA, ARMA, GARCH, seasonal patterns, level shifts
- **`RegressionDataGenerator`**: Linear, nonlinear, logistic, polynomial, with interactions
- **`PanelDataGenerator`**: Longitudinal/panel data with fixed/random effects

### Stochastic Processes
- **`BrownianMotion`**: Standard Wiener process
- **`GeometricBrownianMotion`**: Asset price modeling (Black-Scholes)
- **`OrnsteinUhlenbeck`**: Mean-reverting process (interest rates, spreads)
- **`CIRProcess`**: Cox-Ingersoll-Ross (non-negative, mean-reverting)
- **`HestonProcess`**: Stochastic volatility model

## Project Structure

```
datagen/
├── __init__.py                  # Package exports
├── factory.py                   # DataFactory (high-level API)
├── distributions.py             # Statistical distributions
├── stochastic_processes.py      # BM, GBM, OU, CIR, Heston
├── time_series.py              # AR, MA, ARMA, GARCH
├── regression.py               # Regression data generators
├── panel.py                    # Panel/longitudinal data
├── examples.py                 # Usage examples
└── tests/                      # Unit tests
```

## Design Principles

1. **Zero Dependencies**: Works anywhere Python runs
2. **Reproducible**: All generators support random seeds
3. **Type-Safe**: Consistent return types across all methods
4. **Composable**: Combine low-level and high-level APIs as needed

## Integration Examples

### Export to JSON

```python
import json

factory = DataFactory(seed=42)
data = factory.stock_prices(100)

with open('stock_data.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### Export to CSV

```python
import csv

factory = DataFactory(seed=42)
data = factory.sales_data(1000)

with open('sales.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
```

### Pandas Integration

```python
import pandas as pd

factory = DataFactory(seed=42)
data = factory.user_metrics(n_users=100, n_days=30)

df = pd.DataFrame(data)
print(df.head())
print(df.describe())
```

### Combine Multiple Sources

```python
factory = DataFactory(seed=42)

# Generate related datasets
dashboard_data = {
    'stocks': factory.stock_prices(252, ticker='AAPL'),
    'portfolio': factory.portfolio_data(n_assets=10, n_days=252),
    'users': factory.user_metrics(n_users=500, n_days=252),
    'revenue': factory.revenue_timeseries(n_days=252)
}

# Now dashboard_data contains everything you need
```

## Technical Details

### Random Number Generation
- Uses Box-Muller transform for normal distributions
- Marsaglia-Tsang method for gamma distribution
- Inverse transform sampling for exponential, uniform
- No external libraries (numpy, scipy) required

### Stochastic Process Discretization
- Euler-Maruyama scheme for SDEs
- Exact discretization for geometric Brownian motion
- Proper handling of boundary conditions (CIR non-negativity)

### Time Series Models
- Warmup periods for ARMA to reduce initialization bias
- Correct GARCH(1,1) variance updating
- Seasonal decomposition with sinusoidal components

## Running Examples

```bash
python examples.py
```

## Testing

```bash
python -m pytest tests/
```

## License