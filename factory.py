# datagen/factory.py
"""
High-level data factory for quick dashboard/testing data generation.
"""

import random
from typing import List, Dict, Any, Optional, Literal
from .distributions import DistributionGenerator
from .time_series import TimeSeriesGenerator
from .regression import RegressionDataGenerator
from .panel import PanelDataGenerator
from .stochastic_processes import GeometricBrownianMotion


class DataFactory:
    """
    One-stop shop for generating data matching common dashboard patterns.

    Examples
    --------
    >>> factory = DataFactory(seed=42)
    >>>
    >>> # Stock prices
    >>> stock_data = factory.stock_prices(n_days=100, ticker='AAPL')
    >>>
    >>> # Sales dashboard
    >>> sales = factory.sales_data(n_records=1000, start_date='2024-01-01')
    >>>
    >>> # User metrics
    >>> users = factory.user_metrics(n_users=500, n_days=30)
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.dist = DistributionGenerator(seed=seed)
        self.ts = TimeSeriesGenerator(seed=seed)
        self.reg = RegressionDataGenerator(seed=seed)
        self.panel = PanelDataGenerator(seed=seed)

    # ==========================================
    # Financial Data
    # ==========================================

    def stock_prices(self,
                     n_days: int = 252,
                     ticker: str = 'STOCK',
                     start_price: float = 100.0,
                     volatility: float = 0.2,
                     drift: float = 0.0001) -> List[Dict[str, Any]]:
        """
        Generate stock price time series.

        Returns list of dicts with keys: date, ticker, open, high, low, close, volume
        """
        gbm = GeometricBrownianMotion(
            mu=drift,
            sigma=volatility,
            S0=start_price,
            dt=1.0,
            seed=self.seed
        )

        prices = gbm.generate_path(n_days)

        data = []
        for day in range(n_days):
            price = prices[day]

            # Generate OHLC from closing price
            daily_vol = volatility * price * 0.3
            open_price = price + random.gauss(0, daily_vol * 0.5)
            high = max(price, open_price) + abs(random.gauss(0, daily_vol))
            low = min(price, open_price) - abs(random.gauss(0, daily_vol))
            close = price

            # Volume (lognormal)
            volume = int(random.lognormvariate(15, 1))

            data.append({
                'day': day,
                'ticker': ticker,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })

        return data

    def portfolio_data(self,
                       n_assets: int = 10,
                       n_days: int = 252) -> List[Dict[str, Any]]:
        """
        Generate multi-asset portfolio data.

        Returns list of dicts: day, asset, price, weight, return
        """
        tickers = [f'ASSET_{i:02d}' for i in range(n_assets)]

        # Random weights (sum to 1)
        weights = self.dist.uniform(n_assets, 0, 1)
        total = sum(weights)
        weights = [w / total for w in weights]

        data = []
        for i, ticker in enumerate(tickers):
            prices = self.stock_prices(n_days, ticker, volatility=0.15 + random.random() * 0.2)

            for day_data in prices:
                returns = 0.0
                if day_data['day'] > 0:
                    prev_price = prices[day_data['day'] - 1]['close']
                    returns = (day_data['close'] - prev_price) / prev_price

                data.append({
                    'day': day_data['day'],
                    'asset': ticker,
                    'price': day_data['close'],
                    'weight': round(weights[i], 4),
                    'return': round(returns, 6)
                })

        return data

    # ==========================================
    # Business/Sales Data
    # ==========================================

    def sales_data(self,
                   n_records: int = 1000,
                   n_products: int = 20,
                   n_regions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate sales transaction data.

        Returns: transaction_id, day, product, region, quantity, price, revenue
        """
        products = [f'Product_{chr(65 + i)}' for i in range(n_products)]
        regions = [f'Region_{i + 1}' for i in range(n_regions)]

        # Product base prices
        base_prices = {p: random.uniform(10, 500) for p in products}

        data = []
        for txn_id in range(n_records):
            product = random.choice(products)
            region = random.choice(regions)

            # Quantity (Poisson-ish)
            quantity = max(1, int(random.expovariate(1 / 5)))

            # Price with noise
            price = base_prices[product] * random.uniform(0.9, 1.1)
            revenue = quantity * price

            # Day (random in range)
            day = random.randint(0, 364)

            data.append({
                'transaction_id': f'TXN_{txn_id:06d}',
                'day': day,
                'product': product,
                'region': region,
                'quantity': quantity,
                'price': round(price, 2),
                'revenue': round(revenue, 2)
            })

        return sorted(data, key=lambda x: x['day'])

    def revenue_timeseries(self,
                           n_days: int = 365,
                           base_revenue: float = 10000,
                           growth_rate: float = 0.001,
                           seasonality: bool = True) -> List[Dict[str, Any]]:
        """
        Generate daily revenue with trend and seasonality.

        Returns: day, revenue, cumulative_revenue
        """
        if seasonality:
            series = self.ts.generate_seasonal(
                n_samples=n_days,
                seasonal_period=7,  # Weekly pattern
                seasonal_strength=base_revenue * 0.3,
                trend=base_revenue * growth_rate,
                noise_std=base_revenue * 0.1
            )
            series = [base_revenue + s for s in series]
        else:
            series = self.ts.generate_random_walk(
                n_samples=n_days,
                drift=base_revenue * growth_rate,
                std=base_revenue * 0.1,
                start_value=base_revenue
            )

        cumulative = 0
        data = []
        for day, revenue in enumerate(series):
            revenue = max(0, revenue)
            cumulative += revenue
            data.append({
                'day': day,
                'revenue': round(revenue, 2),
                'cumulative_revenue': round(cumulative, 2)
            })

        return data

    # ==========================================
    # User/Analytics Data
    # ==========================================

    def user_metrics(self,
                     n_users: int = 1000,
                     n_days: int = 30) -> List[Dict[str, Any]]:
        """
        Generate user engagement metrics over time.

        Returns: user_id, day, sessions, page_views, time_spent, converted
        """
        data = []

        for user_id in range(n_users):
            # User characteristics (persistent)
            engagement_level = random.choice(['low', 'medium', 'high'])
            base_sessions = {'low': 1, 'medium': 3, 'high': 8}[engagement_level]

            for day in range(n_days):
                # Daily metrics
                sessions = max(0, int(random.gauss(base_sessions, base_sessions * 0.5)))
                page_views = sessions * int(random.gauss(5, 2))
                time_spent = page_views * random.gauss(45, 15)  # seconds per page

                # Conversion (higher for high engagement)
                conv_prob = {'low': 0.01, 'medium': 0.05, 'high': 0.15}[engagement_level]
                converted = 1 if random.random() < conv_prob else 0

                data.append({
                    'user_id': f'U_{user_id:05d}',
                    'day': day,
                    'sessions': sessions,
                    'page_views': max(0, page_views),
                    'time_spent': max(0, round(time_spent, 1)),
                    'converted': converted,
                    'engagement_level': engagement_level
                })

        return data

    def funnel_data(self,
                    n_visitors: int = 10000,
                    conversion_rates: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Generate conversion funnel data.

        Default funnel: visit -> signup -> activate -> purchase
        """
        if conversion_rates is None:
            conversion_rates = [1.0, 0.3, 0.6, 0.4]  # visit, signup, activate, purchase

        stages = ['visit', 'signup', 'activate', 'purchase']

        data = []
        for visitor_id in range(n_visitors):
            current_stage = 0

            for stage_idx, stage in enumerate(stages):
                if random.random() < conversion_rates[stage_idx]:
                    data.append({
                        'visitor_id': f'V_{visitor_id:06d}',
                        'stage': stage,
                        'stage_order': stage_idx,
                        'reached': 1
                    })
                    current_stage = stage_idx
                else:
                    break

        return data

    # ==========================================
    # IoT/Sensor Data
    # ==========================================

    def sensor_data(self,
                    n_sensors: int = 10,
                    n_readings: int = 1000,
                    sensor_type: Literal['temperature', 'pressure', 'humidity'] = 'temperature'
                    ) -> List[Dict[str, Any]]:
        """
        Generate IoT sensor readings.

        Returns: sensor_id, timestamp, value, anomaly
        """
        # Sensor parameters
        params = {
            'temperature': {'mean': 22, 'std': 2, 'min': -10, 'max': 50},
            'pressure': {'mean': 1013, 'std': 10, 'min': 900, 'max': 1100},
            'humidity': {'mean': 60, 'std': 15, 'min': 0, 'max': 100}
        }[sensor_type]

        data = []
        for sensor_id in range(n_sensors):
            # Each sensor has slight offset
            sensor_offset = random.gauss(0, params['std'] * 0.3)

            # Generate AR process for realistic correlation
            values = self.ts.generate_ar(
                n_samples=n_readings,
                coefficients=[0.8],
                mean=params['mean'] + sensor_offset,
                std=params['std']
            )

            for t, value in enumerate(values):
                # Clip to realistic range
                value = max(params['min'], min(params['max'], value))

                # Anomaly detection (3-sigma rule)
                anomaly = 1 if abs(value - params['mean']) > 3 * params['std'] else 0

                data.append({
                    'sensor_id': f'SENSOR_{sensor_id:03d}',
                    'timestamp': t,
                    'value': round(value, 2),
                    'anomaly': anomaly,
                    'sensor_type': sensor_type
                })

        return data

    # ==========================================
    # Tables/Structured Data
    # ==========================================

    def table_data(self,
                   n_rows: int = 100,
                   schema: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Generate structured table data matching a schema.

        Parameters
        ----------
        n_rows : int
            Number of rows
        schema : dict
            Column name -> type mapping
            Types: 'int', 'float', 'category', 'bool', 'normal', 'uniform'

        Examples
        --------
        >>> schema = {
        ...     'id': 'int',
        ...     'age': 'int',
        ...     'score': 'normal',
        ...     'category': 'category',
        ...     'active': 'bool'
        ... }
        >>> data = factory.table_data(100, schema)
        """
        if schema is None:
            schema = {
                'id': 'int',
                'value': 'normal',
                'category': 'category'
            }

        # Category options
        categories = ['A', 'B', 'C', 'D', 'E']

        data = []
        for row_id in range(n_rows):
            row = {}
            for col_name, col_type in schema.items():
                if col_type == 'int':
                    row[col_name] = row_id if col_name == 'id' else random.randint(0, 100)
                elif col_type == 'float':
                    row[col_name] = round(random.uniform(0, 100), 2)
                elif col_type == 'normal':
                    row[col_name] = round(random.gauss(50, 15), 2)
                elif col_type == 'uniform':
                    row[col_name] = round(random.uniform(0, 1), 4)
                elif col_type == 'category':
                    row[col_name] = random.choice(categories)
                elif col_type == 'bool':
                    row[col_name] = random.choice([True, False])
                else:
                    row[col_name] = None

            data.append(row)

        return data

    # ==========================================
    # ML/AI Training Data
    # ==========================================

    def classification_data(self,
                            n_samples: int = 1000,
                            n_features: int = 5,
                            n_classes: int = 2,
                            separability: float = 0.8) -> Dict[str, List]:
        """
        Generate classification dataset.

        Returns dict with 'X' (features) and 'y' (labels)
        """
        # Generate features
        X = []
        y = []

        for _ in range(n_samples):
            # Assign class
            class_label = random.randint(0, n_classes - 1)

            # Generate features correlated with class
            features = []
            for f in range(n_features):
                # Class-conditional mean
                if random.random() < separability:
                    mean = class_label * 2 - 1  # Separable
                else:
                    mean = 0  # Noise

                features.append(random.gauss(mean, 1))

            X.append(features)
            y.append(class_label)

        return {'X': X, 'y': y}