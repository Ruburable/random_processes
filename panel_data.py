# panel_data.py

import math
import random
from typing import List, Dict, Any, Optional


class PanelDataGenerator:
    """
    Generate panel/longitudinal data with multiple entities over time.

    Features:
    - Fixed effects (entity-specific constants)
    - Time effects
    - Random effects
    - Autocorrelation within entities
    - Cross-sectional correlation
    """

    def __init__(self,
                 n_entities: int = 100,
                 n_periods: int = 50,
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        n_entities : int
            Number of cross-sectional units (e.g., firms, individuals)
        n_periods : int
            Number of time periods
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_entities = n_entities
        self.n_periods = n_periods

        if seed is not None:
            random.seed(seed)

        # Generate entity identifiers
        self.entity_ids = [f"entity_{i:04d}" for i in range(n_entities)]

        # Fixed effects (entity-specific intercepts)
        self.fixed_effects = {
            eid: random.gauss(0, 1)
            for eid in self.entity_ids
        }

    def _box_muller(self) -> float:
        """Generate standard normal random variable."""
        u1 = random.random()
        u2 = random.random()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def generate_continuous_variable(self,
                                     name: str,
                                     mean: float = 0.0,
                                     std: float = 1.0,
                                     ar_coef: float = 0.0,
                                     include_fixed_effect: bool = True,
                                     time_trend: float = 0.0) -> List[Dict[str, Any]]:
        """
        Generate continuous variable with panel structure.

        Parameters
        ----------
        name : str
            Variable name
        mean : float
            Base mean
        std : float
            Standard deviation
        ar_coef : float
            AR(1) coefficient for within-entity autocorrelation (0 to 1)
        include_fixed_effect : bool
            Include entity fixed effects
        time_trend : float
            Linear time trend coefficient

        Returns
        -------
        List of dicts with keys: entity_id, period, value
        """
        data = []

        # Track previous values for AR process
        prev_values = {eid: 0.0 for eid in self.entity_ids}

        for t in range(self.n_periods):
            for eid in self.entity_ids:
                # Components
                base = mean

                # Fixed effect
                if include_fixed_effect:
                    base += self.fixed_effects[eid]

                # Time trend
                base += time_trend * t

                # AR(1) component
                innovation = std * self._box_muller()
                value = base + ar_coef * prev_values[eid] + innovation
                prev_values[eid] = value - base  # Store deviation

                data.append({
                    'entity_id': eid,
                    'period': t,
                    name: value
                })

        return data

    def generate_categorical_variable(self,
                                      name: str,
                                      categories: List[str],
                                      probabilities: Optional[List[float]] = None,
                                      persistence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Generate categorical variable.

        Parameters
        ----------
        name : str
            Variable name
        categories : list
            List of category labels
        probabilities : list, optional
            Probability for each category (must sum to 1)
        persistence : float
            Probability of staying in same category (0 to 1)

        Returns
        -------
        List of dicts with keys: entity_id, period, value
        """
        if probabilities is None:
            probabilities = [1.0 / len(categories)] * len(categories)

        # Cumulative probabilities for sampling
        cum_probs = []
        cum = 0.0
        for p in probabilities:
            cum += p
            cum_probs.append(cum)

        def sample_category():
            r = random.random()
            for i, cp in enumerate(cum_probs):
                if r <= cp:
                    return categories[i]
            return categories[-1]

        data = []
        prev_category = {eid: sample_category() for eid in self.entity_ids}

        for t in range(self.n_periods):
            for eid in self.entity_ids:
                # Persistence: stay in previous category with probability
                if random.random() < persistence:
                    category = prev_category[eid]
                else:
                    category = sample_category()

                prev_category[eid] = category

                data.append({
                    'entity_id': eid,
                    'period': t,
                    name: category
                })

        return data

    def generate_binary_variable(self,
                                 name: str,
                                 probability: float = 0.5,
                                 persistence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Generate binary (0/1) variable.

        Parameters
        ----------
        name : str
            Variable name
        probability : float
            Base probability of 1
        persistence : float
            AR(1) coefficient for persistence

        Returns
        -------
        List of dicts with keys: entity_id, period, value
        """
        data = []
        prev_values = {eid: 0 for eid in self.entity_ids}

        for t in range(self.n_periods):
            for eid in self.entity_ids:
                # Latent variable with persistence
                latent = persistence * prev_values[eid] + (1 - persistence) * self._box_muller()

                # Convert to probability via logistic function
                p = 1 / (1 + math.exp(-latent - math.log(probability / (1 - probability))))
                value = 1 if random.random() < p else 0

                prev_values[eid] = latent

                data.append({
                    'entity_id': eid,
                    'period': t,
                    name: value
                })

        return data

    def merge_variables(self, *variable_lists) -> List[Dict[str, Any]]:
        """
        Merge multiple variables into single panel dataset.

        Parameters
        ----------
        *variable_lists : variable data from generate_* methods

        Returns
        -------
        Merged list of dicts
        """
        if not variable_lists:
            return []

        # Create lookup dict
        merged = {}
        for var_data in variable_lists:
            for row in var_data:
                key = (row['entity_id'], row['period'])
                if key not in merged:
                    merged[key] = {
                        'entity_id': row['entity_id'],
                        'period': row['period']
                    }
                # Add all non-key fields
                for k, v in row.items():
                    if k not in ['entity_id', 'period']:
                        merged[key][k] = v

        # Convert to sorted list
        result = sorted(merged.values(),
                        key=lambda x: (x['entity_id'], x['period']))
        return result


# Example usage
if __name__ == "__main__":
    # Create panel data generator
    panel = PanelDataGenerator(n_entities=5, n_periods=10, seed=42)

    # Generate various variables
    income = panel.generate_continuous_variable(
        name='income',
        mean=50000,
        std=10000,
        ar_coef=0.7,
        time_trend=500
    )

    education = panel.generate_categorical_variable(
        name='education',
        categories=['high_school', 'bachelors', 'masters', 'phd'],
        probabilities=[0.4, 0.35, 0.2, 0.05],
        persistence=0.95  # Education rarely changes
    )

    employed = panel.generate_binary_variable(
        name='employed',
        probability=0.85,
        persistence=0.6
    )

    # Merge into single dataset
    dataset = panel.merge_variables(income, education, employed)

    # Display first 15 rows
    print("Entity | Period | Income    | Education   | Employed")
    print("-" * 60)
    for row in dataset[:15]:
        print(f"{row['entity_id']} | {row['period']:6d} | "
              f"{row['income']:9.2f} | {row['education']:11s} | "
              f"{row['employed']:8d}")