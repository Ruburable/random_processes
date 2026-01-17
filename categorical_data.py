# categorical_data.py

import random
import math
from typing import List, Dict, Optional, Tuple


class CategoricalDataGenerator:
    """
    Generate categorical and qualitative variables with various distributions.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_multinomial(self,
                             n_samples: int,
                             categories: List[str],
                             probabilities: Optional[List[float]] = None) -> List[str]:
        """
        Generate multinomial categorical data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        categories : list
            List of category labels
        probabilities : list, optional
            Probability for each category (default: uniform)

        Returns
        -------
        List of category values
        """
        if probabilities is None:
            probabilities = [1.0 / len(categories)] * len(categories)

        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Cumulative probabilities
        cum_probs = []
        cum = 0.0
        for p in probabilities:
            cum += p
            cum_probs.append(cum)

        # Generate samples
        samples = []
        for _ in range(n_samples):
            r = random.random()
            for i, cp in enumerate(cum_probs):
                if r <= cp:
                    samples.append(categories[i])
                    break

        return samples

    def generate_ordinal(self,
                         n_samples: int,
                         categories: List[str],
                         mean_rank: float = None,
                         std_rank: float = 1.0) -> List[str]:
        """
        Generate ordinal data with underlying continuous distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples
        categories : list
            Ordered category labels (low to high)
        mean_rank : float, optional
            Mean of underlying distribution (default: middle)
        std_rank : float
            Standard deviation of underlying distribution

        Returns
        -------
        List of ordinal values
        """
        n_categories = len(categories)

        if mean_rank is None:
            mean_rank = (n_categories - 1) / 2

        samples = []
        for _ in range(n_samples):
            # Generate from normal distribution
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

            value = mean_rank + std_rank * z

            # Map to category
            rank = max(0, min(n_categories - 1, int(round(value))))
            samples.append(categories[rank])

        return samples

    def generate_likert(self,
                        n_samples: int,
                        n_points: int = 5,
                        mean_response: float = 3.0,
                        std_response: float = 1.0) -> List[int]:
        """
        Generate Likert scale responses (1 to n_points).

        Parameters
        ----------
        n_samples : int
            Number of responses
        n_points : int
            Number of scale points (e.g., 5 or 7)
        mean_response : float
            Mean response value
        std_response : float
            Standard deviation

        Returns
        -------
        List of integer responses
        """
        responses = []
        for _ in range(n_samples):
            # Normal distribution
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

            value = mean_response + std_response * z
            response = max(1, min(n_points, int(round(value))))
            responses.append(response)

        return responses

    def generate_markov_chain(self,
                              n_samples: int,
                              states: List[str],
                              transition_matrix: List[List[float]],
                              initial_state: Optional[str] = None) -> List[str]:
        """
        Generate categorical sequence following Markov chain.

        Parameters
        ----------
        n_samples : int
            Number of samples
        states : list
            List of state labels
        transition_matrix : list of lists
            Transition probability matrix (rows sum to 1)
        initial_state : str, optional
            Starting state (default: random)

        Returns
        -------
        List of states
        """
        n_states = len(states)
        state_idx = {s: i for i, s in enumerate(states)}

        # Initial state
        if initial_state is None:
            current_idx = random.randint(0, n_states - 1)
        else:
            current_idx = state_idx[initial_state]

        sequence = [states[current_idx]]

        # Generate transitions
        for _ in range(n_samples - 1):
            probs = transition_matrix[current_idx]

            # Sample next state
            r = random.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    current_idx = i
                    break

            sequence.append(states[current_idx])

        return sequence

    def generate_hierarchical(self,
                              n_samples: int,
                              hierarchy: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Generate hierarchical categorical data.

        Parameters
        ----------
        n_samples : int
            Number of samples
        hierarchy : dict
            Dict mapping parent categories to lists of child categories
            Example: {'North': ['NY', 'MA'], 'South': ['TX', 'FL']}

        Returns
        -------
        List of (parent, child) tuples
        """
        samples = []
        parents = list(hierarchy.keys())

        for _ in range(n_samples):
            parent = random.choice(parents)
            child = random.choice(hierarchy[parent])
            samples.append((parent, child))

        return samples

    def generate_zipf(self,
                      n_samples: int,
                      categories: List[str],
                      alpha: float = 1.5) -> List[str]:
        """
        Generate categorical data following Zipf's law (power law).
        Useful for modeling word frequencies, city populations, etc.

        Parameters
        ----------
        n_samples : int
            Number of samples
        categories : list
            Category labels (ordered by rank)
        alpha : float
            Zipf exponent (higher = more skewed)

        Returns
        -------
        List of categories
        """
        n_categories = len(categories)

        # Compute Zipf probabilities
        ranks = range(1, n_categories + 1)
        probabilities = [1.0 / (r ** alpha) for r in ranks]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Cumulative probabilities
        cum_probs = []
        cum = 0.0
        for p in probabilities:
            cum += p
            cum_probs.append(cum)

        # Generate samples
        samples = []
        for _ in range(n_samples):
            r = random.random()
            for i, cp in enumerate(cum_probs):
                if r <= cp:
                    samples.append(categories[i])
                    break

        return samples


# Example usage
if __name__ == "__main__":
    gen = CategoricalDataGenerator(seed=42)

    # Example 1: Multinomial
    print("1. Multinomial (fruit preferences):")
    fruits = gen.generate_multinomial(
        n_samples=20,
        categories=['apple', 'banana', 'orange', 'grape'],
        probabilities=[0.3, 0.4, 0.2, 0.1]
    )
    print(fruits[:10])

    # Example 2: Ordinal (satisfaction ratings)
    print("\n2. Ordinal (satisfaction levels):")
    satisfaction = gen.generate_ordinal(
        n_samples=15,
        categories=['very_unsatisfied', 'unsatisfied', 'neutral',
                    'satisfied', 'very_satisfied'],
        mean_rank=3.0,
        std_rank=1.0
    )
    print(satisfaction)

    # Example 3: Likert scale
    print("\n3. Likert scale (1-5):")
    likert = gen.generate_likert(
        n_samples=20,
        n_points=5,
        mean_response=3.5,
        std_response=0.8
    )
    print(likert)

    # Example 4: Markov chain (weather)
    print("\n4. Markov chain (weather states):")
    weather = gen.generate_markov_chain(
        n_samples=15,
        states=['sunny', 'cloudy', 'rainy'],
        transition_matrix=[
            [0.7, 0.2, 0.1],  # from sunny
            [0.3, 0.4, 0.3],  # from cloudy
            [0.2, 0.3, 0.5]  # from rainy
        ]
    )
    print(weather)

    # Example 5: Zipf distribution (word usage)
    print("\n5. Zipf distribution (word frequency):")
    words = gen.generate_zipf(
        n_samples=30,
        categories=['the', 'a', 'to', 'of', 'and', 'in', 'is', 'it'],
        alpha=1.5
    )
    print(words)

    # Count frequencies
    from collections import Counter

    print("Word counts:", Counter(words))