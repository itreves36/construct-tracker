"""Inter-rater reliability (IRR) measures."""

from typing import List, Optional, Union

import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa


def cohens_kappa(ratings: np.ndarray, weights: Optional[str] = None) -> float:
    """
    Calculate Cohen's weighted kappa for two raters.

    Args:
        ratings (np.ndarray): A 2D NumPy array with two columns, each column representing a rater's ratings.
        weights (Optional[str]): A string, either 'linear' or 'quadratic', determining the type of weights to use.
                                 Defaults to None.

    Returns:
        float: The weighted kappa score.

    Example:
        >>> ratings = np.array([[1, 2], [2, 2], [3, 3], [0, 1], [1, 1]])
        >>> cohens_kappa(ratings, weights='linear')
        0.42857142857142855
    """
    kappa = cohen_kappa_score(ratings[:, 0], ratings[:, 1], weights=weights)
    return kappa


def calculate_fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Calculate Fleiss' kappa for three or more raters.

    Args:
        ratings (np.ndarray): A 2D NumPy array where rows represent items and columns represent raters.

    Returns:
        float: The Fleiss' kappa score.

    Example:
        >>> ratings = np.array([[1, 2, 1], [2, 2, 2], [3, 3, 2], [0, 1, 0], [1, 1, 2]])
        >>> calculate_fleiss_kappa(ratings)
        0.2857142857142857
    """
    # Count the number of times each rating occurs per item
    n_items, n_raters = ratings.shape
    max_rating = ratings.max() + 1
    rating_matrix = np.zeros((n_items, max_rating))

    for i in range(n_items):
        for j in range(n_raters):
            rating_matrix[i, ratings[i, j]] += 1

    kappa = fleiss_kappa(rating_matrix, method="fleiss")
    return kappa


def binary_inter_rater_reliability(rater1: Union[List[int], np.ndarray], rater2: Union[List[int], np.ndarray]) -> float:
    """
    Calculate Cohen's Kappa for binary inter-rater reliability.

    Args:
        rater1 (Union[List[int], np.ndarray]): Ratings from the first rater.
        rater2 (Union[List[int], np.ndarray]): Ratings from the second rater.

    Returns:
        float: Cohen's Kappa score.

    Example:
        >>> rater1 = [1, 0, 1, 1, 0]
        >>> rater2 = [1, 0, 0, 1, 0]
        >>> binary_inter_rater_reliability(rater1, rater2)
        0.6
    """
    kappa = cohen_kappa_score(rater1, rater2)
    return kappa


"""
# Example usage
ratings_2_raters = np.array([[1, 2], [2, 2], [3, 3], [0, 1], [1, 1]])
ratings_3_raters = np.array([[1, 2, 1], [2, 2, 2], [3, 3, 2], [0, 1, 0], [1, 1, 2]])

if ratings_2_raters.shape[1] == 2:
    kappa = cohens_kappa(ratings_2_raters)
    print(f"Cohen's Weighted Kappa (2 raters): {kappa}")
elif ratings_2_raters.shape[1] >= 3:
    kappa = calculate_fleiss_kappa(ratings_2_raters)
    print(f"Fleiss' Kappa (3 or more raters): {kappa}")

if ratings_3_raters.shape[1] >= 3:
    kappa = calculate_fleiss_kappa(ratings_3_raters)
    print(f"Fleiss' Kappa (3 or more raters): {kappa}")
"""
