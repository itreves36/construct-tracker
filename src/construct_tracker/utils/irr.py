import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa


def cohens_kappa(ratings, weights=None):
    """
    Calculate Cohen's weighted kappa for two raters.

    :param ratings: a 2D NumPy array with two columns, each column representing a rater's ratings
    :param weights: a string, either 'linear' or 'quadratic', determining the type of weights to use
    :return: the weighted kappa score
    """
    kappa = cohen_kappa_score(ratings[:, 0], ratings[:, 1], weights=weights)
    return kappa


def calculate_fleiss_kappa(ratings):
    """
    Calculate Fleiss' kappa for three or more raters.

    :param ratings: a 2D NumPy array where rows represent items and columns represent raters
    :return: the Fleiss' kappa score
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


def binary_inter_rater_reliability(rater1, rater2):
    """
    Calculate Cohen's Kappa for binary inter-rater reliability.

    Parameters:
    rater1 (list or array): Ratings from the first rater
    rater2 (list or array): Ratings from the second rater

    Returns:
    float: Cohen's Kappa score
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
