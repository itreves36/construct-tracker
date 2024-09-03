"""Some common machine learning pipelines."""

import warnings
from itertools import product
from typing import Any, Dict, List, Optional

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ruff: noqa: E101


def get_pipelines(
    feature_vector: str, model_name: str = "Ridge", tfidf_vectorizer: Optional[bool] = None, random_state: int = 123
) -> Pipeline:
    """Create a machine learning pipeline based on the specified model and feature vector.

    Args:
                                                                    feature_vector: Type of feature vector ('tfidf' or other).
                                                                    model_name: Name of the model to use ('Ridge', 'LogisticRegression', etc.).
                                                                    tfidf_vectorizer: Whether to use TFIDF vectorizer.
                                                                    random_state: Random state for reproducibility.

    Returns:
                                                                    A configured machine learning pipeline.
    """
    # ruff: noqa: F401
    if "LGBM" in model_name:
        from lightgbm import LGBMClassifier, LGBMRegressor
    elif "XGB" in model_name:
        from xgboost import XGBClassifier, XGBRegressor
    elif "Logistic" in model_name or "Ridge" in model_name:
        from sklearn.linear_model import LogisticRegression, Ridge
    # ruff: enable
    model = globals()[model_name]()
    model.set_params(random_state=random_state)

    if feature_vector == "tfidf":
        if tfidf_vectorizer:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                min_df=3,
                ngram_range=(1, 2),
                stop_words=None,  #'english',# these include 'just': stopwords.words('english')+["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would'], strip_accents='unicode',
                sublinear_tf=True,
                # tokenizer=nltk_lemmatize,
                token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'",
                use_idf=True,
            )
            # alternative
            # from nltk import word_tokenize
            # from nltk.stem import WordNetLemmatizer
            # lemmatizer = WordNetLemmatizer()
            # def nltk_lemmatize(text):
            # 	return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]
            # tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk_lemmatize, stop_words='english')
        pipeline = Pipeline(
            [
                ("vectorizer", vectorizer),
                ("model", model),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("standardizer", StandardScaler()),
                ("model", model),
            ]
        )
    return pipeline


def get_params(
    feature_vector: str,
    model_name: str = "Ridge",
    toy: bool = False,
    ridge_alphas: Optional[List[float]] = None,
    ridge_alphas_toy: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Generate a parameter grid for model hyperparameter tuning.

    Args:
                                                                    feature_vector: Type of feature vector ('tfidf' or other).
                                                                    model_name: Name of the model to use ('Ridge', 'LogisticRegression', etc.).
                                                                    toy: Whether to use a toy version of the parameter grid.
                                                                    ridge_alphas: List of alpha values for Ridge regularization.
                                                                    ridge_alphas_toy: List of alpha values for toy version.

    Returns:
                                                                    A dictionary containing the parameter grid.
    """
    if ridge_alphas is None:
        ridge_alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    if ridge_alphas_toy is None:
        ridge_alphas_toy = [0.1, 10]

    if model_name in ["LogisticRegression"]:
        if feature_vector == "tfidf":
            # ruff: noqa: F601
            if toy:
                warnings.warn("WARNING, running toy version")
                param_grid = {
                    "vectorizer__max_features": [256, 512],
                }
            else:
                param_grid = {
                    "vectorizer__max_features": [512, 2048, None],
                    "model__C": ridge_alphas,
                }
            # ruff: enable

        else:
            if toy:
                warnings.warn("WARNING, running toy version")
                param_grid = {
                    "model__C": ridge_alphas_toy,
                }
            else:
                param_grid = {
                    "model__C": ridge_alphas,
                }

    elif model_name in ["Ridge", "Lasso"]:
        if feature_vector == "tfidf":
            if toy:
                warnings.warn("WARNING, running toy version")
                param_grid = {
                    "vectorizer__max_features": [256, 512],
                }
            else:
                param_grid = {
                    "vectorizer__max_features": [512, 2048, None],
                    "model__alpha": ridge_alphas,
                }

        else:
            if toy:
                warnings.warn("WARNING, running toy version")
                param_grid = {
                    "model__alpha": ridge_alphas_toy,
                }
            else:
                param_grid = {
                    "model__alpha": ridge_alphas,
                }

    elif model_name in ["LGBMRegressor", "LGBMClassifier"]:
        if toy:
            warnings.warn("WARNING, running toy version")
            param_grid = {
                # 'vectorizer__max_features': [256,2048],
                # 'model__colsample_bytree': [0.5, 1],
                "model__max_depth": [10, 20],  # -1 is the default and means No max depth
            }
        else:
            if feature_vector == "tfidf":
                param_grid = {
                    "vectorizer__max_features": [256, 2048, None],
                    "model__num_leaves": [30, 45, 60],
                    "model__colsample_bytree": [0.1, 0.5, 1],
                    "model__max_depth": [0, 5, 15],  # 0 is the default and means No max depth
                    "model__min_child_weight": [0.01, 0.001, 0.0001],
                    "model__min_child_samples": [10, 20, 40],  # alias: min_data_in_leaf
                    "vectorizer__max_features": [256, 512],
                }

            param_grid = {
                "model__num_leaves": [30, 45, 60],
                "model__colsample_bytree": [0.1, 0.5, 1],
                "model__max_depth": [0, 5, 15],  # 0 is the default and means No max depth
                "model__min_child_weight": [0.01, 0.001, 0.0001],
                "model__min_child_samples": [10, 20, 40],  # alias: min_data_in_leaf
            }

    elif model_name in ["XGBRegressor", "XGBClassifier"]:
        if toy:
            warnings.warn("WARNING, running toy version")
            param_grid = {
                "model__max_depth": [10, 20],  # -1 is the default and means No max depth
            }
        else:
            if feature_vector == "tfidf":
                param_grid = {
                    "vectorizer__max_features": [256, 2048, None],
                    "model__colsample_bytree": [0.1, 0.5, 1],
                    "model__max_depth": [5, 15, None],  # None is the default and means No max depth
                    "model__min_child_weight": [0.01, 0.001, 0.0001],
                }

            param_grid = {
                "model__colsample_bytree": [0.1, 0.5, 1],
                "model__max_depth": [5, 15, None],  # None is the default and means No max depth
                "model__min_child_weight": [0.01, 0.001, 0.0001],
            }

    return param_grid


def get_combinations(parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameter sets.

    Args:
                                                                    parameters: A dictionary of model parameters and their corresponding values.

    Returns:
                                                                    A list of dictionaries, each representing a unique combination of parameters.

    Example:
                                                                    parameters =   {'model__colsample_bytree': [1, 0.5, 0.1],
                                                                                                                                                                                                    'model__max_depth': [-1,10,20], #-1 is the default and means No max depth
                                                                                                                                                                                                    'model__min_child_weight': [0.01, 0.001, 0.0001],
                                                                                                                                                                                                    'model__min_child_samples': [10, 20,40], #alias: min_data_in_leaf
                                                                                                                                       }


    """

    combinations = list(product(*parameters.values()))

    parameter_set_combinations = []
    for combination in combinations:
        parameter_set_i = {}

        for i, k in enumerate(parameters.keys()):
            parameter_set_i[k] = combination[i]
        parameter_set_combinations.append(parameter_set_i)
    return parameter_set_combinations


# ruff: enable
