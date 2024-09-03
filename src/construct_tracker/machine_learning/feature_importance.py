"""Feature importance for different sklearn and xgboost and lgbm models."""

import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_feature_importance_df(
    trained_model: object,
    model_name: str,
    feature_names: List[str],
    xgboost_method: str = "weight",
    model_name_in_pipeline: str = "estimator",
    lgbm_method: str = "split",
) -> Optional[pd.DataFrame]:
    """
    Generates a DataFrame showing feature importance from various model types.

    Args:
        trained_model (object): A trained model object, either a standalone sklearn model or part of a scikit-learn pipeline.
        model_name (str): The name of the model (e.g., 'SGDRegressor', 'LGBMClassifier').
        feature_names (List[str]): List of feature names.
        xgboost_method (str, optional): Method for XGBoost feature importance. Default is "weight".
        model_name_in_pipeline (str, optional): Name of the model step in the pipeline. Default is "estimator".
        lgbm_method (str, optional): Method for LGBM feature importance. Default is "split".

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing feature importance information, or None if the model type is not recognized.
    """
    # Feature importance using coefficients for linear models and gini
    if model_name in ["SGDRegressor", "Ridge", "Lasso", "LogisticRegression", "LinearSVC"]:
        try:
            coefs = list(trained_model.named_steps["model"].coef_)
        except AttributeError:
            coefs = list(trained_model.best_estimator_.named_steps[model_name_in_pipeline].coef_)

        try:
            coefs_df = pd.DataFrame(coefs, index=["Coef."], columns=feature_names).T
        except ValueError:
            coefs_df = pd.DataFrame(coefs, index=feature_names, columns=["Coef."])

        coefs_df["Abs. Coef."] = coefs_df["Coef."].abs()
        coefs_df = coefs_df.sort_values("Abs. Coef.", ascending=False).reset_index()
        coefs_df = coefs_df.drop(["Abs. Coef."], axis=1)
        coefs_df.index += 1
        coefs_df = coefs_df.reset_index()
        coefs_df.columns = ["Importance", "Feature", "Coef."]

        return coefs_df

    elif model_name in ["LGBMRegressor", "LGBMClassifier"]:
        try:
            importance_split = trained_model.named_steps[model_name_in_pipeline].booster_.feature_importance(
                importance_type="split"
            )
            importance_gain = trained_model.named_steps[model_name_in_pipeline].booster_.feature_importance(
                importance_type="gain"
            )
        except AttributeError:
            importance_split = trained_model.best_estimator_.named_steps[
                model_name_in_pipeline
            ].booster_.feature_importance(importance_type="split")
            importance_gain = trained_model.best_estimator_.named_steps[
                model_name_in_pipeline
            ].booster_.feature_importance(importance_type="gain")

        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "split": importance_split, "gain": importance_gain}
        )
        feature_importance_df = feature_importance_df.sort_values("gain", ascending=False)
        return feature_importance_df

    elif model_name in ["XGBRegressor", "XGBClassifier"]:
        try:
            feature_importance = (
                trained_model.named_steps[model_name_in_pipeline]
                .get_booster()
                .get_score(importance_type=xgboost_method)
            )
        except AttributeError:
            feature_importance = (
                trained_model.best_estimator_.named_steps[model_name_in_pipeline]
                .get_booster()
                .get_score(importance_type=xgboost_method)
            )

        feature_importance_df = pd.DataFrame(list(feature_importance.values()), index=list(feature_importance.keys()))
        feature_importance_df = feature_importance_df.sort_values(0, ascending=False).reset_index()
        feature_importance_df.index += 1
        feature_importance_df = feature_importance_df.reset_index()
        feature_importance_df.columns = ["Importance", "Feature", xgboost_method.capitalize()]

        feature_name_mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
        feature_importance_df["Feature"] = feature_importance_df["Feature"].map(feature_name_mapping)

        return feature_importance_df

    else:
        warnings.warn(f"Model not specified for feature importance: {model_name}")
        return None


def tfidf_feature_importances(
    pipe: object,
    top_k: int = 100,
    savefig_path: str = "",
    model_name_in_pipeline: str = "model",
    xgboost_method: str = "weight",
) -> pd.DataFrame:
    """
    Plots and returns feature importances for a TF-IDF pipeline model.

    Args:
        pipe ('Pipeline'): A scikit-learn pipeline object with a TF-IDF vectorizer and a model.
        top_k (int, optional): Number of top features to display. Default is 100.
        savefig_path (str, optional): Path to save the plot. Default is an empty string, which means no plot is saved.
        model_name_in_pipeline (str, optional): Name of the model step in the pipeline. Default is "model".
        xgboost_method (str, optional): Method for XGBoost feature importance. Default is "weight".

    Returns:
        pd.DataFrame: A DataFrame containing feature importance values and their absolute values.
    """
    feature_names = pipe.named_steps["vectorizer"].get_feature_names_out()

    try:
        coefs = pipe.named_steps["model"].coef_.flatten()
    except AttributeError:
        try:
            coefs = list(
                pipe.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method)
            )
        except AttributeError:
            coefs = (
                pipe.best_estimator_.named_steps[model_name_in_pipeline]
                .get_booster()
                .get_score(importance_type=xgboost_method)
            )

    df = pd.DataFrame(zip(feature_names, coefs), columns=["feature", "value"])
    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "orange" if x > 0 else "dodgerblue")
    df = df.sort_values("abs_value", ascending=False)

    fig, ax = plt.subplots(figsize=(3.5, 6))
    plt.style.use("default")
    sns.barplot(x="value", y="feature", data=df.head(top_k), hue="colors", ax=ax)
    ax.legend_.remove()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(f"Top {top_k} Features", fontsize=14)
    ax.set_xlabel("Coef", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{savefig_path}.png", dpi=300)
    plt.show()

    return df
