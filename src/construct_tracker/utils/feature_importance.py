import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_feature_importance_df(
    trained_model,
    model_name,
    feature_names,
    xgboost_method="weight",
    model_name_in_pipeline="estimator",
    lgbm_method="split",
):
    """
    Function to generate feature importance table for methods that use .coef_ from sklearn
    as well as xgboost models.
    both using sklearn pipelines that go into GridsearchCV, where we need to
    first access the best_estimator to access, for example, the coefficients.

    trained_model: sklearn type model object fit to data
    model_name: str among the ones that appear below
    xgboost_method: str, there are a few options: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score
    """

    #  Feature importance using coefficients for linear models and gini
    if model_name in ["SGDRegressor", "Ridge", "Lasso", "LogisticRegression", "LinearSVC"]:
        try:
            coefs = list(trained_model.named_steps["model"].coef_)
        except:
            coefs = list(
                trained_model.best_estimator_.named_steps[model_name_in_pipeline].coef_
            )  # Obtain coefficients from GridSearch
        try:
            coefs = pd.DataFrame(coefs, index=["Coef."], columns=feature_names).T  # make DF
        except:
            coefs = pd.DataFrame(coefs, index=feature_names, columns=["Coef."])  # make DF
        coefs["Abs. Coef."] = coefs[
            "Coef."
        ].abs()  # add column with absolute values to sort by, both positive and negative values are important.
        coefs = coefs.sort_values(
            "Abs. Coef.", ascending=False
        ).reset_index()  # sort by abs value and reset index to add a feature name column
        coefs = coefs.drop(["Abs. Coef."], axis=1)  # drop abs value, it's job is done
        coefs.index += 1  # Importance for publication, start index with 1 , as in 1st, 2nd, 3rd
        coefs = coefs.reset_index()  # turn index into column
        coefs.columns = ["Importance", "Feature", "Coef."]  # Clean column names
        feature_importance = coefs.copy()
        return feature_importance

    elif model_name in ["LGBMRegressor", "LGBMClassifier"]:
        try:
            importance_split = trained_model.named_steps[model_name_in_pipeline].booster_.feature_importance(
                importance_type="split"
            )
            importance_gain = trained_model.named_steps[model_name_in_pipeline].booster_.feature_importance(
                importance_type="gain"
            )
        # feature_names = trained_model.named_steps[model_name_in_pipeline].booster_.feature_name()
        except:
            importance_split = trained_model.best_estimator_.named_steps[
                model_name_in_pipeline
            ].booster_.feature_importance(importance_type="split")
            importance_gain = trained_model.best_estimator_.named_steps[
                model_name_in_pipeline
            ].booster_.feature_importance(importance_type="gain")
        # feature_names = trained_model.best_estimator_.named_steps[model_name_in_pipeline].booster_.feature_name()

        feature_importance = pd.DataFrame(
            {"feature": feature_names, "split": importance_split, "gain": importance_gain}
        )

        # Sort by gain
        feature_importance = feature_importance.sort_values("gain", ascending=False)
        return feature_importance

    elif model_name in ["XGBRegressor", "XGBClassifier"]:
        # WARNING it will not return values for features that weren't used: if feature 3 wasn't used there will not be a f3 in the results
        try:
            feature_importance = (
                trained_model.named_steps[model_name_in_pipeline]
                .get_booster()
                .get_score(importance_type=xgboost_method)
            )
        except:
            feature_importance = (
                trained_model.best_estimator_.named_steps[model_name_in_pipeline]
                .get_booster()
                .get_score(importance_type=xgboost_method)
            )
        feature_importance_keys = list(feature_importance.keys())
        feature_importance_values = list(feature_importance.values())
        feature_importance = pd.DataFrame(feature_importance_values, index=feature_importance_keys)  # make DF
        feature_importance = feature_importance.sort_values(0, ascending=False)
        feature_importance = feature_importance.reset_index()

        feature_importance.index += 1
        feature_importance = feature_importance.reset_index()
        feature_importance

        feature_importance.columns = ["Importance", "Feature", xgboost_method.capitalize()]

        feature_name_mapping = {}
        for i, feature_name_i in enumerate(feature_names):
            feature_name_mapping[f"f{i}"] = feature_name_i

        # Or manually edit here:
        # feature_name_mapping = {'f0': 'Unnamed: 0', 'f1': 'Adult Mortality', 'f2': 'infant deaths', 'f3': 'percentage expenditure', 'f4': 'Hepatitis B', 'f5': 'Measles ', 'f6': ' BMI ', 'f7': 'under-five deaths ', 'f8': 'Polio', 'f9': 'Diphtheria ', 'f10': ' HIV/AIDS', 'f11': ' thinness  1-19 years', 'f12': ' thinness 5-9 years', 'f13': 'Developing'}

        feature_importance["Feature"] = feature_importance["Feature"].map(feature_name_mapping)
        # Todo: add feature_importances_ for sklearn tree based models
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#feature-importance-based-on-mean-decrease-in-impurity

        return feature_importance
    else:
        warnings.warn(f"model not specificied for feature importance: {model_name}")
        return None






def tfidf_feature_importances(pipe, top_k = 100, savefig_path = '', model_name_in_pipeline = 'model', xgboost_method = 'weight' ):
    # # Using sklearn pipeline:
    feature_names = pipe.named_steps["vectorizer"].get_feature_names_out()
    
    try: coefs = pipe.named_steps["model"].coef_.flatten() # Get the coefficients of each feature
    except: 
        try: coefs = list(pipe.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method )) # pipeline directly
        except:
            # gridsearchcv(pipeline)
            coefs = pipe.best_estimator_.named_steps[model_name_in_pipeline].get_booster().get_score(importance_type=xgboost_method )
    
    # Without sklearn pipeline
    # feature_names = vectorizer.get_feature_names_out()
    # print(len(feature_names ))
    # coefs = pipeline.coef_.flatten() # Get the coefficients of each feature
    
    # Visualize feature importances
    # Sort features by absolute value
    df = pd.DataFrame(zip(feature_names, coefs), columns=["feature", "value"])
    df["abs_value"] = df["value"].apply(lambda x: abs(x))
    df["colors"] = df["value"].apply(lambda x: "orange" if x > 0 else "dodgerblue")
    df = df.sort_values("abs_value", ascending=False) # sort by absolute coefficient value
    
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 6))
    plt.style.use('default')  # Example of applying the 'ggplot' style
    ax = sns.barplot(x="value",
                y="feature",
                data=df.head(top_k),
                hue="colors")
    ax.legend_.remove()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(f"Top {top_k} Features", fontsize=14)
    ax.set_xlabel("Coef", fontsize=12) # coeficient from linear model
    ax.set_ylabel("Feature Name", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(savefig_path+'.png', dpi=300)
    plt.show()
    return df
