"""Compute machine learning performance using different metric and add to a common dataframe."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    output_filename: str,
    classes: list = [0, 1],
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates and saves confusion matrices.

    Args:
                    y_true (np.ndarray): True labels.
                    y_pred (np.ndarray): Predicted labels.
                    output_dir (str): Directory to save the output files.
                    output_filename (str): Filename for the output files.
                    classes (list, optional): List of class names. Defaults to [0, 1].
                    save (bool, optional): Whether to save the output files. Defaults to True.

    Returns:
                    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Confusion matrix, normalized confusion matrix, and meaning matrix.
    """

    try:
        classes = [n.replace("_", " ").capitalize() for n in classes]
    except Exception as e:
        print(f"Error: {e}")
        pass
    cm = confusion_matrix(y_true, y_pred, normalize=None)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df_meaning = pd.DataFrame([["TN", "FP"], ["FN", "TP"]], index=classes, columns=classes)

    cm_norm = confusion_matrix(y_true, y_pred, normalize="all")
    cm_norm = (cm_norm * 100).round(2)
    cm_df_norm = pd.DataFrame(cm_norm, index=classes, columns=classes)

    plt.rcParams["figure.figsize"] = [5, 4]
    ConfusionMatrixDisplay(cm_norm, display_labels=classes).plot()
    plt.tight_layout()

    if save:
        plt.savefig(output_dir + f"cm_{output_filename}.png", dpi=300)
        cm_df_meaning.to_csv(output_dir + f"cm_meaning_{output_filename}.csv")
        cm_df.to_csv(output_dir + f"cm_{output_filename}.csv")
        cm_df_norm.to_csv(output_dir + f"cm_normalized_{output_filename}.csv")

    return cm_df_meaning, cm_df, cm_df_norm


def plot_roc_auc_curve(
    y_true: np.ndarray,
    y_proba_1: np.ndarray,
    output_dir: str,
    output_filename: str,
    fig_format: str = "png",
    dpi: int = 300,
    save: bool = True,
    size: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Plots the ROC AUC curve.

    Args:
                    y_true (np.ndarray): True labels.
                    y_proba_1 (np.ndarray): Predicted probabilities for the positive class.
                    output_dir (str): Directory to save the output files.
                    output_filename (str): Filename for the output files.
                    fig_format (str, optional): Format of the saved figure. Defaults to 'png'.
                    dpi (int, optional): DPI of the saved figure. Defaults to 300.
                    save (bool, optional): Whether to save the figure. Defaults to True.
                    size (int, optional): Font size for the plot. Defaults to 12.

    Returns:
                    tuple[np.ndarray, np.ndarray, float]: False positive rate, true positive rate, and ROC AUC score.
    """
    # calculate the fpr and tpr for all thresholds of the classification
    # Softmax. not really making sense.
    # https://discuss.pytorch.org/t/logits-vs-log-softmax/95979

    fpr, tpr, threshold = metrics.roc_curve(y_true, y_proba_1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.clf()
    sns.set(rc={"figure.figsize": (6, 6)})
    sns.set_style("white")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8

    sns.lineplot(x=fpr, y=tpr, label="ROC AUC = %0.2f" % roc_auc, color="orange", ci=None)
    plt.legend(loc="lower right")
    sns.lineplot(x=[0, 1], y=[0, 1], markers="--", label="random = 0.5", color="dodgerblue")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("Recall\n(aka Sensitivity, True Positive Rate)")  # ,size=size)
    plt.xlabel("False Positive Rate\n(aka, 1-Specificity)")
    plt.tight_layout()
    if save:
        plt.savefig(output_dir + f"roc_auc_curve_{output_filename}.{fig_format}", dpi=dpi)

    return fpr, tpr, roc_auc


def plot_pr_auc_curve(
    y_true: np.ndarray,
    y_proba_1: np.ndarray,
    output_dir: str,
    output_filename: str,
    fig_format: str = "png",
    dpi: int = 300,
    save: bool = True,
    size: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Plots the Precision-Recall AUC curve.

    Args:
                    y_true (np.ndarray): True labels.
                    y_proba_1 (np.ndarray): Predicted probabilities for the positive class.
                    output_dir (str): Directory to save the output files.
                    output_filename (str): Filename for the output files.
                    fig_format (str, optional): Format of the saved figure. Defaults to 'png'.
                    dpi (int, optional): DPI of the saved figure. Defaults to 300.
                    save (bool, optional): Whether to save the figure. Defaults to True.
                    size (int, optional): Font size for the plot. Defaults to 12.

    Returns:
                    tuple[np.ndarray, np.ndarray, float]: Precision, recall, and PR AUC score.
    """
    # def plot_pr_auc_curve(y_true, y_proba_1, output_dir, output_filename, fig_format = 'png', dpi=300, save=True, size = 12):
    plt.clf()
    sns.set(rc={"figure.figsize": (6, 6)})
    sns.set_style("white")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8

    # calculate precision and recall for each threshold
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_proba_1)
    lr_auc = metrics.auc(lr_recall, lr_precision)
    # summarize scores
    # print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    y_true = np.array(y_true)
    baseline = len(y_true[y_true == 1]) / len(
        y_true
    )  #  baseline of PRC is determined by the ratio of positives (P) and sample size (P+N) as y = P / (P + N)
    sns.lineplot(
        x=[0, 1],
        y=[baseline, baseline],
        markers="--",
        label="Baseline (all predicted as 1) = P / (P + N)",
        color="dodgerblue",
    )
    sns.lineplot(
        x=lr_recall, y=lr_precision, label=f"Precision-Recall AUC = {np.round(lr_auc,2)}", color="orange", ci=None
    )  # , marker='.')
    plt.xlabel("Recall\n(aka Sensitivity, True Positive Rate)")
    plt.ylabel("Precision\n(aka Positive Predictive Value)")
    plt.legend()

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    if save:
        plt.savefig(output_dir + f"pr_auc_curve_{output_filename}.{fig_format}", dpi=dpi)

    return lr_precision, lr_recall, lr_auc


def calculate_npv(tn: int, fn: int) -> float:
    """Calculates the Negative Predictive Value (NPV).

    Args:
                    tn (int): True negatives.
                    fn (int): False negatives.

    Returns:
                    float: Negative Predictive Value (NPV).

    Example:
                    tn = 50  # Number of true negatives
                    fn = 10  # Number of false negatives

                    npv = calculate_npv(tn, fn)
                    print(f"Negative Predictive Value (NPV): {npv}")
    """
    # Check for a case where the denominator would be zero
    if tn + fn == 0:
        print("Warning. Undefined (TN + FN is 0, leading to division by zero)")
        return None
    npv = tn / (tn + fn)
    return npv


def custom_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_1: np.ndarray,
    output_dir: str,
    output_filename: str = None,
    best_params: dict = None,
    feature_vector: list = None,
    model_name: str = None,
    classes: list = [0, 1],
    amount_of_clauses: str = "all",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a custom classification report with various metrics.

    Args:
                    y_true (np.ndarray): True labels.
                    y_pred (np.ndarray): Predicted labels.
                    y_proba_1 (np.ndarray): Predicted probabilities for the positive class.
                    output_dir (str): Directory to save the output files.
                    output_filename (str, optional): Filename for the output files. Defaults to None.
                    best_params (dict, optional): Best model parameters. Defaults to None.
                    feature_vector (list, optional): Feature vector. Defaults to None.
                    model_name (str, optional): Name of the model. Defaults to None.
                    classes (list, optional): List of class names. Defaults to [0, 1].
                    amount_of_clauses (str, optional): Number of clauses. Defaults to 'all'.
                    save (bool, optional): Whether to save the output files. Defaults to True.

    Returns:
                    tuple[pd.DataFrame, pd.DataFrame]: Custom classification report and sklearn classification report.
    """
    if len(np.unique(y_true)) == 1:
        sensitivity = metrics.recall_score(y_true, y_pred)
        # Calculate TP and FN
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        # Now you can calculate the False Negative Rate (FNR)
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

        results = pd.DataFrame(
            [
                feature_vector,
                model_name,
                str(classes),
                amount_of_clauses,
                sensitivity,
                np.nan,
                np.nan,
                fnr,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                best_params,
                str(dict(Counter(y_true))),
            ],
            index=[
                "Feature vector",
                "Model",
                "Classes",
                "Amount of clauses",
                "Sensitivity",
                "Specificity",
                "Precision",
                "FNR",
                "F1",
                "ROC AUC",
                "Best th ROC AUC",
                "PR AUC",
                "Best th PR AUC",
                "Best parameters",
                "Support",
            ],
        ).T.round(2)
        if save:
            results.to_csv(output_dir + f"results_{output_filename}.csv")
        return results, None

    else:
        # Generate classification report as a dictionary
        report_dict = classification_report(y_true, y_pred, output_dict=True)

        fpr, tpr, roc_auc = plot_roc_auc_curve(
            y_true, y_proba_1, output_dir, output_filename, fig_format="png", dpi=300, save=save, size=12
        )

        lr_precision, lr_recall, lr_auc = plot_pr_auc_curve(
            y_true, y_proba_1, output_dir, output_filename, fig_format="png", dpi=300, save=save, size=12
        )

        # Convert the dictionary into a DataFrame
        sklearn_cr = pd.DataFrame(report_dict)
        if save:
            sklearn_cr.to_csv(output_dir + f"results_sklearn_{output_filename}.csv")

        # Custom classification report
        custom_cr_all_classes = []
        for i in list(range(len(classes))) + ["macro", "weighted"]:
            # support_both_classes = dict(Counter(y_true))
            if i == 0:
                pos_label = 0
                specificity_label = 1
                average = "binary"
                average_roc_auc = None
                support = dict(Counter(y_true))[pos_label]
            elif i == 1:
                pos_label = 1
                specificity_label = 0
                average = "binary"
                average_roc_auc = None
                support = dict(Counter(y_true))[pos_label]
            elif i == "macro":
                pos_label = 1
                specificity_label = 0
                average = "macro"
                average_roc_auc = average
                support = np.sum(dict(Counter(y_true)).values())
            elif i == "weighted":
                pos_label = 1
                specificity_label = 0
                average = "weighted"
                average_roc_auc = average
                support = np.sum(dict(Counter(y_true)).values())

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fnr = fn / (fn + tp)
            npv = calculate_npv(tn, fn)
            np.set_printoptions(suppress=True)
            roc_auc = roc_auc_score(y_true, y_proba_1, average=average_roc_auc)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label, average=average)

            # calculate precision and recall for each threshold
            lr_precision, lr_recall, thresholds = precision_recall_curve(y_true, y_proba_1, pos_label=pos_label)
            fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
            fscore[np.isnan(fscore)] = 0
            ix = np.argmax(fscore)
            best_threshold = thresholds[ix].item()
            best_threshold_roc_auc = find_optimal_threshold(y_true, y_proba_1)

            pr_auc = auc(lr_recall, lr_precision)
            # AU P-R curve is also approximated by avg. precision
            # avg_pr = metrics.average_precision_score(y_true,y_proba_1)

            sensitivity = metrics.recall_score(y_true, y_pred, pos_label=pos_label, average=average)
            specificity = metrics.recall_score(y_true, y_pred, pos_label=specificity_label, average=average)
            precision = metrics.precision_score(y_true, y_pred, pos_label=pos_label, average=average)

            custom_cr_i = pd.DataFrame(
                [
                    feature_vector,
                    model_name,
                    str(classes),
                    pos_label,
                    average,
                    amount_of_clauses,
                    sensitivity,
                    specificity,
                    precision,
                    npv,
                    fnr,
                    f1,
                    roc_auc,
                    best_threshold_roc_auc,
                    pr_auc,
                    best_threshold,
                    best_params,
                    support,
                ],
                index=[
                    "Feature vector",
                    "Model",
                    "Classes",
                    "Class",
                    "Average",
                    "Amount of clauses",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "Negative Predictive Value",
                    "FNR",
                    "F1",
                    "ROC AUC",
                    "Best th ROC AUC",
                    "PR AUC",
                    "Best th PR AUC",
                    "Best parameters",
                    "Support",
                ],
            ).T
            custom_cr_all_classes.append(custom_cr_i)

    custom_cr = pd.concat(custom_cr_all_classes)
    if save:
        custom_cr.to_csv(output_dir + f"results_{output_filename}.csv")
    return custom_cr, sklearn_cr


def find_optimal_threshold(y_true: np.ndarray, y_proba_1: np.ndarray) -> float:
    """Finds the optimal threshold for binary classification based on Youden's Index.

    Args:
                    y_true (np.ndarray): True binary labels.
                    y_proba_1 (np.ndarray): Target scores, probabilities of the positive class.

    Returns:
                    float: Optimal threshold value.
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_1)

    # Compute Youden's J index
    youdens_j = tpr - fpr

    # Find the optimal threshold
    optimal_idx = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def save_classification_performance(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba_1: np.ndarray,
    output_dir_i: str,
    output_filename: str = None,
    feature_vector: list = None,
    model_name: str = None,
    best_params: dict = None,
    classes: list = [0, 1],
    amount_of_clauses: str = "all",
    save_confusion_matrix: bool = True,
    save_output: bool = True,
) -> tuple:
    """Saves classification performance results and generates reports.

    Args:
                    y_test (np.ndarray): Test labels.
                    y_pred (np.ndarray): Predicted labels.
                    y_proba_1 (np.ndarray): Predicted probabilities for the positive class.
                    output_dir_i (str): Directory to save the output files.
                    output_filename (str, optional): Filename for the output files. Defaults to None.
                    feature_vector (list, optional): Feature vector. Defaults to None.
                    model_name (str, optional): Name of the model. Defaults to None.
                    best_params (dict, optional): Best model parameters. Defaults to None.
                    classes (list, optional): List of class names. Defaults to [0, 1].
                    amount_of_clauses (str, optional): Number of clauses. Defaults to 'all'.
                    save_confusion_matrix (bool, optional): Whether to save the confusion matrix. Defaults to True.
                    save_output (bool, optional): Whether to save the output files. Defaults to True.

    Returns:
                    tuple: Custom classification report, sklearn classification report, confusion matrixes, and predictions.
    """
    if output_filename is None:
        output_filename = f"{feature_vector}_{model_name}_{classes[1]}.csv"

    # Save predictions
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df["y_test"] = y_test

    y_pred_df.columns = ["y_pred", "y_test"]
    y_pred_df["y_proba_1"] = y_proba_1
    if save_output:
        y_pred_df.to_csv(output_dir_i + f"y_pred_{output_filename}.csv", index=False)

    if save_confusion_matrix and len(np.unique(y_pred)) != 1:
        # Save confusion matrix if y_test is not all the same
        cm_df_meaning, cm_df, cm_df_norm = cm(
            y_test, y_pred, output_dir_i, output_filename, classes=classes, save=save_output
        )

    custom_cr, sklearn_cr = custom_classification_report(
        y_test,
        y_pred,
        y_proba_1,
        output_dir_i,
        output_filename,
        feature_vector=feature_vector,
        model_name=model_name,
        best_params=best_params,
        classes=classes,
        amount_of_clauses=amount_of_clauses,
        save=save_output,
    )
    if save_confusion_matrix and len(np.unique(y_pred)) != 1:
        return custom_cr, sklearn_cr, cm_df_meaning, cm_df, cm_df_norm, y_pred_df
    else:
        return custom_cr, sklearn_cr, y_pred_df


def regression_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray = None,
    gridsearch: dict = None,
    best_params: dict = None,
    feature_vector: list = None,
    model_name: str = None,
    metrics_to_report: str = "all",
    plot: bool = True,
    save_fig_path: str = None,
    n: str = "all",
    round_to: int = 2,
    figsize: tuple = (4, 8),
    ordinal_ticks: bool = True,
) -> pd.DataFrame:
    """Generates a regression report including various metrics and plots.

    Args:
                    y_test (np.ndarray): Test labels.
                    y_pred (np.ndarray): Predicted labels.
                    y_train (np.ndarray, optional): Training labels. Defaults to None.
                    gridsearch (dict, optional): Grid search results. Defaults to None.
                    best_params (dict, optional): Best model parameters. Defaults to None.
                    feature_vector (list, optional): Feature vector. Defaults to None.
                    model_name (str, optional): Name of the model. Defaults to None.
                    metrics_to_report (str, list, optional): Metrics to report. Defaults to "all". options: {'all', ['MAE','RMSE','rho', 'Best parameters']}
                    plot (bool, optional): Whether to generate plots. Defaults to True.
                    save_fig_path (str, optional): Path to save the figure. Defaults to None.
                    n (str, optional): Sample size. Defaults to "all".
                    round_to (int, optional): Decimal places for rounding. Defaults to 2.
                    figsize (tuple, optional): Figure size. Defaults to (4, 8).
                    ordinal_ticks (bool, optional): Whether to use ordinal ticks. Defaults to True.

    Returns:
                    pd.DataFrame: DataFrame with regression metrics.
    """

    # Metrics
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    r, p = pearsonr(y_test, y_pred)
    rho, p = spearmanr(y_test, y_pred)

    results_dict = {
        "Features": feature_vector,
        "Estimator": model_name,
        "n": n,
        "y_train_min": np.min(y_train),
        "y_train_max": np.max(y_train),
        "RMSE": np.round(rmse, round_to),
        "MAE": np.round(mae, round_to),
        "R^2": np.round(r2, round_to),
        "r": np.round(r, round_to),
        "rho": np.round(rho, round_to),
        "gridsearch": gridsearch,
        "Best parameters": str(best_params),
    }
    results = pd.DataFrame(results_dict, index=[model_name]).round(3)
    # results_all.append(results)

    if metrics_to_report == "all" or ("RMSE per value" in metrics_to_report and "MAE per value" in metrics_to_report):
        y_pred_test = {}
        y_pred_test["RMSE per value"] = []
        y_pred_test["MAE per value"] = []
        for value in np.unique(y_test):
            y_pred_test_i = [[pred, test] for pred, test in zip(y_pred, y_test) if test == value]
            y_pred_test[value] = np.array(y_pred_test_i)
            y_pred_i = [n[0] for n in y_pred_test_i]
            y_test_i = [n[1] for n in y_pred_test_i]
            rmse_i = metrics.mean_squared_error(y_test_i, y_pred_i, squared=False)
            mae_i = metrics.mean_absolute_error(y_test_i, y_pred_i)
            y_pred_test["RMSE per value"].append(np.round(rmse_i, round_to))
            y_pred_test["MAE per value"].append(np.round(mae_i, round_to))
        # print(y_pred_test['RMSE per value'])
        results_dict.update(
            {"RMSE per value": f"{y_pred_test['RMSE per value']}", "MAE per value": f"{y_pred_test['MAE per value']}"}
        )
        macro_avg_rmse = np.round(np.mean(y_pred_test["RMSE per value"]), round_to)
        macro_avg_mae = np.round(np.mean(y_pred_test["MAE per value"]), round_to)

        results_dict.update(
            {
                "Macro avg. RMSE": f"{macro_avg_rmse}",
                "Macro avg. MAE": f"{macro_avg_mae}",
            }
        )

        # metrics_to_report_2 = metrics_to_report.copy()
        # metrics_to_report_2.remove('RMSE') #redundant
        # metrics_to_report_2.remove('MAE') #redundant
        results = pd.DataFrame(results_dict, index=[model_name])  # replace with updated metrics
        # results = results[metrics_to_report_2]

    # Plot result for a regression task: true value vs predicted values
    # ============================================================
    plt.clf()
    plt.figure(figsize=figsize)  # Width=10 inches, Height=6 inches

    plt.style.use("default")  # Example of applying the 'ggplot' style
    plt.scatter(y_test, y_pred, alpha=0.05)
    # plt.title(f"{feature_vector.capitalize().replace('_',' ')}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")

    ticks = list(np.unique(y_test))
    if ordinal_ticks and len(ticks) < 12:
        plt.xticks(ticks=ticks, labels=[str(int(n)) for n in ticks])

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path + ".png", dpi=300)
    # plt.show()
    return results


# This was moved to feature_importance.py
# def generate_feature_importance_df(
