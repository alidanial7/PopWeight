"""Regression and classification evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict:
    """Compute regression metrics: R², MAE, RMSE, Pearson correlation.

    Args:
        y_true: Ground truth values (Reach_log).
        y_pred: Predicted values (Score).

    Returns:
        Dict with keys: r2, mae, rmse, pearson.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    pearson = float(corr) if not np.isnan(corr) else 0.0
    return {"r2": r2, "mae": mae, "rmse": rmse, "pearson": pearson}


def evaluate_regression(
    test_scored_df: pd.DataFrame,
    seed: int | None = None,
) -> dict:
    """Evaluate Score vs Reach_log on test set.

    Args:
        test_scored_df: DataFrame with Reach_log and Score columns.
        seed: Optional seed to include in output row.

    Returns:
        Dict with seed (if provided), r2, mae, rmse, pearson.
    """
    y_true = test_scored_df["Reach_log"]
    y_pred = test_scored_df["Score"]
    metrics = regression_metrics(y_true, y_pred)
    if seed is not None:
        metrics["seed"] = seed
    return metrics


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict:
    """Compute classification metrics: accuracy, precision, recall, F1.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, confusion_matrix.
        confusion_matrix is [[tn, fp], [fn, tp]].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    kwargs = {"zero_division": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, **kwargs)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
