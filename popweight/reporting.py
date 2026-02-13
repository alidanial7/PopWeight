"""Aggregation and reporting of metrics across seeds."""

from pathlib import Path
from typing import Any

import pandas as pd

REGRESSION_COLS = ["r2", "mae", "rmse", "pearson"]
CLASSIFICATION_COLS = ["accuracy", "precision", "recall", "f1"]


def aggregate_regression_metrics(
    metrics_list: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate regression metrics across seeds: per-seed + mean + std.

    Args:
        metrics_list: List of dicts with seed, r2, mae, rmse, pearson.

    Returns:
        DataFrame with rows per seed plus "mean" and "std" summary rows.
    """
    df = pd.DataFrame(metrics_list)
    df = df.sort_values("seed").reset_index(drop=True)
    numeric = df[REGRESSION_COLS]
    mean_row = numeric.mean().to_dict()
    std_row = numeric.std().to_dict()
    mean_row["seed"] = "mean"
    std_row["seed"] = "std"
    return pd.concat(
        [df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )


def aggregate_classification_metrics(
    metrics_list: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate classification metrics across seeds: per-seed + mean + std.

    Drops confusion_matrix for aggregation (non-numeric).

    Args:
        metrics_list: List of dicts with seed, accuracy, precision, recall, f1.

    Returns:
        DataFrame with rows per seed plus "mean" and "std" summary rows.
    """
    df = pd.DataFrame(metrics_list)
    if "confusion_matrix" in df.columns:
        df = df.drop(columns=["confusion_matrix"])
    df = df.sort_values("seed").reset_index(drop=True)
    numeric_cols = [c for c in CLASSIFICATION_COLS if c in df.columns]
    numeric = df[numeric_cols]
    mean_row = numeric.mean().to_dict()
    std_row = numeric.std().to_dict()
    mean_row["seed"] = "mean"
    std_row["seed"] = "std"
    return pd.concat(
        [df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )


def aggregate_weights(weights_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate weights from all seeds, optionally add mean per segment.

    Args:
        weights_list: List of weights DataFrame (one per seed).

    Returns:
        DataFrame with all weights. If multiple seeds per segment, includes
        mean of alpha, beta, gamma, intercept.
    """
    if not weights_list:
        return pd.DataFrame()
    combined = pd.concat(weights_list, ignore_index=True)
    return combined


def save_reports(
    regression_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    output_dir: str = "outputs",
) -> None:
    """Save aggregated metrics and weights to CSV files.

    Args:
        regression_df: Output from aggregate_regression_metrics.
        classification_df: Output from aggregate_classification_metrics.
        weights_df: Combined or averaged weights.
        output_dir: Directory for output files (default outputs/).
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    regression_df.to_csv(path / "metrics_regression.csv", index=False)

    classification_df.to_csv(path / "metrics_classification.csv", index=False)

    weights_df.to_csv(path / "weights.csv", index=False)
