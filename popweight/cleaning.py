"""Data cleaning and integrity rules."""

from types import ModuleType
from typing import Any

import pandas as pd

import popweight.config

CORE_COLUMNS = ["Likes", "Comments", "Shares", "Reach"]


def clean_core_columns(
    df: pd.DataFrame,
    config: ModuleType = popweight.config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean numeric columns and enforce integrity constraints.

    Converts Likes, Comments, Shares, Reach to numeric; drops rows with
    NaN in these columns; removes invalid values (negative interactions,
    low reach); optionally removes top-reach outliers.

    Args:
        df: DataFrame with schema-validated columns.
        config: Config module with MIN_REACH and
            REMOVE_TOP_REACH_PERCENTILE.

    Returns:
        Tuple of (cleaned DataFrame, report dict with drop counts).
    """
    report: dict[str, Any] = {
        "initial_rows": len(df),
        "dropped_nan": 0,
        "dropped_low_reach": 0,
        "dropped_negative": 0,
        "dropped_outlier_reach": 0,
    }
    work = df.copy()

    for col in CORE_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    before_nan = len(work)
    work = work.dropna(subset=CORE_COLUMNS)
    report["dropped_nan"] = before_nan - len(work)

    min_reach = getattr(config, "MIN_REACH", 1)
    mask_low = work["Reach"] < min_reach
    report["dropped_low_reach"] = int(mask_low.sum())
    work = work[~mask_low]

    mask_neg = (work["Likes"] < 0) | (work["Comments"] < 0) | (work["Shares"] < 0)
    report["dropped_negative"] = int(mask_neg.sum())
    work = work[~mask_neg]

    pct = getattr(config, "REMOVE_TOP_REACH_PERCENTILE", 0.995)
    thr = work["Reach"].quantile(pct)
    mask_outlier = work["Reach"] > thr
    report["dropped_outlier_reach"] = int(mask_outlier.sum())
    work = work[~mask_outlier]

    report["final_rows"] = len(work)
    report["total_dropped"] = report["initial_rows"] - report["final_rows"]

    return work, report
