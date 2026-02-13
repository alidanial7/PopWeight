"""Trending label construction based on Reach percentile."""

import pandas as pd


def compute_segment_thresholds(
    train_df: pd.DataFrame,
    percentile: float,
) -> pd.DataFrame:
    """Compute Reach percentile threshold per segment from train data.

    Uses train only to avoid test leakage. Threshold = value at which
    (percentile * 100)% of segment's Reach values are below.

    Args:
        train_df: Training data with Segment and Reach columns.
        percentile: Fraction (e.g., 0.9 for top 10%).

    Returns:
        DataFrame with columns Segment, threshold.
    """
    thresholds = train_df.groupby("Segment")["Reach"].quantile(percentile).reset_index()
    thresholds.columns = ["Segment", "threshold"]
    return thresholds


def apply_trending_label(
    df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary Trending label: 1 if Reach >= segment threshold else 0.

    Joins thresholds by Segment. Rows with missing thresholds use global
    fallback (percentile of Reach in df).

    Args:
        df: Scored DataFrame with Segment and Reach.
        thresholds_df: Output from compute_segment_thresholds.

    Returns:
        Copy of df with added Trending column.
    """
    out = df.copy()
    merged = out.merge(
        thresholds_df[["Segment", "threshold"]],
        on="Segment",
        how="left",
    )
    missing = merged["threshold"].isna()
    if missing.any():
        fallback_thr = thresholds_df["threshold"].median()
        merged.loc[missing, "threshold"] = fallback_thr
    merged["Trending"] = (merged["Reach"] >= merged["threshold"]).astype(int)
    out["Trending"] = merged["Trending"].values
    return out
