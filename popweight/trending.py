"""Trending label construction based on engagement rate proxy (ER_proxy)."""

import pandas as pd

REACH_EPS = 1  # Matches MIN_REACH; cleaning drops rows with Reach < 1


def _add_er_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Add Eng and ER_proxy columns.

    Eng = Likes + Comments + Shares
    ER_proxy = Eng / Reach (safe division; Reach=0 rows removed in cleaning).

    Requires: Likes, Comments, Shares, Reach.
    """
    out = df.copy()
    eng = (
        out["Likes"].astype(float)
        + out["Comments"].astype(float)
        + out["Shares"].astype(float)
    )
    reach_safe = out["Reach"].astype(float).clip(lower=REACH_EPS)
    out["Eng"] = eng
    out["ER_proxy"] = (eng / reach_safe).astype(float)
    return out


def compute_segment_thresholds(
    train_df: pd.DataFrame,
    percentile: float,
) -> pd.DataFrame:
    """Compute ER_proxy percentile threshold per segment from train data.

    Uses train only to avoid test leakage.
    Threshold = value at which (percentile * 100)% of segment's
    ER_proxy values are below.

    Args:
        train_df: Training data with Segment, Likes, Comments, Shares, Reach.
        percentile: Fraction (e.g., 0.9 for top 10%).

    Returns:
        DataFrame with columns Segment, threshold.
    """
    work = _add_er_proxy(train_df)
    thresholds = work.groupby("Segment")["ER_proxy"].quantile(percentile).reset_index()
    thresholds.columns = ["Segment", "threshold"]
    thresholds["threshold"] = thresholds["threshold"].astype(float)
    return thresholds


def apply_trending_label(
    df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary Trending label: 1 if ER_proxy >= segment threshold else 0.

    Computes Eng and ER_proxy from Likes, Comments, Shares, Reach.
    Rows with missing segment thresholds use median threshold of all segments.

    Args:
        df: DataFrame with Segment, Likes, Comments, Shares, Reach.
        thresholds_df: Output from compute_segment_thresholds.

    Returns:
        Copy of df with added Trending column.
    """
    out = _add_er_proxy(df)
    merged = out.merge(
        thresholds_df[["Segment", "threshold"]],
        on="Segment",
        how="left",
    )
    missing = merged["threshold"].isna()
    if missing.any():
        fallback_thr = thresholds_df["threshold"].median()
        merged.loc[missing, "threshold"] = fallback_thr
    merged["Trending"] = (merged["ER_proxy"] >= merged["threshold"]).astype(int)
    result = df.copy()
    result["Trending"] = merged["Trending"].values
    return result
