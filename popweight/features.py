"""Feature engineering and transformed interaction features."""

import numpy as np
import pandas as pd


def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add double-log and log transforms for interaction and reach columns.

    Creates:
    - Likes_ll, Comments_ll, Shares_ll: log(log(x + 1) + 1)
    - Reach_log: log(Reach + 1)

    Args:
        df: Cleaned DataFrame with Likes, Comments, Shares, Reach.

    Returns:
        DataFrame with transform columns added (copy).
    """
    out = df.copy()
    for col, out_col in [
        ("Likes", "Likes_ll"),
        ("Comments", "Comments_ll"),
        ("Shares", "Shares_ll"),
    ]:
        x = out[col].astype(float) + 1
        out[out_col] = np.log(np.log(x) + 1)
    out["Reach_log"] = np.log(out["Reach"].astype(float) + 1)
    return out


def add_segment_key(
    df: pd.DataFrame,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Add Segment column and strip whitespace from categorical columns.

    Segment = keys[0] + "__" + keys[1] (e.g., Platform__Post Type).
    Also strips whitespace from categorical columns: Platform, Post Type,
    Weekday Type, Time Periods, Age Group, Sentiment.

    Args:
        df: DataFrame with segment key columns.
        keys: Column names for segment key. Defaults to ["Platform",
            "Post Type"].

    Returns:
        DataFrame with Segment column and cleaned categories (copy).
    """
    if keys is None:
        keys = ["Platform", "Post Type"]
    out = df.copy()
    cat_cols = [
        "Platform",
        "Post Type",
        "Weekday Type",
        "Time Periods",
        "Age Group",
        "Sentiment",
    ]
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    out["Segment"] = out[keys[0]].astype(str) + "__" + out[keys[1]].astype(str)
    return out
