"""Scoring: compute engagement Score via learned weights."""

import pandas as pd


def apply_scores(df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Score = intercept + alpha*Likes_ll + beta*Comments_ll + gamma*Shares_ll.

    Joins weights by (Platform, Post Type). Rows with missing weights use
    fallback (mean of all segment weights).

    Args:
        df: DataFrame with Platform, Post Type, Likes_ll, Comments_ll,
            Shares_ll.
        weights_df: Weights from fit_segment_weights (Platform, Post Type,
            alpha, beta, gamma, intercept).

    Returns:
        Copy of df with added Score column.
    """
    out = df.copy()
    merge_cols = ["Platform", "Post Type"]
    w = weights_df[merge_cols + ["alpha", "beta", "gamma", "intercept"]].copy()
    merged = out.merge(w, on=merge_cols, how="left")

    missing = merged["alpha"].isna()
    if missing.any():
        fallback = weights_df[["alpha", "beta", "gamma", "intercept"]].mean()
        merged.loc[missing, "alpha"] = fallback["alpha"]
        merged.loc[missing, "beta"] = fallback["beta"]
        merged.loc[missing, "gamma"] = fallback["gamma"]
        merged.loc[missing, "intercept"] = fallback["intercept"]

    merged["Score"] = (
        merged["intercept"]
        + merged["alpha"] * merged["Likes_ll"]
        + merged["beta"] * merged["Comments_ll"]
        + merged["gamma"] * merged["Shares_ll"]
    )
    out["Score"] = merged["Score"].values
    return out
