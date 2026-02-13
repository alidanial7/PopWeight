"""Weight learning per segment via linear regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATURE_COLS = ["Likes_ll", "Comments_ll", "Shares_ll"]
ER_EPS = 1e-10


def compute_log_ER_proxy(df: pd.DataFrame) -> pd.Series:
    """Compute log(ER_proxy) target aligned with Trending.

    Eng = Likes + Comments + Shares
    ER_proxy = (Eng / Reach.clip(lower=1)).clip(lower=ER_EPS)
    Returns log(ER_proxy).

    Requires: Likes, Comments, Shares, Reach.
    """
    eng = (
        df["Likes"].astype(float)
        + df["Comments"].astype(float)
        + df["Shares"].astype(float)
    )
    reach_safe = df["Reach"].astype(float).clip(lower=1)
    er_proxy = (eng / reach_safe).clip(lower=ER_EPS)
    return np.log(er_proxy)


def _fit_linreg(X: pd.DataFrame, y: pd.Series) -> tuple[float, float, float, float]:
    """Fit linear regression, return (intercept, alpha, beta, gamma)."""
    model = LinearRegression()
    model.fit(X, y)
    intercept = float(model.intercept_)
    coefs = model.coef_
    return intercept, float(coefs[0]), float(coefs[1]), float(coefs[2])


def _fit_segment(
    seg_df: pd.DataFrame,
    min_samples: int,
) -> tuple[pd.Series | None, str]:
    """Fit weights for one segment. Returns (row Series or None, strategy)."""
    n = len(seg_df)
    if n < min_samples:
        return None, "insufficient"
    X = seg_df[FEATURE_COLS]
    y = compute_log_ER_proxy(seg_df)
    intercept, alpha, beta, gamma = _fit_linreg(X, y)
    y_pred = (
        X["Likes_ll"] * alpha
        + X["Comments_ll"] * beta
        + X["Shares_ll"] * gamma
        + intercept
    )  # noqa: E501
    r2 = float(r2_score(y, y_pred)) if len(y) > 1 else 0.0
    return pd.Series(
        {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "intercept": intercept,
            "n_train": n,
            "r2_train": r2,
            "strategy": "segment",
        }
    ), "segment"


def fit_segment_weights(
    train_df: pd.DataFrame,
    seed: int | None = None,
    min_segment_samples: int = 20,
) -> pd.DataFrame:
    """Learn (alpha, beta, gamma) per segment to predict log(ER_proxy).

    Linear regression: X = [Likes_ll, Comments_ll, Shares_ll],
    y = log(ER_proxy) where ER_proxy = Eng/Reach.
    Segments with < min_segment_samples use global fallback weights.

    Args:
        train_df: Training data with Segment, FEATURE_COLS, Likes,
            Comments, Shares, Reach.
        seed: Optional seed for reproducibility (stored in output).
        min_segment_samples: Minimum rows to fit segment-specific weights.

    Returns:
        DataFrame with columns: seed, Platform, Post Type, Segment, alpha,
        beta, gamma, intercept, n_train, r2_train, strategy.
    """
    X_global = train_df[FEATURE_COLS]
    y_global = compute_log_ER_proxy(train_df)
    global_intercept, global_alpha, global_beta, global_gamma = _fit_linreg(
        X_global, y_global
    )

    rows: list[dict] = []
    for segment, seg_df in train_df.groupby("Segment", sort=True):
        platform = seg_df["Platform"].iloc[0]
        post_type = seg_df["Post Type"].iloc[0]
        row_dict: dict = {
            "seed": seed,
            "Platform": platform,
            "Post Type": post_type,
            "Segment": segment,
        }
        sr, strategy = _fit_segment(seg_df, min_segment_samples)
        if strategy == "insufficient":
            row_dict.update(
                {
                    "alpha": global_alpha,
                    "beta": global_beta,
                    "gamma": global_gamma,
                    "intercept": global_intercept,
                    "n_train": len(seg_df),
                    "r2_train": None,
                    "strategy": "global_fallback",
                }
            )
        else:
            row_dict.update(sr.to_dict())
        rows.append(row_dict)

    return pd.DataFrame(rows)
