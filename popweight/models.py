"""Trending classifier training."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLS = [
    "Score",
    "Platform",
    "Post Type",
    "Weekday Type",
    "Time Periods",
    "Age Group",
    "Sentiment",
]
CAT_COLS = [
    "Platform",
    "Post Type",
    "Weekday Type",
    "Time Periods",
    "Age Group",
    "Sentiment",
]
TARGET_COL = "Trending"


def _get_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Extract feature matrix and column names. Fill missing categoricals."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    if "Score" not in available:
        raise ValueError("DataFrame must have 'Score' column")
    out = df[available].copy()
    for col in CAT_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    return out, available


def train_trending_classifier(train_scored_labeled_df: pd.DataFrame):
    """Train Gradient Boosting classifier to predict Trending.

    Features: Score (must), Platform, Post Type, Weekday Type, Time Periods,
    Age Group, Sentiment. Categoricals are one-hot encoded.

    Args:
        train_scored_labeled_df: DataFrame with Score, Trending, and
            categorical columns.

    Returns:
        Fitted sklearn Pipeline (preprocessor + classifier).
    """
    X_raw, available = _get_X(train_scored_labeled_df)
    y = train_scored_labeled_df[TARGET_COL]

    numeric_cols = [c for c in ["Score"] if c in available]
    cat_cols = [c for c in CAT_COLS if c in available]

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No features available for training")
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = GradientBoostingClassifier(random_state=0)
    pipe = Pipeline([("preprocess", preprocessor), ("clf", model)])
    pipe.fit(X_raw, y)
    return pipe


def _optimal_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find threshold in [0.01, 0.99] step 0.01 that maximizes F1 on train."""
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.01, 1.0, 0.01):
        yp = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def predict_trending(
    model,
    test_scored_labeled_df: pd.DataFrame,
    threshold: float | None = None,
    train_scored_labeled_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict Trending class and probability for positive class.

    If threshold is None, selects threshold on train by maximizing F1
    (scan 0.01 to 0.99 step 0.01). Requires train_scored_labeled_df.

    Args:
        model: Fitted pipeline from train_trending_classifier.
        test_scored_labeled_df: DataFrame with same feature columns as train.
        threshold: Decision threshold (default None = optimize on train).
        train_scored_labeled_df: Required when threshold is None.

    Returns:
        Tuple of (y_pred, y_prob, threshold) where y_pred is class labels
        (0/1), y_prob is P(Trending=1), and threshold is the one used.
    """
    X_test, _ = _get_X(test_scored_labeled_df)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    if threshold is not None:
        thr = float(threshold)
    else:
        if train_scored_labeled_df is None:
            raise ValueError("train_scored_labeled_df required when threshold is None")
        X_train, _ = _get_X(train_scored_labeled_df)
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_true_train = train_scored_labeled_df[TARGET_COL].to_numpy()
        thr = _optimal_threshold_f1(y_true_train, y_prob_train)

    y_pred_test = (y_prob_test >= thr).astype(int)
    return y_pred_test, y_prob_test, thr
