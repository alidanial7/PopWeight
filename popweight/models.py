"""Trending classifier training."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
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


def predict_trending(
    model,
    test_scored_labeled_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict Trending class and probability for positive class.

    Args:
        model: Fitted pipeline from train_trending_classifier.
        test_scored_labeled_df: DataFrame with same feature columns as train.

    Returns:
        Tuple of (y_pred, y_prob) where y_pred is class labels (0/1) and
        y_prob is P(Trending=1).
    """
    X_raw, _ = _get_X(test_scored_labeled_df)
    y_pred = model.predict(X_raw)
    y_prob = model.predict_proba(X_raw)[:, 1]
    return y_pred.astype(int), y_prob
