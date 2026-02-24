"""Trending classifier training."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLS = [
    "Score",
    "Eng_log",
    "Platform",
    "Post Type",
    "Weekday Type",
    "Time Periods",
    "Age Group",
    "Sentiment",
]
NUMERIC_COLS = ["Score", "Eng_log"]
CAT_COLS = [
    "Platform",
    "Post Type",
    "Weekday Type",
    "Time Periods",
    "Age Group",
    "Sentiment",
]
TARGET_COL = "Trending"


def _add_eng_log(df: pd.DataFrame) -> pd.DataFrame:
    """Add Eng_log = log(Likes + Comments + Shares + 1). Modifies copy."""
    out = df.copy()
    eng = (
        out["Likes"].astype(float)
        + out["Comments"].astype(float)
        + out["Shares"].astype(float)
    )
    out["Eng_log"] = np.log(eng + 1)
    return out


def _get_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Extract feature matrix and column names. Fill missing categoricals."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    if "Score" not in available or "Eng_log" not in available:
        raise ValueError("DataFrame must have 'Score' and 'Eng_log' columns")
    out = df[available].copy()
    for col in CAT_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    return out, available


def train_trending_classifier(
    train_scored_labeled_df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 0,
) -> tuple:
    """Train Gradient Boosting classifier to predict Trending.

    Splits train into train_sub (1-val_ratio) and val_sub (val_ratio)
    stratified by Trending. Fits on train_sub, selects threshold that
    maximizes F1 on val_sub.

    Features: Score, Eng_log (must), Platform, Post Type, Weekday Type,
    Time Periods, Age Group, Sentiment. Eng_log = log(Likes+Comments+Shares+1).
    Categoricals are one-hot encoded.

    Args:
        train_scored_labeled_df: DataFrame with Score, Likes, Comments,
            Shares, Trending, and categorical columns.
        val_ratio: Fraction for validation (default 0.2).
        random_state: Random state for stratified split.

    Returns:
        Tuple of (fitted Pipeline, chosen_threshold).
    """
    train_df = _add_eng_log(train_scored_labeled_df)
    X_raw, available = _get_X(train_df)
    y = train_df[TARGET_COL]

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_raw,
            y,
            test_size=val_ratio,
            stratify=y,
            random_state=random_state,
        )
    except ValueError:
        # Fallback when stratification impossible (e.g. one class rare)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_raw,
            y,
            test_size=val_ratio,
            random_state=random_state,
        )

    numeric_cols = [c for c in NUMERIC_COLS if c in available]
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
    model = GradientBoostingClassifier(random_state=random_state)
    pipe = Pipeline([("preprocess", preprocessor), ("clf", model)])
    pipe.fit(X_tr, y_tr)

    y_prob_val = pipe.predict_proba(X_val)[:, 1]
    chosen_thr = _optimal_threshold_f1(y_val.to_numpy(), y_prob_val)

    return pipe, chosen_thr


def _optimal_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find threshold in [0.01, 0.99] step 0.01 that maximizes F1."""
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
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict Trending class and probability for positive class.

    Uses the fixed threshold from train_trending_classifier (chosen on val).
    Do not select threshold on the same data used to fit the classifier.

    Args:
        model: Fitted pipeline from train_trending_classifier.
        test_scored_labeled_df: DataFrame with same feature columns as train.
        threshold: Decision threshold (from train_trending_classifier).

    Returns:
        Tuple of (y_pred, y_prob, threshold) where y_pred is class labels
        (0/1), y_prob is P(Trending=1), and threshold is the one used.
    """
    test_df = _add_eng_log(test_scored_labeled_df)
    X_test, _ = _get_X(test_df)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    thr = float(threshold)
    y_pred_test = (y_prob_test >= thr).astype(int)
    return y_pred_test, y_prob_test, thr
