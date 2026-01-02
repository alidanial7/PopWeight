"""Data preprocessing utilities for social media engagement data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def log_log_normalize(series: pd.Series) -> pd.Series:
    """
    Apply log-log normalization to a series.

    Formula: x' = log(log(x + 1) + 1)

    Parameters
    ----------
    series : pd.Series
        Input series to normalize.

    Returns
    -------
    pd.Series
        Log-log normalized series.
    """
    return np.log(np.log(series + 1) + 1)


def extract_temporal_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Extract temporal features from timestamp column.

    Extracts:
    - Hour_of_day: Hour of the day (0-23)
    - Day_of_week: Day of the week (0=Monday, 6=Sunday)
    - Is_Weekend: Boolean indicating if the day is weekend (Saturday/Sunday)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the timestamp column.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features added.
    """
    df = df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Extract temporal features
    df["Hour_of_day"] = df[timestamp_col].dt.hour
    df["Day_of_week"] = df[timestamp_col].dt.dayofweek
    df["Is_Weekend"] = df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: str | None = None,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Preprocess social media engagement data.

    Performs the following transformations:
    1. Handle missing values (median for numerical, 'None' for categorical)
    2. Log-log normalization for Likes, Comments, Shares
    3. Temporal engineering from Post Timestamp
    4. One-hot encoding for Platform, Post Type, Sentiment
    5. StandardScaler for numerical features (Audience Age)
    6. Target preparation with log transformation for Reach/Impressions

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw data.
    target_col : str, optional
        Target column name ('Reach' or 'Impressions'). If None, target
        transformation is skipped.
    fit_scaler : bool, default True
        Whether to fit a new scaler or use the provided one.
    scaler : StandardScaler, optional
        Pre-fitted scaler to use for transformation. Only used if
        fit_scaler is False.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler | None]
        Preprocessed DataFrame and fitted scaler (if fit_scaler=True).
    """
    df = df.copy()

    # Step 1: Handle missing values
    # Fill numerical columns with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Fill categorical columns with 'None'
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("None")

    # Step 2: Log-log normalization for Likes, Comments, Shares
    engagement_cols = ["Likes", "Comments", "Shares"]
    for col in engagement_cols:
        if col in df.columns:
            df[f"{col}_log_log"] = log_log_normalize(df[col])
            # Keep original column for reference
            # Optionally drop: df = df.drop(columns=[col])

    # Step 3: Temporal engineering from Post Timestamp
    timestamp_col = "Post Timestamp"
    if timestamp_col in df.columns:
        df = extract_temporal_features(df, timestamp_col)

    # Step 4: One-hot encoding for Platform, Post Type, Sentiment
    categorical_to_encode = ["Platform", "Post Type", "Sentiment"]
    for col in categorical_to_encode:
        if col in df.columns:
            # Get unique values for one-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col.replace(" ", "_"), dtype=int)
            df = pd.concat([df, dummies], axis=1)
            # Keep original column for reference
            # Optionally drop: df = df.drop(columns=[col])

    # Step 5: StandardScaler for numerical features (Audience Age)
    age_col = "Audience Age"
    if age_col in df.columns:
        if fit_scaler:
            scaler = StandardScaler()
            df[[age_col]] = scaler.fit_transform(df[[age_col]])
        elif scaler is not None:
            df[[age_col]] = scaler.transform(df[[age_col]])

    # Step 6: Target preparation with log transformation
    if target_col and target_col in df.columns:
        df[f"{target_col}_log"] = np.log(df[target_col] + 1)

    return df, scaler


def preprocess_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess training data (fits scalers).

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame with raw data.

    Returns
    -------
    pd.DataFrame
        Preprocessed training DataFrame.
    """
    df_processed, _ = preprocess_data(df, target_col="Reach", fit_scaler=True)
    return df_processed


def preprocess_test_data(
    df: pd.DataFrame, scaler: StandardScaler | None = None
) -> pd.DataFrame:
    """
    Preprocess test data (uses provided scaler or fits new one).

    Parameters
    ----------
    df : pd.DataFrame
        Test DataFrame with raw data.
    scaler : StandardScaler, optional
        Pre-fitted scaler from training data. If None, fits a new scaler.

    Returns
    -------
    pd.DataFrame
        Preprocessed test DataFrame.
    """
    fit_scaler = scaler is None
    df_processed, _ = preprocess_data(
        df, target_col="Reach", fit_scaler=fit_scaler, scaler=scaler
    )
    return df_processed
