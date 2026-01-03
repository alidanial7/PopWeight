"""Cross-sectional analysis models for engagement weight extraction."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm


def train_segment_model_linear(
    df_segment: pd.DataFrame,
    min_samples: int = 10,
    target_col: str = "Reach_log",
) -> tuple[LinearRegression, float, int] | None:
    """
    Train a Linear Regression model for a specific segment.

    Parameters
    ----------
    df_segment : pd.DataFrame
        DataFrame containing data for a specific Platform-PostType segment.
    min_samples : int, default 10
        Minimum number of samples required to train a model.

    Returns
    -------
    tuple[LinearRegression, float, int] | None
        Tuple of (model, r2_score, n_samples) if successful, None otherwise.
    """
    if len(df_segment) < min_samples:
        return None

    # Prepare features and target
    feature_cols = ["Likes_log_log", "Comments_log_log", "Shares_log_log"]

    # Check if all required columns exist
    missing_cols = [
        col for col in feature_cols + [target_col] if col not in df_segment.columns
    ]
    if missing_cols:
        return None

    X = df_segment[feature_cols].values
    y = df_segment[target_col].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate R-squared
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return model, r2, len(df_segment)


def train_segment_model_rf(
    df_segment: pd.DataFrame,
    min_samples: int = 10,
    target_col: str = "Engagement_Rate",
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, float, int] | None:
    """
    Train a Random Forest Regressor for a specific segment.

    Parameters
    ----------
    df_segment : pd.DataFrame
        DataFrame containing data for a specific Platform-PostType segment.
    min_samples : int, default 10
        Minimum number of samples required to train a model.
    target_col : str, default "Engagement_Rate"
        Target column name.
    n_estimators : int, default 100
        Number of trees in the Random Forest.
    random_state : int, default 42
        Random state for reproducibility.

    Returns
    -------
    tuple[RandomForestRegressor, float, int] | None
        Tuple of (model, r2_score, n_samples) if successful, None otherwise.
    """
    if len(df_segment) < min_samples:
        return None

    # Prepare features (include Engagement_Density if available)
    feature_cols = ["Likes_log_log", "Comments_log_log", "Shares_log_log"]
    if "Engagement_Density" in df_segment.columns:
        feature_cols.append("Engagement_Density")

    # Check if all required columns exist
    missing_cols = [
        col for col in feature_cols + [target_col] if col not in df_segment.columns
    ]
    if missing_cols:
        return None

    X = df_segment[feature_cols].values
    y = df_segment[target_col].values

    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)

    # Calculate R-squared
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return model, r2, len(df_segment)


def normalize_weights(
    alpha: float, beta: float, gamma: float
) -> tuple[float, float, float]:
    """
    Normalize weights so they sum to 1.

    Parameters
    ----------
    alpha : float
        Weight for Likes.
    beta : float
        Weight for Comments.
    gamma : float
        Weight for Shares.

    Returns
    -------
    tuple[float, float, float]
        Normalized weights (alpha_norm, beta_norm, gamma_norm).
    """
    total = abs(alpha) + abs(beta) + abs(gamma)
    if total == 0:
        return 0.0, 0.0, 0.0

    alpha_norm = abs(alpha) / total
    beta_norm = abs(beta) / total
    gamma_norm = abs(gamma) / total

    return alpha_norm, beta_norm, gamma_norm


def analyze_engagement_weights(
    df: pd.DataFrame,
    platform_col: str = "Platform",
    post_type_col: str = "Post Type",
    min_samples: int = 10,
    show_progress: bool = False,
    model_type: str = "linear",
    target_col: str = "Reach_log",
) -> tuple[pd.DataFrame, dict | None]:
    """
    Perform cross-sectional analysis of engagement weights.

    Trains separate models (Linear Regression or Random Forest) for each
    Platform-PostType combination and extracts normalized weights.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with engagement features.
    platform_col : str, default "Platform"
        Name of the platform column.
    post_type_col : str, default "Post Type"
        Name of the post type column.
    min_samples : int, default 10
        Minimum number of samples required per segment.
    show_progress : bool, default False
        Whether to show progress bar during analysis.
    model_type : str, default "linear"
        Model type: "linear" for Linear Regression, "rf" for Random Forest.
    target_col : str, default "Reach_log"
        Target column name.

    Returns
    -------
    tuple[pd.DataFrame, dict | None]
        Results table with columns: Platform, Post Type, Alpha_Likes,
        Beta_Comments, Gamma_Shares, R_Squared, N_Samples, Model_Type,
        and optionally a dictionary of trained models (for RF).
    """
    results = []
    models_dict = {}  # Store models for RF predictions

    # Get unique combinations
    platforms = df[platform_col].unique()
    post_types = df[post_type_col].unique()

    # Create iterator with or without progress bar
    if show_progress:
        iterator = tqdm(
            [(p, pt) for p in platforms for pt in post_types],
            desc="  Training models",
            unit="segment",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
    else:
        iterator = [(p, pt) for p in platforms for pt in post_types]

    for platform, post_type in iterator:
        # Filter segment
        mask = (df[platform_col] == platform) & (df[post_type_col] == post_type)
        df_segment = df[mask].copy()

        if len(df_segment) == 0:
            continue

        # Train model based on model_type
        if model_type == "rf":
            model_result = train_segment_model_rf(
                df_segment, min_samples=min_samples, target_col=target_col
            )
        else:
            model_result = train_segment_model_linear(
                df_segment, min_samples=min_samples, target_col=target_col
            )

        if model_result is None:
            continue

        model, r2, n_samples = model_result

        # Extract weights based on model type
        if model_type == "rf":
            # Use feature importances from Random Forest
            feature_importances = model.feature_importances_
            # Feature order: Likes_log_log, Comments_log_log, Shares_log_log,
            # [Engagement_Density]
            alpha = feature_importances[0]  # Likes_log_log
            beta = feature_importances[1]  # Comments_log_log
            gamma = feature_importances[2]  # Shares_log_log
            intercept = 0.0  # RF doesn't have explicit intercept
            # Store model for prediction
            models_dict[(platform, post_type)] = model
        else:
            # Linear Regression coefficients
            alpha = model.coef_[0]  # Likes_log_log
            beta = model.coef_[1]  # Comments_log_log
            gamma = model.coef_[2]  # Shares_log_log
            intercept = model.intercept_  # Bias term

        # Normalize weights (for display, but keep raw for prediction)
        alpha_norm, beta_norm, gamma_norm = normalize_weights(alpha, beta, gamma)

        # Store results (keep both normalized and raw coefficients)
        results.append(
            {
                "Platform": platform,
                "Post Type": post_type,
                "Alpha_Likes": alpha_norm,
                "Beta_Comments": beta_norm,
                "Gamma_Shares": gamma_norm,
                "Alpha_Raw": alpha,  # Raw coefficient/importance
                "Beta_Raw": beta,  # Raw coefficient/importance
                "Gamma_Raw": gamma,  # Raw coefficient/importance
                "Intercept": intercept,  # Intercept (0 for RF)
                "R_Squared": r2,
                "N_Samples": n_samples,
                "Model_Type": model_type,
                "Target_Col": target_col,
            }
        )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        empty_df = pd.DataFrame(
            columns=[
                "Platform",
                "Post Type",
                "Alpha_Likes",
                "Beta_Comments",
                "Gamma_Shares",
                "R_Squared",
                "N_Samples",
            ]
        )
        return empty_df, None

    # Return models_dict only for RF, None for linear
    return results_df, (models_dict if model_type == "rf" else None)


def extract_weights_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and format weights table for display.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from analyze_engagement_weights.

    Returns
    -------
    pd.DataFrame
        Formatted table with rounded values for display.
    """
    display_df = results_df.copy()

    # Round values for display
    display_df["Alpha_Likes"] = display_df["Alpha_Likes"].round(4)
    display_df["Beta_Comments"] = display_df["Beta_Comments"].round(4)
    display_df["Gamma_Shares"] = display_df["Gamma_Shares"].round(4)
    display_df["R_Squared"] = display_df["R_Squared"].round(4)

    # Sort by Platform and Post Type
    display_df = display_df.sort_values(["Platform", "Post Type"]).reset_index(
        drop=True
    )

    return display_df
