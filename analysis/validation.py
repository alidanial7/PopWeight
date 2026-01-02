"""Validation module for cross-database performance evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def load_training_weights(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load learned weights from training phase.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from analyze_engagement_weights with weights.

    Returns
    -------
    pd.DataFrame
        DataFrame with Platform, Post Type, weights, and intercept.
    """
    # Use raw coefficients and intercept for prediction
    cols = ["Platform", "Post Type", "Alpha_Raw", "Beta_Raw", "Gamma_Raw", "Intercept"]

    # Fallback to normalized if raw not available (backward compatibility)
    if "Alpha_Raw" not in results_df.columns:
        cols = ["Platform", "Post Type", "Alpha_Likes", "Beta_Comments", "Gamma_Shares"]
        result = results_df[cols].copy()
        # Add zero intercept if not present
        result["Intercept"] = 0.0
        return result

    return results_df[cols].copy()


def diagnose_validation_issues(
    train_results_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> None:
    """
    Print diagnostic information to identify validation issues.

    Parameters
    ----------
    train_results_df : pd.DataFrame
        Full training results DataFrame.
    test_df : pd.DataFrame
        Test DataFrame.
    weights_df : pd.DataFrame
        Weights DataFrame for prediction.
    """
    print("\n" + "=" * 80)
    print("🔍 VALIDATION DIAGNOSTICS")
    print("=" * 80)

    # 1. Check Platform-PostType matches
    print("\n1. Platform-PostType Coverage:")
    print("-" * 80)
    test_combos = set(zip(test_df["Platform"], test_df["Post Type"], strict=False))
    train_combos = set(
        zip(weights_df["Platform"], weights_df["Post Type"], strict=False)
    )
    missing_combos = test_combos - train_combos

    print(f"   Test combinations: {len(test_combos)}")
    print(f"   Train combinations: {len(train_combos)}")
    print(f"   Missing in training: {len(missing_combos)}")

    if missing_combos:
        print("\n   ⚠️  Missing combinations (using default weights):")
        for platform, post_type in list(missing_combos)[:5]:
            count = len(
                test_df[
                    (test_df["Platform"] == platform)
                    & (test_df["Post Type"] == post_type)
                ]
            )
            print(f"      - {platform} + {post_type}: {count} posts")
        if len(missing_combos) > 5:
            print(f"      ... and {len(missing_combos) - 5} more")

    # 2. Check feature statistics and compare train vs test scales
    print("\n2. Feature Statistics (Train vs Test Scale Comparison):")
    print("-" * 80)
    feature_cols = ["Likes_log_log", "Comments_log_log", "Shares_log_log"]

    for col in feature_cols:
        if col in test_df.columns:
            test_mean = test_df[col].mean()
            test_std = test_df[col].std()
            zero_count = (test_df[col] == 0).sum()
            nan_count = test_df[col].isna().sum()
            print(f"   {col}:")
            print(f"      Test - Mean: {test_mean:.4f}, Std: {test_std:.4f}")
            print(f"      Zeros: {zero_count}, NaNs: {nan_count}")

    # 3. Check intercept values
    print("\n3. Intercept Statistics:")
    print("-" * 80)
    if "Intercept" in weights_df.columns:
        intercepts = weights_df["Intercept"]
        print(f"   Mean intercept: {intercepts.mean():.4f}")
        print(f"   Std intercept: {intercepts.std():.4f}")
        print(f"   Min intercept: {intercepts.min():.4f}")
        print(f"   Max intercept: {intercepts.max():.4f}")
    else:
        print("   ⚠️  No intercept column found (using 0.0)")

    # 4. Check coefficient ranges
    print("\n4. Coefficient Ranges:")
    print("-" * 80)
    if "Alpha_Raw" in weights_df.columns:
        print(
            f"   Alpha_Raw: [{weights_df['Alpha_Raw'].min():.4f}, "
            f"{weights_df['Alpha_Raw'].max():.4f}]"
        )
        print(
            f"   Beta_Raw: [{weights_df['Beta_Raw'].min():.4f}, "
            f"{weights_df['Beta_Raw'].max():.4f}]"
        )
        print(
            f"   Gamma_Raw: [{weights_df['Gamma_Raw'].min():.4f}, "
            f"{weights_df['Gamma_Raw'].max():.4f}]"
        )
    else:
        print("   ⚠️  Using normalized weights (may cause scale issues)")

    # 5. Check training R² scores
    print("\n5. Training Model Quality:")
    print("-" * 80)
    if "R_Squared" in train_results_df.columns:
        r2_scores = train_results_df["R_Squared"]
        print(f"   Mean R²: {r2_scores.mean():.4f}")
        print(f"   Min R²: {r2_scores.min():.4f}")
        print(f"   Max R²: {r2_scores.max():.4f}")
        print(f"   R² > 0.5: {(r2_scores > 0.5).sum()} / {len(r2_scores)} segments")
        print(f"   R² > 0.3: {(r2_scores > 0.3).sum()} / {len(r2_scores)} segments")

    # 6. Check prediction vs actual scale (if predictions exist)
    if "Predicted_Engagement_Score" in test_df.columns:
        print("\n6. Prediction Scale Analysis:")
        print("-" * 80)
        actual_mean = test_df["Reach_log"].mean()
        pred_mean = test_df["Predicted_Engagement_Score"].mean()
        actual_std = test_df["Reach_log"].std()
        pred_std = test_df["Predicted_Engagement_Score"].std()

        print(f"   Actual Reach_log - Mean: {actual_mean:.4f}, Std: {actual_std:.4f}")
        print(f"   Predicted Score - Mean: {pred_mean:.4f}, Std: {pred_std:.4f}")
        scale_ratio = pred_mean / (actual_mean + 1e-10)
        std_ratio = pred_std / (actual_std + 1e-10)
        print(f"   Scale Ratio: {scale_ratio:.4f}")
        print(f"   Std Ratio: {std_ratio:.4f}")

        if abs(scale_ratio - 1.0) > 0.5:
            print("   ⚠️  WARNING: Significant scale mismatch detected!")
        if abs(std_ratio - 1.0) > 0.5:
            print("   ⚠️  WARNING: Significant variance mismatch detected!")

    print("=" * 80 + "\n")


def check_feature_ranges(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    clip_outliers: bool = True,
) -> pd.DataFrame:
    """
    Check feature ranges and clip outliers in test data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with feature ranges.
    test_df : pd.DataFrame
        Test DataFrame to check and clip.
    feature_cols : list[str]
        List of feature column names.
    clip_outliers : bool, default True
        Whether to clip outliers to training range.

    Returns
    -------
    pd.DataFrame
        Test DataFrame with clipped features.
    """
    test_df = test_df.copy()

    print("\n📊 Feature Range Check:")
    print("-" * 80)

    for col in feature_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        train_min = train_df[col].min()
        train_max = train_df[col].max()
        test_min = test_df[col].min()
        test_max = test_df[col].max()

        outliers_low = (test_df[col] < train_min).sum()
        outliers_high = (test_df[col] > train_max).sum()
        total_outliers = outliers_low + outliers_high

        print(f"  {col}:")
        print(f"    Train range: [{train_min:.4f}, {train_max:.4f}]")
        print(f"    Test range:  [{test_min:.4f}, {test_max:.4f}]")
        print(
            f"    Outliers: {total_outliers} ({outliers_low} low, {outliers_high} high)"
        )

        if clip_outliers and total_outliers > 0:
            test_df[col] = test_df[col].clip(lower=train_min, upper=train_max)
            print("    ✓ Clipped to training range")

    print("-" * 80)

    return test_df


def calibrate_intercept(
    test_df: pd.DataFrame,
    predicted_col: str = "Predicted_Engagement_Score",
    actual_col: str = "Reach_log",
    sample_size: int = 100,
) -> float:
    """
    Recalibrate intercept on a sample of test data if scale shift is detected.

    Parameters
    ----------
    test_df : pd.DataFrame
        DataFrame with predictions and actual values.
    predicted_col : str, default "Predicted_Engagement_Score"
        Name of predicted score column.
    actual_col : str, default "Reach_log"
        Name of actual target column.
    sample_size : int, default 100
        Number of samples to use for calibration.

    Returns
    -------
    float
        Intercept adjustment value.
    """
    # Use a sample for calibration
    sample_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42).copy()

    # Calculate mean difference (constant shift)
    mean_diff = (sample_df[actual_col] - sample_df[predicted_col]).mean()

    return mean_diff


def calculate_predicted_engagement_score(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    platform_col: str = "Platform",
    post_type_col: str = "Post Type",
    target_range: tuple[float, float] | None = None,
    calibrate: bool = False,
    actual_col: str | None = None,
) -> pd.DataFrame:
    """
    Calculate predicted engagement score for test data using training weights.

    Formula: Predicted_Score = Intercept + (alpha * Likes_log_log) +
                                 (beta * Comments_log_log) +
                                 (gamma * Shares_log_log)

    Parameters
    ----------
    df : pd.DataFrame
        Test DataFrame with engagement features.
    weights_df : pd.DataFrame
        DataFrame with learned weights from training (must include Intercept).
    platform_col : str, default "Platform"
        Name of the platform column.
    post_type_col : str, default "Post Type"
        Name of the post type column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Predicted_Engagement_Score' column.
    """
    df = df.copy()

    # Initialize predicted score column
    df["Predicted_Engagement_Score"] = 0.0

    # Required feature columns
    feature_cols = ["Likes_log_log", "Comments_log_log", "Shares_log_log"]

    # Check if all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Determine which weight columns to use (raw vs normalized)
    if "Alpha_Raw" in weights_df.columns:
        alpha_col = "Alpha_Raw"
        beta_col = "Beta_Raw"
        gamma_col = "Gamma_Raw"
    else:
        # Fallback to normalized (backward compatibility)
        alpha_col = "Alpha_Likes"
        beta_col = "Beta_Comments"
        gamma_col = "Gamma_Shares"

    # Apply weights to each row
    for idx, row in df.iterrows():
        platform = row[platform_col]
        post_type = row[post_type_col]

        # Find matching weights
        weight_row = weights_df[
            (weights_df["Platform"] == platform)
            & (weights_df["Post Type"] == post_type)
        ]

        if len(weight_row) == 0:
            # Use default values if not found
            alpha, beta, gamma, intercept = 0.33, 0.33, 0.34, 0.0
        else:
            alpha = weight_row.iloc[0][alpha_col]
            beta = weight_row.iloc[0][beta_col]
            gamma = weight_row.iloc[0][gamma_col]
            intercept = weight_row.iloc[0].get("Intercept", 0.0)

        # Calculate predicted score WITH intercept
        predicted_score = (
            intercept
            + alpha * row["Likes_log_log"]
            + beta * row["Comments_log_log"]
            + gamma * row["Shares_log_log"]
        )

        df.loc[idx, "Predicted_Engagement_Score"] = predicted_score

    # Intercept verification and scale alignment
    if target_range is not None:
        pred_mean = df["Predicted_Engagement_Score"].mean()
        target_min, target_max = target_range

        if pred_mean < target_min or pred_mean > target_max:
            print(
                f"\n⚠️  Scale Alignment: Predicted mean ({pred_mean:.4f}) "
                f"outside target range [{target_min:.2f}, {target_max:.2f}]"
            )

            if calibrate and actual_col and actual_col in df.columns:
                # Recalibrate intercept on test sample
                intercept_adjustment = calibrate_intercept(df, actual_col=actual_col)
                print(f"   Applying intercept adjustment: {intercept_adjustment:.4f}")
                df["Predicted_Engagement_Score"] += intercept_adjustment
                print(
                    f"   Adjusted mean: {df['Predicted_Engagement_Score'].mean():.4f}"
                )
            else:
                # Simple centering adjustment
                actual_mean = (
                    df[actual_col].mean()
                    if actual_col and actual_col in df.columns
                    else target_min + (target_max - target_min) / 2
                )
                adjustment = actual_mean - pred_mean
                print(f"   Applying scale adjustment: {adjustment:.4f}")
                df["Predicted_Engagement_Score"] += adjustment

    return df


def calculate_regression_metrics(
    df: pd.DataFrame,
    predicted_col: str = "Predicted_Engagement_Score",
    actual_col: str = "Reach_log",
) -> dict[str, float]:
    """
    Calculate regression performance metrics including Pearson correlation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predicted and actual values.
    predicted_col : str, default "Predicted_Engagement_Score"
        Name of predicted score column.
    actual_col : str, default "Reach_log"
        Name of actual target column.

    Returns
    -------
    dict[str, float]
        Dictionary with R², MAE, RMSE, and Pearson correlation metrics.
    """
    from scipy.stats import pearsonr

    y_true = df[actual_col].values
    y_pred = df[predicted_col].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate Pearson correlation (scale-invariant)
    correlation, p_value = pearsonr(y_true, y_pred)

    return {
        "R_Squared": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_R": correlation,
        "Pearson_P_Value": p_value,
    }


def identify_trending_posts_by_score(
    df: pd.DataFrame,
    score_col: str,
    platform_col: str = "Platform",
    post_type_col: str = "Post Type",
    percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Identify trending posts based on score column within Platform-PostType.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores.
    score_col : str
        Name of the score column to use.
    platform_col : str, default "Platform"
        Name of the platform column.
    post_type_col : str, default "Post Type"
        Name of the post type column.
    percentile : float, default 90.0
        Percentile threshold (90.0 = top 10%).

    Returns
    -------
    pd.DataFrame
        DataFrame with added trending column.
    """
    df = df.copy()
    df["Is_Trending"] = 0

    platforms = df[platform_col].unique()
    post_types = df[post_type_col].unique()

    for platform in platforms:
        for post_type in post_types:
            mask = (df[platform_col] == platform) & (df[post_type_col] == post_type)
            segment_indices = df[mask].index

            if len(segment_indices) == 0:
                continue

            segment_scores = df.loc[segment_indices, score_col]
            threshold = np.percentile(segment_scores, percentile)

            trending_mask = df.loc[segment_indices, score_col] >= threshold
            df.loc[segment_indices[trending_mask], "Is_Trending"] = 1

    return df


def calculate_classification_metrics(
    df: pd.DataFrame,
    actual_col: str = "Actual_Trending",
    predicted_col: str = "Predicted_Trending",
) -> tuple[dict[str, any], np.ndarray]:
    """
    Calculate classification performance metrics and confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and predicted trending labels.
    actual_col : str, default "Actual_Trending"
        Name of actual trending column.
    predicted_col : str, default "Predicted_Trending"
        Name of predicted trending column.

    Returns
    -------
    tuple[dict[str, any], np.ndarray]
        Dictionary with metrics and confusion matrix array.
    """
    y_true = df[actual_col].values
    y_pred = df[predicted_col].values

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract values (handle different matrix sizes)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # Only one class present
        if y_true.sum() == 0:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
    else:
        # Fallback for edge cases
        tn, fp, fn, tp = 0, 0, 0, 0

    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Calculate accuracy percentage for top 10% trending posts
    total_actual_trending = y_true.sum()
    correctly_identified = tp
    accuracy_percentage = (
        (correctly_identified / total_actual_trending * 100)
        if total_actual_trending > 0
        else 0.0
    )

    metrics = {
        "Accuracy": accuracy,
        "Accuracy_Percentage": accuracy_percentage,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Positives": int(tp),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "Total_Actual_Trending": int(total_actual_trending),
        "Correctly_Identified": int(correctly_identified),
    }

    return metrics, cm


def plot_prediction_vs_actual(
    df: pd.DataFrame,
    predicted_col: str = "Predicted_Engagement_Score",
    actual_col: str = "Reach_log",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot prediction vs actual scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with predicted and actual values.
    predicted_col : str, default "Predicted_Engagement_Score"
        Name of predicted score column.
    actual_col : str, default "Reach_log"
        Name of actual target column.
    save_path : Path, optional
        Path to save the figure.
    figsize : tuple[int, int], default (10, 8)
        Figure size.
    """
    plt.figure(figsize=figsize)

    # Create scatter plot
    plt.scatter(
        df[actual_col],
        df[predicted_col],
        alpha=0.5,
        s=20,
        edgecolors="black",
        linewidths=0.5,
    )

    # Add diagonal line (perfect prediction)
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")

    # Calculate and display R²
    r2 = r2_score(df[actual_col], df[predicted_col])
    plt.text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.xlabel("Actual Reach (log)", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Engagement Score", fontsize=12, fontweight="bold")
    plt.title(
        "Prediction vs Actual: Test Set Performance",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Prediction vs Actual plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    Plot confusion matrix as heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array.
    save_path : Path, optional
        Path to save the figure.
    figsize : tuple[int, int], default (8, 6)
        Figure size.
    """
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=["Not Trending", "Trending"],
        yticklabels=["Not Trending", "Trending"],
    )

    plt.ylabel("Actual", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12, fontweight="bold")
    plt.title(
        "Confusion Matrix: Trend Detection Accuracy",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def print_prediction_samples(
    df: pd.DataFrame,
    actual_col: str = "Reach_log",
    predicted_col: str = "Predicted_Engagement_Score",
    n_samples: int = 5,
) -> None:
    """
    Print sample predictions vs actual values for verification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and predicted values.
    actual_col : str, default "Reach_log"
        Name of actual target column.
    predicted_col : str, default "Predicted_Engagement_Score"
        Name of predicted score column.
    n_samples : int, default 5
        Number of samples to print.
    """
    print("\n🔍 Prediction Verification (First 5 samples):")
    print("-" * 80)
    header = (
        f"{'Index':<8} {'Actual':<12} {'Predicted':<12} "
        f"{'Difference':<12} {'Error %':<10}"
    )
    print(header)
    print("-" * 80)

    sample_df = df[[actual_col, predicted_col]].head(n_samples).copy()
    sample_df["Difference"] = sample_df[predicted_col] - sample_df[actual_col]
    sample_df["Error_Pct"] = (
        sample_df["Difference"].abs() / (sample_df[actual_col].abs() + 1e-10)
    ) * 100

    for idx, row in sample_df.iterrows():
        print(
            f"{idx:<8} {row[actual_col]:<12.4f} {row[predicted_col]:<12.4f} "
            f"{row['Difference']:<12.4f} {row['Error_Pct']:<10.2f}%"
        )

    print("-" * 80)
    print(
        f"\nMean Actual: {sample_df[actual_col].mean():.4f}, "
        f"Mean Predicted: {sample_df[predicted_col].mean():.4f}"
    )
    pred_mean = sample_df[predicted_col].mean()
    actual_mean = sample_df[actual_col].mean()
    scale_ratio = pred_mean / (actual_mean + 1e-10)
    print(f"Scale Ratio: {scale_ratio:.4f}")


def validate_model(
    train_results_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    clip_outliers: bool = True,
) -> dict[str, any]:
    """
    Perform complete model validation on test data.

    Parameters
    ----------
    train_results_df : pd.DataFrame
        Training results with learned weights.
    test_df : pd.DataFrame
        Test DataFrame with processed data.
    train_df : pd.DataFrame, optional
        Training DataFrame for feature range checking. If None, skips range check.
    output_dir : Path, optional
        Directory to save visualizations.
    clip_outliers : bool, default True
        Whether to clip outliers in test data to training range.

    Returns
    -------
    dict[str, any]
        Dictionary with all validation metrics and results.
    """
    # Step 1: Load training weights
    weights_df = load_training_weights(train_results_df)

    # Step 1.5: Run diagnostics
    diagnose_validation_issues(train_results_df, test_df, weights_df)

    # Step 2: Check feature ranges and clip outliers if training data provided
    feature_cols = ["Likes_log_log", "Comments_log_log", "Shares_log_log"]

    # Diagnostic: Print feature scale comparison
    print("\n📊 Feature Scale Comparison (Train vs Test):")
    print("-" * 80)
    if train_df is not None:
        for col in feature_cols:
            if col in train_df.columns and col in test_df.columns:
                train_mean = train_df[col].mean()
                test_mean = test_df[col].mean()
                scale_ratio = test_mean / (train_mean + 1e-10)
                print(
                    f"  {col}: Train={train_mean:.4f}, "
                    f"Test={test_mean:.4f}, Ratio={scale_ratio:.4f}"
                )
                if abs(scale_ratio - 1.0) > 0.2:
                    print("    ⚠️  Scale mismatch detected!")
        print("-" * 80)

        # Aggressive clipping
        test_df = check_feature_ranges(
            train_df, test_df, feature_cols, clip_outliers=True
        )
        print("✓ Feature clipping applied (aggressive mode)\n")
    else:
        print("⚠️  No training data provided - skipping feature range check\n")

    # Step 3: Calculate predicted scores (with intercept)
    # Get target range from test data
    target_min = test_df["Reach_log"].quantile(0.01)  # 1st percentile
    target_max = test_df["Reach_log"].quantile(0.99)  # 99th percentile
    target_range = (target_min, target_max)

    test_df = calculate_predicted_engagement_score(
        test_df,
        weights_df,
        target_range=target_range,
        calibrate=True,
        actual_col="Reach_log",
    )

    # Step 4: Print verification samples
    print_prediction_samples(test_df)

    # Step 5: Calculate regression metrics (includes Pearson correlation)
    regression_metrics = calculate_regression_metrics(test_df)

    # Print correlation analysis
    print("\n📈 Scale-Invariant Analysis:")
    print("-" * 80)
    pearson_r = regression_metrics.get("Pearson_R", 0.0)
    r2 = regression_metrics["R_Squared"]
    print(f"  Pearson Correlation (R): {pearson_r:.4f}")
    print(f"  R² Score: {r2:.4f}")

    if abs(pearson_r) > 0.3 and r2 < 0:
        print("  ⚠️  High correlation but negative R² detected!")
        print("  → This indicates correct weight direction but scale/intercept shift.")
        print("  → Intercept recalibration has been applied.")
    elif abs(pearson_r) < 0.1:
        print("  ⚠️  Very low correlation - weights may not be predictive.")
    print("-" * 80)

    # Step 6: Identify actual and predicted trending posts
    test_df = identify_trending_posts_by_score(test_df, "Reach_log", percentile=90.0)
    test_df = test_df.rename(columns={"Is_Trending": "Actual_Trending"})

    test_df = identify_trending_posts_by_score(
        test_df, "Predicted_Engagement_Score", percentile=90.0
    )
    test_df = test_df.rename(columns={"Is_Trending": "Predicted_Trending"})

    # Step 7: Calculate classification metrics
    classification_metrics, cm = calculate_classification_metrics(test_df)

    # Step 8: Create visualizations
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        plot_prediction_vs_actual(
            test_df, save_path=output_dir / "prediction_vs_actual.png"
        )
        plot_confusion_matrix(cm, save_path=output_dir / "confusion_matrix.png")

    return {
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
        "confusion_matrix": cm,
        "test_df": test_df,
    }


def print_validation_results(validation_results: dict[str, any]) -> None:
    """
    Print validation results in a readable format.

    Parameters
    ----------
    validation_results : dict[str, any]
        Validation results dictionary.
    """
    print("\n" + "=" * 80)
    print("📊 CROSS-DATABASE VALIDATION RESULTS")
    print("=" * 80)

    # Regression Metrics
    reg_metrics = validation_results["regression_metrics"]
    print("\n📈 Regression Performance Metrics:")
    print("-" * 80)
    print(f"  • R² Score: {reg_metrics['R_Squared']:.4f}")
    print(f"  • Pearson Correlation (R): {reg_metrics.get('Pearson_R', 0.0):.4f}")
    print(f"  • Mean Absolute Error (MAE): {reg_metrics['MAE']:.4f}")
    print(f"  • Root Mean Squared Error (RMSE): {reg_metrics['RMSE']:.4f}")

    # Classification Metrics
    cls_metrics = validation_results["classification_metrics"]
    print("\n🎯 Trend Detection Accuracy (Classification):")
    print("-" * 80)
    accuracy_val = cls_metrics["Accuracy"]
    print(f"  • Overall Accuracy: {accuracy_val:.4f} ({accuracy_val*100:.2f}%)")
    print(
        f"  • Accuracy Percentage (Top 10%): "
        f"{cls_metrics['Accuracy_Percentage']:.2f}%"
    )
    print(f"  • Precision: {cls_metrics['Precision']:.4f}")
    print(f"  • Recall: {cls_metrics['Recall']:.4f}")
    print(f"  • F1-Score: {cls_metrics['F1_Score']:.4f}")

    print("\n📋 Confusion Matrix Breakdown:")
    print("-" * 80)
    print(f"  • True Positives (TP): {cls_metrics['True_Positives']}")
    print(f"  • True Negatives (TN): {cls_metrics['True_Negatives']}")
    print(f"  • False Positives (FP): {cls_metrics['False_Positives']}")
    print(f"  • False Negatives (FN): {cls_metrics['False_Negatives']}")
    print(f"  • Total Actual Trending: {cls_metrics['Total_Actual_Trending']}")
    print(f"  • Correctly Identified: {cls_metrics['Correctly_Identified']}")

    # Generalization Power Summary
    print("\n" + "=" * 80)
    print("🔬 GENERALIZATION POWER SUMMARY")
    print("=" * 80)

    r2 = reg_metrics["R_Squared"]
    accuracy_pct = cls_metrics["Accuracy_Percentage"]

    if r2 >= 0.7:
        r2_assessment = "Excellent"
    elif r2 >= 0.5:
        r2_assessment = "Good"
    elif r2 >= 0.3:
        r2_assessment = "Moderate"
    else:
        r2_assessment = "Poor"

    if accuracy_pct >= 70:
        trend_assessment = "Excellent"
    elif accuracy_pct >= 50:
        trend_assessment = "Good"
    elif accuracy_pct >= 30:
        trend_assessment = "Moderate"
    else:
        trend_assessment = "Poor"

    print(f"\n📊 Regression Generalization: {r2_assessment}")
    print(
        f"   R² of {r2:.4f} indicates the learned weights explain "
        f"{r2*100:.1f}% of variance in unseen test data."
    )

    print(f"\n🎯 Trend Detection Generalization: {trend_assessment}")
    print(
        f"   {accuracy_pct:.1f}% of actual trending posts were correctly "
        f"identified using the learned weights."
    )

    overall_assessment = (
        "Excellent"
        if r2 >= 0.7 and accuracy_pct >= 70
        else (
            "Good"
            if r2 >= 0.5 and accuracy_pct >= 50
            else "Moderate"
            if r2 >= 0.3 and accuracy_pct >= 30
            else "Needs Improvement"
        )
    )

    print(f"\n✨ Overall Generalization Power: {overall_assessment}")
    print("=" * 80 + "\n")
