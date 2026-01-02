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
        DataFrame with Platform, Post Type, and weights.
    """
    return results_df[
        ["Platform", "Post Type", "Alpha_Likes", "Beta_Comments", "Gamma_Shares"]
    ].copy()


def calculate_predicted_engagement_score(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    platform_col: str = "Platform",
    post_type_col: str = "Post Type",
) -> pd.DataFrame:
    """
    Calculate predicted engagement score for test data using training weights.

    Formula: Predicted_Score = (alpha * Likes_log_log) +
                                 (beta * Comments_log_log) +
                                 (gamma * Shares_log_log)

    Parameters
    ----------
    df : pd.DataFrame
        Test DataFrame with engagement features.
    weights_df : pd.DataFrame
        DataFrame with learned weights from training.
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
            # Use equal weights if not found
            alpha, beta, gamma = 0.33, 0.33, 0.34
        else:
            alpha = weight_row.iloc[0]["Alpha_Likes"]
            beta = weight_row.iloc[0]["Beta_Comments"]
            gamma = weight_row.iloc[0]["Gamma_Shares"]

        # Calculate predicted score
        predicted_score = (
            alpha * row["Likes_log_log"]
            + beta * row["Comments_log_log"]
            + gamma * row["Shares_log_log"]
        )

        df.loc[idx, "Predicted_Engagement_Score"] = predicted_score

    return df


def calculate_regression_metrics(
    df: pd.DataFrame,
    predicted_col: str = "Predicted_Engagement_Score",
    actual_col: str = "Reach_log",
) -> dict[str, float]:
    """
    Calculate regression performance metrics.

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
        Dictionary with R², MAE, and RMSE metrics.
    """
    y_true = df[actual_col].values
    y_pred = df[predicted_col].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "R_Squared": r2,
        "MAE": mae,
        "RMSE": rmse,
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


def validate_model(
    train_results_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, any]:
    """
    Perform complete model validation on test data.

    Parameters
    ----------
    train_results_df : pd.DataFrame
        Training results with learned weights.
    test_df : pd.DataFrame
        Test DataFrame with processed data.
    output_dir : Path, optional
        Directory to save visualizations.

    Returns
    -------
    dict[str, any]
        Dictionary with all validation metrics and results.
    """
    # Step 1: Load training weights
    weights_df = load_training_weights(train_results_df)

    # Step 2: Calculate predicted scores
    test_df = calculate_predicted_engagement_score(test_df, weights_df)

    # Step 3: Calculate regression metrics
    regression_metrics = calculate_regression_metrics(test_df)

    # Step 4: Identify actual and predicted trending posts
    test_df = identify_trending_posts_by_score(test_df, "Reach_log", percentile=90.0)
    test_df = test_df.rename(columns={"Is_Trending": "Actual_Trending"})

    test_df = identify_trending_posts_by_score(
        test_df, "Predicted_Engagement_Score", percentile=90.0
    )
    test_df = test_df.rename(columns={"Is_Trending": "Predicted_Trending"})

    # Step 5: Calculate classification metrics
    classification_metrics, cm = calculate_classification_metrics(test_df)

    # Step 6: Create visualizations
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
        else "Good"
        if r2 >= 0.5 and accuracy_pct >= 50
        else "Moderate"
        if r2 >= 0.3 and accuracy_pct >= 30
        else "Needs Improvement"
    )

    print(f"\n✨ Overall Generalization Power: {overall_assessment}")
    print("=" * 80 + "\n")
