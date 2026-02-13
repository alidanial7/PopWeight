"""Baseline comparison: equal-weight score vs PopWeight."""

from typing import Any

from popweight.evaluation import (
    classification_metrics,
    evaluate_regression,
)
from popweight.models import predict_trending, train_trending_classifier
from popweight.scoring import apply_baseline_scores
from popweight.splits import Split
from popweight.trending import apply_trending_label, compute_segment_thresholds


def run_baseline_evaluation(
    split: Split,
    trend_percentile: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run baseline pipeline: Score_baseline regression + classifier metrics.

    Computes Score_baseline = Likes_ll + Comments_ll + Shares_ll;
    evaluates regression vs log(ER_proxy); trains classifier with Score_baseline;
    evaluates classification.

    Args:
        split: Train/test split with Segment, Reach, Likes_ll, etc.
        trend_percentile: Percentile for trending threshold (e.g., 0.9).
        seed: Optional seed for output.

    Returns:
        Dict with baseline_regression, baseline_classification, and seed.
    """
    thresholds = compute_segment_thresholds(split.train_df, trend_percentile)
    train_baseline = apply_baseline_scores(split.train_df)
    test_baseline = apply_baseline_scores(split.test_df)
    train_labeled = apply_trending_label(split.train_df, thresholds)
    test_labeled = apply_trending_label(split.test_df, thresholds)

    reg = evaluate_regression(
        test_baseline,
        seed=seed,
        score_col="Score_baseline",
    )
    train_for_clf = train_labeled.assign(Score=train_baseline["Score_baseline"])
    test_for_clf = test_labeled.assign(Score=test_baseline["Score_baseline"])
    model = train_trending_classifier(train_for_clf)
    y_pred, _, _ = predict_trending(
        model,
        test_for_clf,
        train_scored_labeled_df=train_for_clf,
    )
    cls = classification_metrics(
        test_labeled["Trending"].to_numpy(),
        y_pred,
    )
    if seed is not None:
        cls["seed"] = seed
    return {
        "baseline_regression": reg,
        "baseline_classification": cls,
        "seed": seed,
    }


def compare_with_popweight(
    pw_reg: dict[str, Any],
    pw_cls: dict[str, Any],
    bl_reg: dict[str, Any],
    bl_cls: dict[str, Any],
) -> dict[str, Any]:
    """Compute delta = PopWeight - baseline for each metric.

    Args:
        pw_reg: PopWeight regression metrics (r2, mae, rmse, pearson).
        pw_cls: PopWeight classification metrics (accuracy, f1, etc.).
        bl_reg: Baseline regression metrics.
        bl_cls: Baseline classification metrics.

    Returns:
        Dict with delta_r2, delta_mae, delta_rmse, delta_pearson,
        delta_accuracy, delta_precision, delta_recall, delta_f1.
    """
    return {
        "delta_r2": pw_reg["r2"] - bl_reg["r2"],
        "delta_mae": pw_reg["mae"] - bl_reg["mae"],
        "delta_rmse": pw_reg["rmse"] - bl_reg["rmse"],
        "delta_pearson": pw_reg["pearson"] - bl_reg["pearson"],
        "delta_accuracy": pw_cls["accuracy"] - bl_cls["accuracy"],
        "delta_precision": pw_cls["precision"] - bl_cls["precision"],
        "delta_recall": pw_cls["recall"] - bl_cls["recall"],
        "delta_f1": pw_cls["f1"] - bl_cls["f1"],
    }
