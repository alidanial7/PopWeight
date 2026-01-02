"""Analysis module for engagement weight analysis."""

from .insights import (
    generate_statistical_insights,
    print_statistical_insights,
)
from .models import (
    analyze_engagement_weights,
    extract_weights_table,
)
from .trend_detection import (
    calculate_engagement_zscore,
    get_top_trending_posts,
    identify_trending_posts,
    print_trending_analysis,
    train_trending_classifier,
)
from .validation import (
    calculate_classification_metrics,
    calculate_predicted_engagement_score,
    calculate_regression_metrics,
    check_feature_ranges,
    diagnose_validation_issues,
    plot_confusion_matrix,
    plot_prediction_vs_actual,
    print_prediction_samples,
    print_validation_results,
    validate_model,
)
from .visualizations import (
    create_facet_grid_chart,
    create_heatmap,
)

__all__ = [
    "analyze_engagement_weights",
    "extract_weights_table",
    "create_heatmap",
    "create_facet_grid_chart",
    "generate_statistical_insights",
    "print_statistical_insights",
    "identify_trending_posts",
    "train_trending_classifier",
    "calculate_engagement_zscore",
    "get_top_trending_posts",
    "print_trending_analysis",
    "validate_model",
    "calculate_predicted_engagement_score",
    "calculate_regression_metrics",
    "calculate_classification_metrics",
    "check_feature_ranges",
    "diagnose_validation_issues",
    "print_prediction_samples",
    "plot_prediction_vs_actual",
    "plot_confusion_matrix",
    "print_validation_results",
]
