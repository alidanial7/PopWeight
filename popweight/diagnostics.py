"""Pipeline diagnostics and failure detection."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

CAT_COLS = [
    "Platform",
    "Post Type",
    "Weekday Type",
    "Time Periods",
    "Age Group",
    "Sentiment",
]
TRANSFORM_COLS = ["Likes_ll", "Comments_ll", "Shares_ll", "Reach_log", "Segment"]


def _has_trailing_whitespace(series: pd.Series) -> bool:
    """Check if any string value has leading/trailing whitespace."""
    for val in series.dropna():
        if isinstance(val, str) and (val != val.strip()):
            return True
    return False


def run_diagnostics(
    df_features: pd.DataFrame,
    splits: list,
    weights_df: pd.DataFrame,
    min_segment_samples: int = 20,
) -> dict[str, Any]:
    """Detect common pipeline failures.

    Checks: missing segments in train/test, too-small segments,
    NaNs after transforms, unexpected category values (trailing spaces).

    Args:
        df_features: DataFrame after add_transforms and add_segment_key.
        splits: List of Split objects.
        weights_df: Combined weights from all seeds.
        min_segment_samples: Minimum rows per segment threshold.

    Returns:
        Report dict with checks and status (fatal_issues, warnings).
    """
    report: dict[str, Any] = {
        "fatal_issues": [],
        "warnings": [],
        "checks": {},
    }

    # NaNs after transforms
    nan_cols = [c for c in TRANSFORM_COLS if c in df_features.columns]
    if nan_cols:
        nan_counts = df_features[nan_cols].isna().sum()
        if nan_counts.any():
            report["fatal_issues"].append("NaNs in transform columns")
            report["checks"]["nan_after_transforms"] = nan_counts.to_dict()
        else:
            report["checks"]["nan_after_transforms"] = "none"

    # Trailing whitespace in categoricals
    trailing = []
    for col in CAT_COLS:
        if col in df_features.columns and _has_trailing_whitespace(df_features[col]):
            trailing.append(col)
    if trailing:
        report["warnings"].append(f"Trailing whitespace in: {trailing}")
        report["checks"]["trailing_whitespace"] = trailing
    else:
        report["checks"]["trailing_whitespace"] = "none"

    # Missing segments in train or test
    missing_in_train = []
    small_segments = []
    for split in splits:
        train_seg = set(split.train_df["Segment"].unique())
        test_seg = set(split.test_df["Segment"].unique())
        m = test_seg - train_seg
        if m:
            missing_in_train.append({"seed": split.seed, "segments": list(m)})
        seg_counts = split.train_df.groupby("Segment").size()
        small = seg_counts[seg_counts < min_segment_samples]
        if len(small) > 0:
            small_segments.append(
                {
                    "seed": split.seed,
                    "segments": small.to_dict(),
                }
            )

    if missing_in_train:
        report["fatal_issues"].append("Test segments missing from train")
        report["checks"]["missing_segments"] = missing_in_train
    else:
        report["checks"]["missing_segments"] = "none"

    if small_segments:
        report["warnings"].append("Segments below min_segment_samples")
        report["checks"]["small_segments"] = small_segments
    else:
        report["checks"]["small_segments"] = "none"

    # Weights coverage
    weight_segments = set(weights_df["Segment"].unique())
    feature_segments = set(df_features["Segment"].unique())
    missing_weights = feature_segments - weight_segments
    if missing_weights:
        report["warnings"].append(
            f"Missing weights for segments: {list(missing_weights)}"
        )
        report["checks"]["missing_weights"] = list(missing_weights)
    else:
        report["checks"]["missing_weights"] = "none"

    report["ok"] = len(report["fatal_issues"]) == 0
    return report


def save_diagnostics_report(
    report: dict[str, Any],
    output_path: str = "outputs/diagnostics_report.json",
) -> None:
    """Save diagnostics report to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
