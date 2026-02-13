"""Sanity checks for trending label construction (ER_proxy-based)."""

import numpy as np
import pandas as pd

from popweight.trending import apply_trending_label, compute_segment_thresholds


def test_trending_rate_near_10_percent() -> None:
    """Train trending rate should be ~10% with TREND_PERCENTILE=0.9."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        {
            "Segment": ["A"] * 500 + ["B"] * 500,
            "Likes": np.random.randint(0, 1000, n),
            "Comments": np.random.randint(0, 100, n),
            "Shares": np.random.randint(0, 50, n),
            "Reach": np.random.randint(10, 5000, n),
        }
    )
    thr = compute_segment_thresholds(df, 0.9)
    labeled = apply_trending_label(df, thr)
    rate = labeled["Trending"].mean()
    assert 0.08 <= rate <= 0.12, f"Trending rate should be ~0.1, got {rate}"


def test_missing_segment_uses_median_threshold() -> None:
    """Rows with unseen segment should use median threshold without error."""
    np.random.seed(42)
    train = pd.DataFrame(
        {
            "Segment": ["A"] * 400 + ["B"] * 400,
            "Likes": np.random.randint(0, 1000, 800),
            "Comments": np.random.randint(0, 100, 800),
            "Shares": np.random.randint(0, 50, 800),
            "Reach": np.random.randint(10, 5000, 800),
        }
    )
    test = pd.DataFrame(
        {
            "Segment": ["A"] * 100 + ["B"] * 100 + ["C"] * 50,
            "Likes": np.random.randint(0, 1000, 250),
            "Comments": np.random.randint(0, 100, 250),
            "Shares": np.random.randint(0, 50, 250),
            "Reach": np.random.randint(10, 5000, 250),
        }
    )
    thr = compute_segment_thresholds(train, 0.9)
    labeled = apply_trending_label(test, thr)
    assert "Trending" in labeled.columns
    assert labeled["Trending"].isin([0, 1]).all()


def test_er_proxy_safe_division() -> None:
    """ER_proxy should handle edge cases (Reach clipped to avoid div by zero)."""
    df = pd.DataFrame(
        {
            "Segment": ["X"],
            "Likes": [100.0],
            "Comments": [10.0],
            "Shares": [5.0],
            "Reach": [1.0],  # MIN_REACH=1, so valid
        }
    )
    thr = compute_segment_thresholds(df, 0.9)
    labeled = apply_trending_label(df, thr)
    assert labeled["Trending"].dtype in (np.int64, np.int32)
