"""Repeated train/test split with segment coverage."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

MAX_SPLIT_TRIES = 20


@dataclass
class Split:
    """A single train/test split with segment coverage."""

    seed: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def make_splits(
    df: pd.DataFrame,
    seeds: list[int],
    train_ratio: float,
) -> list[Split]:
    """Create repeated train/test splits with segment coverage.

    For each seed, splits the data and ensures every segment in test
    also appears in train. Re-samples up to MAX_SPLIT_TRIES if needed;
    otherwise drops test rows with uncovered segments.

    Args:
        df: DataFrame with Segment column (from add_segment_key).
        seeds: Random seeds for reproducibility.
        train_ratio: Fraction of data for training (e.g., 0.8).

    Returns:
        List of Split objects, one per seed.
    """
    if "Segment" not in df.columns:
        raise ValueError("DataFrame must have 'Segment' column")
    splits: list[Split] = []
    for seed in seeds:
        train_df, test_df = _split_with_coverage(df, train_ratio, seed)
        splits.append(Split(seed=seed, train_df=train_df, test_df=test_df))
    return splits


def _split_with_coverage(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ensuring every test segment exists in train."""
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for try_num in range(MAX_SPLIT_TRIES):
        rng = seed + try_num
        tr, te = train_test_split(
            df,
            train_size=train_ratio,
            random_state=rng,
            stratify=df["Segment"],
        )
        train_segments = set(tr["Segment"].unique())
        test_segments = set(te["Segment"].unique())
        missing = test_segments - train_segments
        if not missing:
            return tr.reset_index(drop=True), te.reset_index(drop=True)
        train_df, test_df = tr, te
    test_df = test_df[test_df["Segment"].isin(train_df["Segment"].unique())]
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
