"""Column normalization and schema validation."""

import re

import pandas as pd

REQUIRED_COLUMNS = [
    "Platform",
    "Post Type",
    "Likes",
    "Comments",
    "Shares",
    "Reach",
    "Weekday Type",
    "Time Periods",
    "Age Group",
    "Sentiment",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip whitespace and fix common variants.

    Strips header whitespace and collapses multiple consecutive spaces
    to a single space.

    Args:
        df: Input DataFrame with potentially messy column names.

    Returns:
        DataFrame with normalized column names (copy).
    """
    out = df.copy()
    new_names = []
    for col in out.columns:
        name = str(col).strip()
        name = re.sub(r"\s+", " ", name)
        new_names.append(name)
    out.columns = new_names
    return out


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure all required columns exist in the DataFrame.

    Raises:
        ValueError: If any required columns are missing, with a message
            listing all missing columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {', '.join(missing)}"
        raise ValueError(msg)
