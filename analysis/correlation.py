"""Correlation analysis module for engagement metrics.

This module provides functions to calculate and analyze correlations between
engagement metrics (Likes, Comments, Shares) and performance metrics (Reach).
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import read_from_sqlite


def load_data_with_progress(
    db_path: Path, table_name: str, data_source: str
) -> pd.DataFrame:
    """
    Load data from SQLite database with progress indicator.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    table_name : str
        Name of the table to read from.
    data_source : str
        Name of the data source (for display purposes).

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame from the database.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    ValueError
        If the table does not exist or cannot be read.
    """
    import time

    print(f"\n📖 Loading {data_source} data...")
    try:
        with tqdm(
            total=100,
            desc="Reading database",
            bar_format="{l_bar}{bar}| {n_fmt}%",
        ) as pbar:
            # Simulate progress for database reading
            for _ in range(0, 100, 25):
                time.sleep(0.05)
                pbar.update(25)

            df = read_from_sqlite(db_path=str(db_path), table_name=table_name)
            pbar.update(100 - pbar.n)

        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df
    except Exception as e:
        error_msg = (
            f"❌ Error loading data: {e}\n\n"
            f"Please ensure you have run:\n"
            f"  1. python import_{data_source}.py"
        )
        raise ValueError(error_msg) from e


def interpret_correlation_strength(correlation: float) -> tuple[str, str]:
    """
    Interpret correlation coefficient strength and direction.

    Parameters
    ----------
    correlation : float
        Pearson correlation coefficient (range: -1 to 1).

    Returns
    -------
    tuple[str, str]
        Tuple of (strength, direction) where:
        - strength: "negligible", "weak", "moderate", "strong", or "very strong"
        - direction: "positive" or "negative"
    """
    abs_corr = abs(correlation)

    # Determine strength based on absolute value
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    # Determine direction
    direction = "positive" if correlation > 0 else "negative"

    return strength, direction


def calculate_correlation(
    col1: str, col2: str, data_source: str = "train"
) -> dict[str, float | str]:
    """
    Calculate correlation between two columns in the dataset.

    This function loads data from the specified data source, calculates
    Pearson correlation coefficient, and returns comprehensive results.

    Parameters
    ----------
    col1 : str
        First column name (e.g., "Likes", "Comments", "Shares").
    col2 : str
        Second column name (e.g., "Reach").
    data_source : str, default "train"
        Data source to use: "train" or "test". Determines which database
        and table to read from.

    Returns
    -------
    dict[str, float | str]
        Dictionary containing:
        - "correlation": Correlation coefficient (float)
        - "strength": Strength interpretation (str)
        - "direction": Direction interpretation (str)
        - "col1_stats": Dictionary with mean, std, min, max for col1
        - "col2_stats": Dictionary with mean, std, min, max for col2

    Raises
    ------
    ValueError
        If columns don't exist in the data or data cannot be loaded.
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / f"{data_source}.db"
    table_name = f"{data_source}_data_raw"

    # Load data
    df = load_data_with_progress(db_path, table_name, data_source)

    # Validate columns exist
    if col1 not in df.columns:
        available = ", ".join(df.columns[:10])
        raise ValueError(
            f"Column '{col1}' not found in data.\n" f"Available columns: {available}..."
        )

    if col2 not in df.columns:
        available = ", ".join(df.columns[:10])
        raise ValueError(
            f"Column '{col2}' not found in data.\n" f"Available columns: {available}..."
        )

    # Calculate correlation
    correlation = df[col1].corr(df[col2])

    # Interpret correlation
    strength, direction = interpret_correlation_strength(correlation)

    # Calculate statistics
    col1_stats = {
        "mean": float(df[col1].mean()),
        "std": float(df[col1].std()),
        "min": float(df[col1].min()),
        "max": float(df[col1].max()),
    }

    col2_stats = {
        "mean": float(df[col2].mean()),
        "std": float(df[col2].std()),
        "min": float(df[col2].min()),
        "max": float(df[col2].max()),
    }

    return {
        "correlation": float(correlation),
        "strength": strength,
        "direction": direction,
        "col1_stats": col1_stats,
        "col2_stats": col2_stats,
        "col1": col1,
        "col2": col2,
    }


def display_correlation_results(results: dict[str, float | str]) -> None:
    """
    Display correlation analysis results in a formatted output.

    Parameters
    ----------
    results : dict[str, float | str]
        Results dictionary from calculate_correlation function.
    """
    col1 = results["col1"]
    col2 = results["col2"]
    correlation = results["correlation"]
    strength = results["strength"]
    direction = results["direction"]
    col1_stats = results["col1_stats"]
    col2_stats = results["col2_stats"]

    # Display header
    print("\n" + "=" * 80)
    print(f"📊 CORRELATION ANALYSIS: {col1} vs {col2}")
    print("=" * 80)

    # Display correlation results
    print("\n" + "-" * 80)
    print("CORRELATION RESULTS")
    print("-" * 80)
    print(f"Column 1: {col1}")
    print(f"Column 2: {col2}")
    print(f"Correlation Coefficient: {correlation:.4f}")
    print(f"Interpretation: {strength} {direction} correlation")

    # Display statistical summary
    print("\n" + "-" * 80)
    print("STATISTICAL SUMMARY")
    print("-" * 80)
    print(f"{col1}:")
    print(f"  Mean: {col1_stats['mean']:.2f}")
    print(f"  Std:  {col1_stats['std']:.2f}")
    print(f"  Min:  {col1_stats['min']:.2f}")
    print(f"  Max:  {col1_stats['max']:.2f}")
    print(f"\n{col2}:")
    print(f"  Mean: {col2_stats['mean']:.2f}")
    print(f"  Std:  {col2_stats['std']:.2f}")
    print(f"  Min:  {col2_stats['min']:.2f}")
    print(f"  Max:  {col2_stats['max']:.2f}")

    # Display footer
    print("\n" + "=" * 80)
    print("✅ CORRELATION ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


def analyze_correlation(col1: str, col2: str, data_source: str = "train") -> None:
    """
    Complete correlation analysis workflow.

    This is the main entry point for correlation analysis. It calculates
    correlation and displays results in a user-friendly format.

    Parameters
    ----------
    col1 : str
        First column name.
    col2 : str
        Second column name.
    data_source : str, default "train"
        Data source to use: "train" or "test".

    Examples
    --------
    >>> analyze_correlation("Likes", "Reach", data_source="train")
    >>> analyze_correlation("Comments", "Reach", data_source="train")
    >>> analyze_correlation("Shares", "Reach", data_source="train")
    """
    try:
        results = calculate_correlation(col1, col2, data_source)
        display_correlation_results(results)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ {e}")
        print()
