"""SQLite storage for intermediate results."""

import sqlite3
from pathlib import Path

import pandas as pd

_db_path: str | None = None

# Table names for reference
TABLES = [
    "raw_working_file",
    "clean_data",
    "features_data",
    "weights",
    "regression_metrics",
    "classification_metrics",
    "aggregate_metrics",
]


def init_db(sqlite_path: str) -> None:
    """Initialize SQLite database and ensure output directory exists.

    Creates the parent directory if needed and sets the database path
    for subsequent write_df/read_df calls.

    Args:
        sqlite_path: Path to the SQLite file (e.g., outputs/results.sqlite).
    """
    global _db_path
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    _db_path = sqlite_path


def _get_path() -> str:
    """Return the configured database path. Raises if not initialized."""
    if _db_path is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_path


def write_df(table_name: str, df: pd.DataFrame) -> None:
    """Write a DataFrame to a SQLite table (replaces if exists).

    Args:
        table_name: Target table name.
        df: DataFrame to persist.
    """
    path = _get_path()
    with sqlite3.connect(path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def read_df(table_name: str, where: str | None = None) -> pd.DataFrame:
    """Read a DataFrame from a SQLite table.

    Args:
        table_name: Source table name.
        where: Optional SQL WHERE clause (e.g., "seed = 0"). Do not
            include the word WHERE.

    Returns:
        DataFrame with table contents.
    """
    path = _get_path()
    query = f'SELECT * FROM "{table_name}"'
    if where:
        query += f" WHERE {where}"
    with sqlite3.connect(path) as conn:
        return pd.read_sql(query, conn)
