"""Data loading utilities for the PopWeight project.

This module provides reusable functions for loading data from SQLite databases
with progress indicators and error handling.
"""

import time
from pathlib import Path

from tqdm import tqdm

from utils import read_from_sqlite


def load_processed_data(
    project_root: Path, data_source: str, data_type: str = "processed"
) -> tuple:
    """
    Load processed data from SQLite database with progress indicator.

    Parameters
    ----------
    project_root : Path
        Root directory of the project.
    data_source : str
        Data source name: "train" or "test".
    data_type : str, default "processed"
        Type of data to load: "processed" or "raw".

    Returns
    -------
    tuple
        Tuple containing (DataFrame, None) on success.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    ValueError
        If the table does not exist or data cannot be loaded.
    """
    db_path = project_root / "data" / f"{data_source}.db"
    table_name = f"{data_source}_data_{data_type}"

    print(f"\n📖 Loading {data_type} {data_source} data...")
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
        return df, None
    except Exception as e:
        error_msg = (
            f"❌ Error loading data: {e}\n\n"
            f"Please ensure you have run:\n"
            f"  1. python import_{data_source}.py\n"
            f"  2. python preprocess_{data_source}.py"
        )
        raise ValueError(error_msg) from e
