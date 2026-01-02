"""Utilities for saving and loading trained model results."""

from pathlib import Path

import pandas as pd

from utils import read_from_sqlite, save_to_sqlite


def save_training_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "training_results",
) -> Path:
    """
    Save training results (weights) to file for later use.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with learned weights.
    output_dir : Path
        Directory to save results.
    filename : str, default "training_results"
        Base filename (without extension).

    Returns
    -------
    Path
        Path to saved file.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save to CSV
    csv_path = output_dir / f"{filename}.csv"
    results_df.to_csv(csv_path, index=False)

    # Save to SQLite for easy loading
    db_path = output_dir / f"{filename}.db"
    save_to_sqlite(
        results_df,
        db_path=str(db_path),
        table_name="training_results",
        if_exists="replace",
    )

    return csv_path


def load_training_results(
    results_path: Path,
    from_db: bool = True,
) -> pd.DataFrame:
    """
    Load training results from file.

    Parameters
    ----------
    results_path : Path
        Path to results file (CSV or DB).
    from_db : bool, default True
        If True, loads from SQLite DB. If False, loads from CSV.

    Returns
    -------
    pd.DataFrame
        Training results DataFrame.
    """
    if from_db:
        if results_path.suffix == ".csv":
            # Convert CSV path to DB path
            db_path = results_path.with_suffix(".db")
        else:
            db_path = results_path

        if not db_path.exists():
            raise FileNotFoundError(
                f"Training results database not found at: {db_path}\n"
                "Please run training mode first."
            )

        return read_from_sqlite(db_path=str(db_path), table_name="training_results")
    else:
        if not results_path.exists():
            raise FileNotFoundError(
                f"Training results file not found at: {results_path}\n"
                "Please run training mode first."
            )

        return pd.read_csv(results_path)
