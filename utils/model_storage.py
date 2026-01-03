"""Utilities for saving and loading trained model results."""

import pickle
from pathlib import Path

import pandas as pd

from utils import read_from_sqlite, save_to_sqlite


def save_training_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "training_results",
    models_dict: dict | None = None,
) -> Path:
    """
    Save training results (weights) and models to file for later use.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame with learned weights.
    output_dir : Path
        Directory to save results.
    filename : str, default "training_results"
        Base filename (without extension).
    models_dict : dict, optional
        Dictionary of trained models (for Random Forest).

    Returns
    -------
    Path
        Path to saved CSV file.
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

    # Save models if provided (for Random Forest)
    if models_dict is not None:
        models_path = output_dir / f"{filename}_models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models_dict, f)

    return csv_path


def load_training_results(
    results_path: Path,
    from_db: bool = True,
    load_models: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict | None]:
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
    pd.DataFrame or tuple[pd.DataFrame, dict | None]
        Training results DataFrame, and optionally models dictionary.
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

        results_df = read_from_sqlite(
            db_path=str(db_path), table_name="training_results"
        )
    else:
        if not results_path.exists():
            raise FileNotFoundError(
                f"Training results file not found at: {results_path}\n"
                "Please run training mode first."
            )

        results_df = pd.read_csv(results_path)

    # Load models if requested
    models_dict = None
    if load_models:
        if results_path.suffix == ".csv":
            models_path = results_path.parent / f"{results_path.stem}_models.pkl"
        else:
            models_path = results_path.parent / "training_results_models.pkl"

        if models_path.exists():
            with open(models_path, "rb") as f:
                models_dict = pickle.load(f)

    if load_models:
        return results_df, models_dict
    else:
        return results_df
