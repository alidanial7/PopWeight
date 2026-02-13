"""Excel ingestion for loading the Working File sheet."""

import pandas as pd


def load_working_file(path: str, sheet: str) -> pd.DataFrame:
    """Load dataset from Excel file using the specified sheet.

    Uses openpyxl engine and preserves column names. Trims whitespace from
    headers and string columns.

    Args:
        path: Path to the Excel file.
        sheet: Sheet name to load (e.g., "Working File").

    Returns:
        DataFrame with trimmed column names and string values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df
