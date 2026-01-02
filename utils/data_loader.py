"""Utility functions for loading data files."""

from pathlib import Path

import pandas as pd


def load_social_media_data(
    file_path: str | None = None, sheet_name: str | None = None
) -> pd.DataFrame:
    """
    Load social media engagement data from Excel file.

    Parameters
    ----------
    file_path : str, optional
        Path to the Excel file. If None, uses the default path:
        'data/social_media_engagement_data.xlsx'
    sheet_name : str, optional
        Name or index of the sheet to read. If None, reads the first sheet.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the social media engagement data.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the file cannot be read or parsed.

    Examples
    --------
    >>> df = load_social_media_data()
    >>> df = load_social_media_data(sheet_name='Sheet1')
    >>> df = load_social_media_data(file_path='custom/path/data.xlsx')
    """
    # Default file path
    if file_path is None:
        # Get the project root directory (parent of utils)
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "social_media_engagement_data.xlsx"
    else:
        file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Excel file not found at: {file_path}\n"
            f"Please ensure the file exists at the specified path."
        )

    try:
        # Read Excel file
        # If sheet_name is None, default to first sheet (index 0)
        # to ensure DataFrame return
        if sheet_name is None:
            sheet_name = 0

        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

        # If multiple sheets were requested and we got a dict, return the first one
        if isinstance(df, dict):
            df = list(df.values())[0]

        return df
    except Exception as e:
        raise ValueError(
            f"Error reading Excel file at {file_path}: {str(e)}\n"
            "Make sure the file is a valid Excel file and that 'openpyxl' "
            "is installed."
        ) from e
