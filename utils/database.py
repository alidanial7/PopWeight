"""Utility functions for SQLite database operations."""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect


def save_to_sqlite(
    df: pd.DataFrame,
    db_path: str | Path,
    table_name: str = "social_media_data",
    if_exists: str = "replace",
) -> None:
    """
    Save a DataFrame to a SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save to the database.
    db_path : str | Path
        Path to the SQLite database file. If the file doesn't exist,
        it will be created.
    table_name : str, default "social_media_data"
        Name of the table to store the data in.
    if_exists : str, default "replace"
        How to behave if the table already exists:
        - "fail": Raise a ValueError
        - "replace": Drop the table before inserting new values
        - "append": Insert new values to the existing table

    Raises
    ------
    ValueError
        If if_exists is set to "fail" and the table already exists.
    Exception
        If there's an error writing to the database.

    Examples
    --------
    >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    >>> save_to_sqlite(df, "data.db", table_name="my_table")
    >>> save_to_sqlite(
    ...     df, "data.db", table_name="my_table", if_exists="append"
    ... )
    """
    db_path = Path(db_path)
    # Create parent directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
        )
        engine.dispose()
    except Exception as e:
        raise Exception(
            f"Error saving DataFrame to SQLite database at {db_path}: {str(e)}"
        ) from e


def read_from_sqlite(
    db_path: str | Path,
    table_name: str = "social_media_data",
    query: str | None = None,
) -> pd.DataFrame:
    """
    Read data from a SQLite database into a DataFrame.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.
    table_name : str, default "social_media_data"
        Name of the table to read from. Ignored if query is provided.
    query : str, optional
        SQL query to execute. If provided, table_name is ignored.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the database.

    Raises
    ------
    FileNotFoundError
        If the specified database file does not exist.
    ValueError
        If the table doesn't exist or the query fails.

    Examples
    --------
    >>> df = read_from_sqlite("data.db", table_name="social_media_data")
    >>> query = "SELECT * FROM social_media_data WHERE Likes > 100"
    >>> df = read_from_sqlite("data.db", query=query)
    """
    db_path = Path(db_path)

    # Check if database exists
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at: {db_path}\n"
            f"Please ensure the database file exists at the specified path."
        )

    try:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        if query:
            df = pd.read_sql(query, con=engine)
        else:
            df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
        engine.dispose()

        return df
    except Exception as e:
        raise ValueError(
            f"Error reading from SQLite database at {db_path}: {str(e)}\n"
            "Make sure the database file is valid and the table/query exists."
        ) from e


def list_tables(db_path: str | Path) -> list[str]:
    """
    List all tables in a SQLite database.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.

    Returns
    -------
    list[str]
        List of table names in the database.

    Raises
    ------
    FileNotFoundError
        If the specified database file does not exist.

    Examples
    --------
    >>> tables = list_tables("data.db")
    >>> print(tables)
    ['social_media_data', 'other_table']
    """
    db_path = Path(db_path)

    # Check if database exists
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at: {db_path}\n"
            f"Please ensure the database file exists at the specified path."
        )

    try:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        engine.dispose()
        return tables
    except Exception as e:
        raise ValueError(
            f"Error listing tables from SQLite database at {db_path}: {str(e)}"
        ) from e
