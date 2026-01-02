"""Command to import Excel data into SQLite database."""

import argparse
from pathlib import Path

from utils import load_social_media_data, save_to_sqlite


def import_excel_command() -> None:
    """
    Command-line command to read data from Excel and save to SQLite database.

    Usage:
        python -m commands.import_excel --excel data/train.xlsx --db data/data.db
        python -m commands.import_excel --excel data/train.xlsx \\
            --db data/data.db --table my_table
        python -m commands.import_excel --excel data/train.xlsx \\
            --db data/data.db --sheet Sheet1
    """
    parser = argparse.ArgumentParser(
        description="Import Excel data into SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--excel",
        type=str,
        required=True,
        help="Path to the Excel file to import",
    )

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to the SQLite database file (will be created if it doesn't exist)",
    )

    parser.add_argument(
        "--table",
        type=str,
        default="social_media_data",
        help="Name of the table to store data in (default: social_media_data)",
    )

    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Name or index of the Excel sheet to read (default: first sheet)",
    )

    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["fail", "replace", "append"],
        default="replace",
        help=(
            "How to behave if the table already exists: "
            "fail (raise error), replace (drop and recreate), "
            "or append (add new rows) (default: replace)"
        ),
    )

    args = parser.parse_args()

    # Validate Excel file exists
    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Error: Excel file not found at {excel_path}")
        return

    # Load data from Excel
    print(f"Loading data from Excel file: {excel_path}")
    try:
        df = load_social_media_data(file_path=str(excel_path), sheet_name=args.sheet)
        print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Save to database
    db_path = Path(args.db)
    print(f"\nSaving data to database: {db_path}")
    print(f"  Table: {args.table}")
    print(f"  Mode: {args.if_exists}")

    try:
        save_to_sqlite(
            df=df,
            db_path=str(db_path),
            table_name=args.table,
            if_exists=args.if_exists,
        )
        print(f"✓ Successfully saved {len(df)} rows to database")
        print(f"✓ Database location: {db_path.absolute()}")
    except Exception as e:
        print(f"Error saving to database: {e}")
        return


if __name__ == "__main__":
    import_excel_command()
