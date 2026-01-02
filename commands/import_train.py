"""Command to import training data from Excel into SQLite database."""

import argparse
from pathlib import Path

from utils import load_social_media_data, save_to_sqlite


def import_train_command() -> None:
    """
    Command-line command to import training data from Excel to SQLite database.

    Usage:
        python -m commands.import_train
        python -m commands.import_train --excel data/train.xlsx --db data/train.db
    """
    parser = argparse.ArgumentParser(
        description="Import training data from Excel into SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--excel",
        type=str,
        default=None,
        help=(
            "Path to the training Excel file "
            "(default: data/train.xlsx relative to project root)"
        ),
    )

    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help=(
            "Path to the SQLite database file "
            "(default: data/train.db relative to project root)"
        ),
    )

    parser.add_argument(
        "--table",
        type=str,
        default="train_data",
        help="Name of the table to store data in (default: train_data)",
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

    # Get project root (parent of commands directory)
    # This works whether run as module or script
    commands_dir = Path(__file__).parent
    project_root = commands_dir.parent

    # Set default paths
    excel_path = (
        Path(args.excel) if args.excel else project_root / "data" / "train.xlsx"
    )
    db_path = Path(args.db) if args.db else project_root / "data" / "train.db"

    # Validate Excel file exists
    if not excel_path.exists():
        print(f"Error: Excel file not found at {excel_path}")
        print("Please ensure the file exists or specify a different path with --excel")
        return

    # Load data from Excel
    print(f"Loading training data from Excel file: {excel_path}")
    try:
        df = load_social_media_data(file_path=str(excel_path), sheet_name=args.sheet)
        print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Save to database
    print(f"\nSaving training data to database: {db_path}")
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
    import_train_command()
