"""Command to preprocess test data from SQLite database."""

import argparse
from pathlib import Path

from utils import (
    preprocess_test_data,
    read_from_sqlite,
    save_to_sqlite,
)


def preprocess_test_command() -> None:
    """
    Command-line command to preprocess test data from SQLite database.

    Usage:
        python -m commands.preprocess_test
        python -m commands.preprocess_test --db data/test.db --table test_data
    """
    parser = argparse.ArgumentParser(
        description="Preprocess test data from SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help=(
            "Path to the SQLite database file "
            "(default: data/test.db relative to project root)"
        ),
    )

    parser.add_argument(
        "--table",
        type=str,
        default="test_data",
        help="Name of the table to read from (default: test_data)",
    )

    parser.add_argument(
        "--output-table",
        type=str,
        default=None,
        help=(
            "Name of the table to save preprocessed data to "
            "(default: same as --table, replaces existing)"
        ),
    )

    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["fail", "replace", "append"],
        default="replace",
        help=(
            "How to behave if the output table already exists: "
            "fail (raise error), replace (drop and recreate), "
            "or append (add new rows) (default: replace)"
        ),
    )

    args = parser.parse_args()

    # Get project root (parent of commands directory)
    commands_dir = Path(__file__).parent
    project_root = commands_dir.parent

    # Set default paths
    db_path = Path(args.db) if args.db else project_root / "data" / "test.db"
    output_table = args.output_table if args.output_table else args.table

    # Validate database exists
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        print(
            "Please ensure the database exists or specify a different path " "with --db"
        )
        return

    # Load data from database
    print(f"Loading test data from database: {db_path}")
    print(f"  Table: {args.table}")
    try:
        df = read_from_sqlite(db_path=str(db_path), table_name=args.table)
        print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading from database: {e}")
        return

    # Preprocess data
    print("\nPreprocessing test data...")
    print("  - Handling missing values")
    print("  - Applying log-log normalization to Likes, Comments, Shares")
    print("  - Extracting temporal features from Post Timestamp")
    print("  - One-hot encoding Platform, Post Type, Sentiment")
    print("  - Scaling Audience Age with StandardScaler")
    print("  - Preparing target variable (Reach) with log transformation")
    try:
        df_processed = preprocess_test_data(df)
        print(
            f"✓ Preprocessed data: {len(df_processed)} rows × "
            f"{len(df_processed.columns)} columns"
        )
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return

    # Save to database
    print(f"\nSaving preprocessed test data to database: {db_path}")
    print(f"  Table: {output_table}")
    print(f"  Mode: {args.if_exists}")

    try:
        save_to_sqlite(
            df=df_processed,
            db_path=str(db_path),
            table_name=output_table,
            if_exists=args.if_exists,
        )
        print(f"✓ Successfully saved {len(df_processed)} rows to database")
        print(f"✓ Database location: {db_path.absolute()}")
    except Exception as e:
        print(f"Error saving to database: {e}")
        return


if __name__ == "__main__":
    preprocess_test_command()
