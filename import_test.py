"""Script to import test data from Excel to SQLite database."""

import sys
import time
from pathlib import Path

from tqdm import tqdm

from utils import load_social_media_data, save_to_sqlite


def print_header(title: str) -> None:
    """Print a beautiful header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}", file=sys.stderr)


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def main():
    """Import test data from Excel to SQLite database."""
    print_header("🧪 Test Data Import")

    # Get project root directory
    project_root = Path(__file__).parent

    # Define paths
    excel_path = project_root / "data" / "test.xlsx"
    db_path = project_root / "data" / "test.db"
    table_name = "test_data_raw"

    # Check if Excel file exists
    if not excel_path.exists():
        print_error(f"Excel file not found at {excel_path}")
        return

    print_info(f"Source: {excel_path.name}")
    print_info(f"Destination: {db_path.name}")
    print_info(f"Table: {table_name}\n")

    # Load data from Excel with progress
    print("📖 Loading data from Excel...")
    try:
        with tqdm(
            total=100, desc="Reading Excel", bar_format="{l_bar}{bar}| {n_fmt}%"
        ) as pbar:
            # Simulate progress for Excel reading
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = load_social_media_data(file_path=str(excel_path))
            pbar.update(100 - pbar.n)

        print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        columns_preview = ", ".join(df.columns[:5].tolist())
        columns_suffix = "..." if len(df.columns) > 5 else ""
        print(f"   Columns: {columns_preview}{columns_suffix}\n")
    except Exception as e:
        print_error(f"Error loading Excel file: {e}")
        return

    # Save to database with progress
    print("💾 Saving to database...")
    try:
        total_rows = len(df)

        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} " "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        with tqdm(
            total=total_rows,
            desc="Writing to DB",
            unit="rows",
            bar_format=bar_format,
        ) as pbar:
            # Save to database (pandas doesn't support progress, so we simulate)
            save_to_sqlite(
                df=df,
                db_path=str(db_path),
                table_name=table_name,
                if_exists="replace",
            )
            # Update progress bar to completion
            pbar.update(total_rows)

        print_success(f"Saved {len(df):,} rows to database")
        print_success(f"Database location: {db_path.absolute()}")
        print(f"\n{'=' * 70}")
        print("✨ Import completed successfully!")
        print(f"{'=' * 70}\n")
    except Exception as e:
        print_error(f"Error saving to database: {e}")
        return


if __name__ == "__main__":
    main()
