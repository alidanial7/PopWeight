"""Script to preprocess training data from SQLite database."""

import sys
import time
from pathlib import Path

from tqdm import tqdm

from utils import (
    filter_essential_columns,
    list_tables,
    preprocess_train_data,
    read_from_sqlite,
    save_to_sqlite,
)


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
    """Preprocess training data from SQLite database."""
    print_header("🔧 Training Data Preprocessing")

    # Get project root directory
    project_root = Path(__file__).parent

    # Define paths
    db_path = project_root / "data" / "train.db"
    table_name = "train_data_raw"
    output_table = "train_data_processed"

    # Check if database exists
    if not db_path.exists():
        print_error(f"Database file not found at {db_path}")
        print_info("Please run 'python import_train.py' first to import data")
        return

    # Check if the required table exists
    try:
        available_tables = list_tables(db_path=str(db_path))
        if table_name not in available_tables:
            print_error(f"Table '{table_name}' not found in database")
            print_info(f"Available tables in database: {', '.join(available_tables)}")
            print_info(
                f"\nPlease run 'python import_train.py' first to create "
                f"the '{table_name}' table"
            )
            return
    except Exception as e:
        print_error(f"Error checking database tables: {e}")
        return

    print_info(f"Source: {db_path.name}")
    print_info(f"Input Table: {table_name} (raw data)")
    print_info(f"Output Table: {output_table} (processed data)\n")

    # Load data from database with progress
    print("📖 Loading data from database...")
    try:
        with tqdm(
            total=100, desc="Reading DB", bar_format="{l_bar}{bar}| {n_fmt}%"
        ) as pbar:
            # Simulate progress for database reading
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = read_from_sqlite(db_path=str(db_path), table_name=table_name)
            pbar.update(100 - pbar.n)

        print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        columns_preview = ", ".join(df.columns[:5].tolist())
        columns_suffix = "..." if len(df.columns) > 5 else ""
        print(f"   Columns: {columns_preview}{columns_suffix}\n")
    except Exception as e:
        print_error(f"Error loading from database: {e}")
        return

    # Preprocess data with detailed step-by-step progress
    print("⚙️  Preprocessing data...")
    print("   Steps to perform:")
    print("   🔍 Handling missing values")
    print("   📊 Log-log normalization (Likes, Comments, Shares)")
    print("   ⏰ Extracting temporal features (Post Timestamp)")
    print("   🔢 One-hot encoding (Platform, Post Type, Sentiment)")
    print("   📏 Scaling Audience Age (StandardScaler)")
    print("   🎯 Preparing target variable (Reach log transform)")
    print()
    try:
        total_rows = len(df)

        # Show preprocessing progress
        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} " "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        with tqdm(
            total=total_rows,
            desc="⚙️  Preprocessing",
            unit="rows",
            bar_format=bar_format,
            ncols=80,
        ) as pbar:
            # Perform preprocessing
            df_processed = preprocess_train_data(df)
            # Update progress bar to completion
            pbar.update(total_rows)

        print()
        print_success(
            f"Preprocessed {len(df_processed):,} rows × "
            f"{len(df_processed.columns)} columns"
        )
        new_cols = len(df_processed.columns) - len(df.columns)
        print(f"   ✨ New columns added: {new_cols}")
        print(f"   📈 Original columns: {len(df.columns)}")
        print(f"   📊 Final columns: {len(df_processed.columns)}")
        print()
    except Exception as e:
        print_error(f"Error preprocessing data: {e}")
        return

    # Filter to only essential columns
    print("🔍 Filtering to essential columns...")
    try:
        df_filtered = filter_essential_columns(df_processed)
        print_success(
            f"Filtered to {len(df_filtered.columns)} essential columns "
            f"(removed {len(df_processed.columns) - len(df_filtered.columns)} "
            f"unused columns)"
        )
        print(f"   Essential columns: {', '.join(df_filtered.columns[:10].tolist())}")
        if len(df_filtered.columns) > 10:
            print(f"   ... and {len(df_filtered.columns) - 10} more")
        print()
    except Exception as e:
        print_error(f"Error filtering columns: {e}")
        print_info("Saving all columns instead...")
        df_filtered = df_processed

    # Save to database with progress
    print("💾 Saving preprocessed data to database...")
    try:
        total_rows = len(df_filtered)

        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} " "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        with tqdm(
            total=total_rows,
            desc="Writing to DB",
            unit="rows",
            bar_format=bar_format,
        ) as pbar:
            # Save to database
            save_to_sqlite(
                df=df_filtered,
                db_path=str(db_path),
                table_name=output_table,
                if_exists="replace",
            )
            # Update progress bar to completion
            pbar.update(total_rows)

        print()
        print_success(f"Saved {len(df_filtered):,} rows to database")
        print_success(f"Database location: {db_path.absolute()}")
        print_success(f"Table name: {output_table}")
        print(f"\n{'=' * 70}")
        print("✨ Preprocessing completed successfully!")
        print(f"{'=' * 70}\n")
    except Exception as e:
        print_error(f"Error saving to database: {e}")
        return


if __name__ == "__main__":
    main()
