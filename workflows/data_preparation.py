"""Data preparation workflows.

This module contains workflows for data generation, splitting, importing,
and preprocessing operations.
"""

import sys
import time
from pathlib import Path

from tqdm import tqdm

from utils import (
    filter_essential_columns,
    list_tables,
    load_social_media_data,
    preprocess_test_data,
    preprocess_train_data,
    read_from_sqlite,
    save_to_sqlite,
)


def _print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def _print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def _print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}", file=sys.stderr)


def _print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def split_data() -> None:
    """Split base data into train and test datasets."""
    _print_header("📊 Data Splitting Tool")

    project_root = Path(__file__).parent.parent

    base_file = project_root / "data" / "social_media_engagement_data.xlsx"
    train_output = project_root / "data" / "train.xlsx"
    test_output = project_root / "data" / "test.xlsx"

    if not base_file.exists():
        _print_error(f"Base data file not found at {base_file}")
        return

    _print_info(f"Source file: {base_file.name}")
    _print_info(f"Train output: {train_output.name}")
    _print_info(f"Test output: {test_output.name}\n")

    # Load base data
    print("📖 Loading base data...")
    try:
        bar_format = "{l_bar}{bar}| {n_fmt}%"
        with tqdm(total=100, desc="Reading Excel", bar_format=bar_format) as pbar:
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = load_social_media_data(file_path=str(base_file))
            pbar.update(100 - pbar.n)

        _print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        _print_error(f"Error loading Excel file: {e}")
        return

    # Get train percentage
    while True:
        try:
            percentage = input(
                "\nEnter train data percentage (0-100, e.g., 70): "
            ).strip()
            percentage_float = float(percentage)
            if 0 < percentage_float < 100:
                break
            _print_error("Please enter a value between 0 and 100")
        except ValueError:
            _print_error("Please enter a valid number")

    train_percentage = percentage_float
    test_percentage = 100 - train_percentage

    # Confirm
    print("\n" + "=" * 70)
    print("⚠️  WARNING: This will overwrite existing files!")
    print("=" * 70)
    confirm = input("\nDo you want to proceed? (yes/no): ").strip().lower()

    if confirm not in ["yes", "y"]:
        print("\n👋 Operation cancelled.")
        return

    # Split data
    print("\n🔄 Splitting data...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_rows = len(df_shuffled)
    train_size = int(total_rows * (train_percentage / 100))
    train_df = df_shuffled.iloc[:train_size].copy()
    test_df = df_shuffled.iloc[train_size:].copy()

    # Save files
    train_df.to_excel(train_output, index=False, engine="openpyxl")
    test_df.to_excel(test_output, index=False, engine="openpyxl")

    train_msg = f"Train set: {len(train_df):,} rows ({train_percentage:.1f}%)"
    _print_success(train_msg)
    test_msg = f"Test set: {len(test_df):,} rows ({test_percentage:.1f}%)"
    _print_success(test_msg)
    print(f"\n{'=' * 70}\n")


def import_train_data() -> None:
    """Import training data from Excel to SQLite database."""
    _print_header("📊 Training Data Import")

    project_root = Path(__file__).parent.parent
    excel_path = project_root / "data" / "train.xlsx"
    db_path = project_root / "data" / "train.db"
    table_name = "train_data_raw"

    if not excel_path.exists():
        _print_error(f"Excel file not found at {excel_path}")
        return

    _print_info(f"Source: {excel_path.name}")
    _print_info(f"Destination: {db_path.name}")
    _print_info(f"Table: {table_name}\n")

    # Load from Excel
    print("📖 Loading data from Excel...")
    try:
        bar_format = "{l_bar}{bar}| {n_fmt}%"
        with tqdm(total=100, desc="Reading Excel", bar_format=bar_format) as pbar:
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = load_social_media_data(file_path=str(excel_path))
            pbar.update(100 - pbar.n)

        _print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        _print_error(f"Error loading Excel file: {e}")
        return

    # Save to database
    print("💾 Saving to database...")
    try:
        save_to_sqlite(df=df, db_path=str(db_path), table_name=table_name)
        _print_success(f"Saved {len(df):,} rows to database")
        print(f"\n{'=' * 70}\n")
    except Exception as e:
        _print_error(f"Error saving to database: {e}")


def import_test_data() -> None:
    """Import test data from Excel to SQLite database."""
    _print_header("🧪 Test Data Import")

    project_root = Path(__file__).parent.parent
    excel_path = project_root / "data" / "test.xlsx"
    db_path = project_root / "data" / "test.db"
    table_name = "test_data_raw"

    if not excel_path.exists():
        _print_error(f"Excel file not found at {excel_path}")
        return

    _print_info(f"Source: {excel_path.name}")
    _print_info(f"Destination: {db_path.name}")
    _print_info(f"Table: {table_name}\n")

    # Load from Excel
    print("📖 Loading data from Excel...")
    try:
        bar_format = "{l_bar}{bar}| {n_fmt}%"
        with tqdm(total=100, desc="Reading Excel", bar_format=bar_format) as pbar:
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = load_social_media_data(file_path=str(excel_path))
            pbar.update(100 - pbar.n)

        _print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        _print_error(f"Error loading Excel file: {e}")
        return

    # Save to database
    print("💾 Saving to database...")
    try:
        save_to_sqlite(df=df, db_path=str(db_path), table_name=table_name)
        _print_success(f"Saved {len(df):,} rows to database")
        print(f"\n{'=' * 70}\n")
    except Exception as e:
        _print_error(f"Error saving to database: {e}")


def preprocess_train() -> None:
    """Preprocess training data from SQLite database."""
    _print_header("🔧 Training Data Preprocessing")

    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "train.db"
    table_name = "train_data_raw"
    output_table = "train_data_processed"

    if not db_path.exists():
        _print_error(f"Database file not found at {db_path}")
        return

    try:
        available_tables = list_tables(db_path=str(db_path))
        if table_name not in available_tables:
            _print_error(f"Table '{table_name}' not found in database")
            return
    except Exception as e:
        _print_error(f"Error checking database: {e}")
        return

    # Load data
    print("📖 Loading data from database...")
    try:
        with tqdm(
            total=100, desc="Reading DB", bar_format="{l_bar}{bar}| {n_fmt}%"
        ) as pbar:
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = read_from_sqlite(db_path=str(db_path), table_name=table_name)
            pbar.update(100 - pbar.n)

        _print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        _print_error(f"Error loading from database: {e}")
        return

    # Preprocess
    print("⚙️  Preprocessing data...")
    try:
        df_processed = preprocess_train_data(df)
        _print_success(
            f"Preprocessed {len(df_processed):,} rows × "
            f"{len(df_processed.columns)} columns"
        )
    except Exception as e:
        _print_error(f"Error preprocessing data: {e}")
        return

    # Filter columns
    print("🔍 Filtering to essential columns...")
    try:
        df_filtered = filter_essential_columns(df_processed)
        col_count = len(df_filtered.columns)
        _print_success(f"Filtered to {col_count} essential columns")
    except Exception as e:
        _print_error(f"Error filtering columns: {e}")
        df_filtered = df_processed

    # Save
    print("💾 Saving preprocessed data...")
    try:
        save_to_sqlite(df=df_filtered, db_path=str(db_path), table_name=output_table)
        _print_success(f"Saved {len(df_filtered):,} rows to database")
        print(f"\n{'=' * 70}\n")
    except Exception as e:
        _print_error(f"Error saving to database: {e}")


def preprocess_test() -> None:
    """Preprocess test data from SQLite database."""
    _print_header("🔧 Test Data Preprocessing")

    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "test.db"
    table_name = "test_data_raw"
    output_table = "test_data_processed"

    if not db_path.exists():
        _print_error(f"Database file not found at {db_path}")
        return

    try:
        available_tables = list_tables(db_path=str(db_path))
        if table_name not in available_tables:
            _print_error(f"Table '{table_name}' not found in database")
            return
    except Exception as e:
        _print_error(f"Error checking database: {e}")
        return

    # Load data
    print("📖 Loading data from database...")
    try:
        with tqdm(
            total=100, desc="Reading DB", bar_format="{l_bar}{bar}| {n_fmt}%"
        ) as pbar:
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = read_from_sqlite(db_path=str(db_path), table_name=table_name)
            pbar.update(100 - pbar.n)

        _print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        _print_error(f"Error loading from database: {e}")
        return

    # Preprocess
    print("⚙️  Preprocessing data...")
    try:
        df_processed = preprocess_test_data(df)
        _print_success(
            f"Preprocessed {len(df_processed):,} rows × "
            f"{len(df_processed.columns)} columns"
        )
    except Exception as e:
        _print_error(f"Error preprocessing data: {e}")
        return

    # Filter columns
    print("🔍 Filtering to essential columns...")
    try:
        df_filtered = filter_essential_columns(df_processed)
        col_count = len(df_filtered.columns)
        _print_success(f"Filtered to {col_count} essential columns")
    except Exception as e:
        _print_error(f"Error filtering columns: {e}")
        df_filtered = df_processed

    # Save
    print("💾 Saving preprocessed data...")
    try:
        save_to_sqlite(df=df_filtered, db_path=str(db_path), table_name=output_table)
        _print_success(f"Saved {len(df_filtered):,} rows to database")
        print(f"\n{'=' * 70}\n")
    except Exception as e:
        _print_error(f"Error saving to database: {e}")
