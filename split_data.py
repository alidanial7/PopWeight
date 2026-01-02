"""Script to split base data into train and test datasets."""

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import load_social_media_data


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


def get_train_percentage() -> float:
    """
    Get train percentage from user input.

    Returns
    -------
    float
        Train percentage (0-100).
    """
    while True:
        try:
            percentage = input(
                "\nEnter train data percentage (0-100, e.g., 70 for 70%): "
            ).strip()

            percentage_float = float(percentage)

            if 0 < percentage_float < 100:
                return percentage_float
            else:
                print_error("Please enter a value between 0 and 100 (exclusive)")
        except ValueError:
            print_error("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled by user.")
            sys.exit(0)


def split_dataframe(
    df: pd.DataFrame, train_percentage: float, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to split.
    train_percentage : float
        Percentage of data for training (0-100).
    random_state : int, default 42
        Random state for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (train_df, test_df).
    """
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split index
    total_rows = len(df_shuffled)
    train_size = int(total_rows * (train_percentage / 100))

    # Split data
    train_df = df_shuffled.iloc[:train_size].copy()
    test_df = df_shuffled.iloc[train_size:].copy()

    return train_df, test_df


def main():
    """Split base data into train and test datasets."""
    print_header("📊 Data Splitting Tool")

    # Get project root directory
    project_root = Path(__file__).parent

    # Define paths
    base_file = project_root / "data" / "social_media_engagement_data.xlsx"
    train_output = project_root / "data" / "train.xlsx"
    test_output = project_root / "data" / "test.xlsx"

    # Check if base file exists
    if not base_file.exists():
        print_error(f"Base data file not found at {base_file}")
        print_info("Please ensure the file exists at the specified path")
        return

    print_info(f"Source file: {base_file.name}")
    print_info(f"Train output: {train_output.name}")
    print_info(f"Test output: {test_output.name}\n")

    # Load base data
    print("📖 Loading base data...")
    try:
        with tqdm(
            total=100, desc="Reading Excel", bar_format="{l_bar}{bar}| {n_fmt}%"
        ) as pbar:
            # Simulate progress for Excel reading
            for _ in range(0, 100, 20):
                time.sleep(0.05)
                pbar.update(20)

            df = load_social_media_data(file_path=str(base_file))
            pbar.update(100 - pbar.n)

        print_success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        columns_preview = ", ".join(df.columns[:5].tolist())
        columns_suffix = "..." if len(df.columns) > 5 else ""
        print(f"   Columns: {columns_preview}{columns_suffix}\n")
    except Exception as e:
        print_error(f"Error loading Excel file: {e}")
        return

    # Show data summary
    print("=" * 70)
    print("📋 Data Summary:")
    print("-" * 70)
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    if "Platform" in df.columns:
        platforms = df["Platform"].unique()
        print(f"  Platforms: {', '.join(platforms)}")
    if "Post Type" in df.columns:
        post_types = df["Post Type"].unique()
        print(f"  Post Types: {', '.join(post_types)}")
    print("=" * 70)

    # Get train percentage from user
    train_percentage = get_train_percentage()
    test_percentage = 100 - train_percentage

    print_info(f"Train percentage: {train_percentage:.1f}%")
    print_info(f"Test percentage: {test_percentage:.1f}%")

    # Confirm split
    print("\n" + "=" * 70)
    print("⚠️  WARNING: This will overwrite existing train.xlsx and test.xlsx files!")
    print("=" * 70)
    confirm = input("\nDo you want to proceed? (yes/no): ").strip().lower()

    if confirm not in ["yes", "y"]:
        print("\n👋 Operation cancelled.")
        return

    # Split data
    print("\n" + "=" * 70)
    print("🔄 Splitting Data")
    print("=" * 70)
    print("\nShuffling and splitting data...")

    try:
        with tqdm(
            total=100,
            desc="Splitting data",
            bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
        ) as pbar:
            train_df, test_df = split_dataframe(df, train_percentage)
            pbar.update(100)

        print_success(f"Train set: {len(train_df):,} rows ({train_percentage:.1f}%)")
        print_success(f"Test set: {len(test_df):,} rows ({test_percentage:.1f}%)")
    except Exception as e:
        print_error(f"Error splitting data: {e}")
        return

    # Save train data
    print("\n💾 Saving train data...")
    try:
        total_rows = len(train_df)

        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} " "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        with tqdm(
            total=total_rows,
            desc="Writing train.xlsx",
            unit="rows",
            bar_format=bar_format,
        ) as pbar:
            train_df.to_excel(train_output, index=False, engine="openpyxl")
            pbar.update(total_rows)

        print_success(f"Saved {len(train_df):,} rows to {train_output.name}")
    except Exception as e:
        print_error(f"Error saving train data: {e}")
        return

    # Save test data
    print("\n💾 Saving test data...")
    try:
        total_rows = len(test_df)

        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} " "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        with tqdm(
            total=total_rows,
            desc="Writing test.xlsx",
            unit="rows",
            bar_format=bar_format,
        ) as pbar:
            test_df.to_excel(test_output, index=False, engine="openpyxl")
            pbar.update(total_rows)

        print_success(f"Saved {len(test_df):,} rows to {test_output.name}")
    except Exception as e:
        print_error(f"Error saving test data: {e}")
        return

    # Final summary
    print("\n" + "=" * 70)
    print("✅ DATA SPLIT COMPLETE")
    print("=" * 70)
    print("\n📁 Files created:")
    print(f"  • {train_output.name}: {len(train_df):,} rows ({train_percentage:.1f}%)")
    print(f"  • {test_output.name}: {len(test_df):,} rows ({test_percentage:.1f}%)")
    print(f"\n📊 Total: {len(df):,} rows split successfully")
    print(f"📂 Location: {train_output.parent.absolute()}")
    print("\n💡 Next steps:")
    print("  1. python import_train.py")
    print("  2. python import_test.py")
    print("  3. python preprocess_train.py")
    print("  4. python preprocess_test.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
