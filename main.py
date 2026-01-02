"""Main script to load and explore social media engagement data."""

from pathlib import Path

from utils import load_social_media_data


def main():
    """Load data and print column information."""
    print("Loading social media engagement data...")

    # Load the data
    # Get the project root directory
    project_root = Path(__file__).parent
    train_path = project_root / "data" / "train.xlsx"
    df = load_social_media_data(file_path=train_path)

    print("\nData loaded successfully!")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Print column names
    print("Columns:")
    print("-" * 50)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    print("\n" + "=" * 50)
    print("Column Data Types:")
    print("-" * 50)
    print(df.dtypes)

    print("\n" + "=" * 50)
    print("First few rows:")
    print("-" * 50)
    print(df.head())


if __name__ == "__main__":
    main()
