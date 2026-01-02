"""Main script to load and explore social media engagement data."""

from pathlib import Path

from utils import list_tables, load_social_media_data, read_from_sqlite, save_to_sqlite


def main():
    """Load data and print column information."""
    print("Loading social media engagement data...")

    # Get the project root directory
    project_root = Path(__file__).parent
    train_path = project_root / "data" / "train.xlsx"
    db_path = project_root / "data" / "social_media_data.db"

    # Load the data from Excel
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

    # Save to SQLite database
    print("\n" + "=" * 50)
    print("Saving data to SQLite database...")
    print("-" * 50)
    save_to_sqlite(df, db_path, table_name="social_media_data")
    print(f"Data saved to: {db_path}")

    # List tables in the database
    print("\n" + "=" * 50)
    print("Tables in database:")
    print("-" * 50)
    tables = list_tables(db_path)
    for table in tables:
        print(f"  - {table}")

    # Read data back from SQLite
    print("\n" + "=" * 50)
    print("Reading data from SQLite database...")
    print("-" * 50)
    df_from_db = read_from_sqlite(db_path, table_name="social_media_data")
    print(
        f"Data read successfully! "
        f"Shape: {df_from_db.shape[0]} rows × {df_from_db.shape[1]} columns"
    )

    # Example: Read with a custom query
    print("\n" + "=" * 50)
    print("Example: Reading posts with more than 100 likes...")
    print("-" * 50)
    query = "SELECT * FROM social_media_data WHERE Likes > 100 LIMIT 5"
    df_filtered = read_from_sqlite(db_path, query=query)
    print(f"Found {len(df_filtered)} posts (showing first 5):")
    if len(df_filtered) > 0:
        print(df_filtered[["Post ID", "Likes", "Comments", "Shares"]].head())


if __name__ == "__main__":
    main()
