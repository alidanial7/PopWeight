"""Main script for engagement weight analysis with train/test modes."""

import time
from pathlib import Path

from tqdm import tqdm

from analysis import (
    analyze_engagement_weights,
    calculate_engagement_zscore,
    create_facet_grid_chart,
    create_heatmap,
    extract_weights_table,
    generate_statistical_insights,
    get_top_trending_posts,
    identify_trending_posts,
    print_statistical_insights,
    print_trending_analysis,
    print_validation_results,
    train_trending_classifier,
    validate_model,
)
from utils import load_training_results, read_from_sqlite, save_training_results


def train_model():
    """
    Train model on training data and find best options.

    Performs cross-sectional analysis, generates visualizations,
    statistical insights, and saves training results.
    """
    print("\n" + "=" * 80)
    print("🎓 TRAINING MODE: Learning Engagement Weights")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent
    db_path = project_root / "data" / "train.db"
    table_name = "train_data_processed"

    # Load preprocessed data
    print("\n📖 Loading preprocessed training data...")
    try:
        with tqdm(
            total=100,
            desc="Reading database",
            bar_format="{l_bar}{bar}| {n_fmt}%",
        ) as pbar:
            # Simulate progress for database reading
            for _ in range(0, 100, 25):
                time.sleep(0.05)
                pbar.update(25)

            df = read_from_sqlite(db_path=str(db_path), table_name=table_name)
            pbar.update(100 - pbar.n)

        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(
            "\nPlease ensure you have run:"
            "\n  1. python import_train.py"
            "\n  2. python preprocess_train.py"
        )
        return

    # Step 1: Cross-Sectional Analysis
    print("\n" + "=" * 80)
    print("📊 STEP 1: Cross-Sectional Analysis")
    print("=" * 80)
    print(
        "\nTraining Linear Regression models for each Platform-PostType "
        "combination..."
    )

    results_df = analyze_engagement_weights(df, min_samples=10, show_progress=True)

    if len(results_df) == 0:
        print("❌ No valid segments found. Check your data and try again.")
        return

    print(f"\n✓ Analyzed {len(results_df)} Platform-PostType combinations")

    # Step 2: Display Results Table
    print("\n" + "=" * 80)
    print("📋 STEP 2: Results Table")
    print("=" * 80)

    weights_table = extract_weights_table(results_df)
    print("\nEngagement Weights Summary:")
    print("-" * 80)
    print(weights_table.to_string(index=False))
    print()

    # Step 3: Create Visualizations
    print("\n" + "=" * 80)
    print("📈 STEP 3: Visualizations")
    print("=" * 80)

    # Create output directory for figures
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Heatmap
    print("\nCreating heatmap: Gamma (Shares) importance...")
    with tqdm(
        total=100,
        desc="  Generating heatmap",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        heatmap_path = output_dir / "gamma_heatmap.png"
        create_heatmap(results_df, save_path=heatmap_path)
        pbar.update(100)

    # Facet Grid Chart
    print("\nCreating facet grid: Weight changes across Post Types...")
    with tqdm(
        total=100,
        desc="  Generating facet grid",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        facet_path = output_dir / "weights_facet_grid.png"
        create_facet_grid_chart(results_df, save_path=facet_path)
        pbar.update(100)

    # Step 4: Statistical Insights
    print("\n" + "=" * 80)
    print("🔍 STEP 4: Statistical Insights")
    print("=" * 80)

    print("\nGenerating statistical insights...")
    with tqdm(
        total=100,
        desc="  Computing insights",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        insights = generate_statistical_insights(results_df)
        pbar.update(100)

    print_statistical_insights(insights)

    # Step 5: Trend Detection
    print("\n" + "=" * 80)
    print("🔥 STEP 5: Trend Detection")
    print("=" * 80)

    print("\nIdentifying trending posts (top 10% by Platform-PostType)...")
    with tqdm(
        total=100,
        desc="  Identifying trends",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        df_with_trends = identify_trending_posts(df, percentile=90.0)
        pbar.update(100)

    trending_count = df_with_trends["Is_Trending"].sum()
    print(f"✓ Identified {trending_count} trending posts")

    print("\nCalculating engagement Z-scores...")
    with tqdm(
        total=100,
        desc="  Computing Z-scores",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        df_with_zscore = calculate_engagement_zscore(df_with_trends)
        pbar.update(100)

    print("✓ Calculated Z-scores for all engagement metrics")

    print("\nTraining Random Forest classifier...")
    with tqdm(
        total=100,
        desc="  Training classifier",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        classifier, classifier_metrics = train_trending_classifier(df_with_zscore)
        pbar.update(100)

    print("✓ Classifier trained successfully")

    print("\nExtracting top trending posts...")
    with tqdm(
        total=100,
        desc="  Analyzing posts",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        top_posts = get_top_trending_posts(df_with_zscore, results_df, top_n=5)
        pbar.update(100)

    print_trending_analysis(top_posts, classifier_metrics)

    # Save training results
    print("\n💾 Saving training results...")
    results_path = save_training_results(results_df, output_dir)
    print(f"✓ Training results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"  • Heatmap: {heatmap_path.name}")
    print(f"  • Facet Grid: {facet_path.name}")
    print(f"  • Training Results: {results_path.name}")
    print(
        f"\n📊 Analyzed {len(results_df)} segments across "
        f"{results_df['Platform'].nunique()} platforms and "
        f"{results_df['Post Type'].nunique()} post types"
    )
    print(f"🔥 Identified {trending_count} trending posts")
    print("\n💡 Next step: Run 'test' mode to validate on test data")
    print("=" * 80 + "\n")


def test_model():
    """
    Test model on test data and provide validation analysis.

    Loads learned weights from training and validates on test set.
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING MODE: Validating on Test Data")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent
    output_dir = project_root / "outputs"

    # Load training results
    print("\n📖 Loading training results...")
    try:
        results_path = output_dir / "training_results.db"
        train_results_df = load_training_results(results_path, from_db=True)
        print(f"✓ Loaded training results: {len(train_results_df)} segments")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease run 'train' mode first to generate training results.")
        return

    # Load test data
    print("\n📖 Loading preprocessed test data...")
    try:
        test_db_path = project_root / "data" / "test.db"
        with tqdm(
            total=100,
            desc="Reading test database",
            bar_format="{l_bar}{bar}| {n_fmt}%",
        ) as pbar:
            for _ in range(0, 100, 25):
                time.sleep(0.05)
                pbar.update(25)

            test_df = read_from_sqlite(
                db_path=str(test_db_path), table_name="test_data_processed"
            )
            pbar.update(100 - pbar.n)

        print(f"✓ Loaded {len(test_df):,} rows × {len(test_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        print(
            "\nPlease ensure you have run:"
            "\n  1. python import_test.py"
            "\n  2. python preprocess_test.py"
        )
        return

    # Load training data for feature range checking
    print("\n📖 Loading training data for feature range validation...")
    try:
        train_db_path = project_root / "data" / "train.db"
        train_df = read_from_sqlite(
            db_path=str(train_db_path), table_name="train_data_processed"
        )
        print(f"✓ Loaded training data: {len(train_df):,} rows")
    except Exception as e:
        print(f"⚠️  Warning: Could not load training data for range check: {e}")
        print("   Validation will proceed without feature range checking.")
        train_df = None

    # Perform validation
    print("\n" + "=" * 80)
    print("🔍 VALIDATION ANALYSIS")
    print("=" * 80)

    print("\nValidating model on test data...")
    with tqdm(
        total=100,
        desc="  Validating model",
        bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]",
    ) as pbar:
        validation_results = validate_model(
            train_results_df,
            test_df,
            train_df=train_df,
            output_dir=output_dir,
            clip_outliers=True,
        )
        pbar.update(100)

    # Print validation results
    print_validation_results(validation_results)

    # Summary
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Validation results saved to: {output_dir}")
    print("  • Prediction vs Actual: prediction_vs_actual.png")
    print("  • Confusion Matrix: confusion_matrix.png")
    print("=" * 80 + "\n")


def main():
    """
    Main entry point with train/test mode selection.

    Provides interactive menu to choose between training and testing.
    Loops until user types 'q' to quit.
    """
    print("\n" + "=" * 80)
    print("🔬 ENGAGEMENT WEIGHT ANALYSIS SYSTEM")
    print("=" * 80)

    while True:
        print("\nSelect mode:")
        print("  1. Train - Learn weights from training data")
        print("  2. Test  - Validate weights on test data")
        print("  q. Quit - Exit the program")
        print()

        choice = input("Enter choice (1, 2, or 'q' to quit): ").strip().lower()

        if choice == "1" or choice == "train":
            train_model()
            print("\n" + "-" * 80)
            print("Returning to main menu...")
            print("-" * 80)
        elif choice == "2" or choice == "test":
            test_model()
            print("\n" + "-" * 80)
            print("Returning to main menu...")
            print("-" * 80)
        elif choice == "q" or choice == "quit":
            print("\n👋 Exiting...")
            print("=" * 80 + "\n")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 'q'.")
            print()


if __name__ == "__main__":
    main()
