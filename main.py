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

    # Try both Linear and Random Forest models
    print("Training models (Linear + Random Forest)...")
    results_df_linear, _ = analyze_engagement_weights(
        df,
        min_samples=10,
        show_progress=True,
        model_type="linear",
        target_col="Reach_log",
    )

    results_df_rf, rf_models = analyze_engagement_weights(
        df,
        min_samples=10,
        show_progress=True,
        model_type="rf",
        target_col="Engagement_Rate",
    )

    # Compare models and select best
    if len(results_df_linear) > 0 and len(results_df_rf) > 0:
        linear_mean_r2 = results_df_linear["R_Squared"].mean()
        rf_mean_r2 = results_df_rf["R_Squared"].mean()
        print(
            f"📊 Comparison: Linear R²={linear_mean_r2:.4f}, "
            f"RF R²={rf_mean_r2:.4f}",
            end="",
        )

        if rf_mean_r2 > linear_mean_r2:
            print(" → Using RF")
            results_df = results_df_rf
            selected_models = rf_models
        else:
            print(" → Using Linear")
            results_df = results_df_linear
            selected_models = None
    elif len(results_df_rf) > 0:
        print("✓ Using Random Forest (Linear failed)")
        results_df = results_df_rf
        selected_models = rf_models
    else:
        print("✓ Using Linear Regression")
        results_df = results_df_linear
        selected_models = None

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
    print("\n📈 STEP 3: Generating Visualizations")
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    heatmap_path = output_dir / "gamma_heatmap.png"
    create_heatmap(results_df, save_path=heatmap_path)

    facet_path = output_dir / "weights_facet_grid.png"
    create_facet_grid_chart(results_df, save_path=facet_path)
    print("  ✓ Visualizations saved")

    # Step 4: Statistical Insights (concise)
    print("\n🔍 STEP 4: Statistical Insights")
    insights = generate_statistical_insights(results_df)
    print_statistical_insights(insights)

    # Step 5: Trend Detection
    print("\n🔥 STEP 5: Trend Detection")
    df_with_trends = identify_trending_posts(df, percentile=90.0)
    trending_count = df_with_trends["Is_Trending"].sum()
    print(f"  Identified {trending_count} trending posts")

    df_with_zscore = calculate_engagement_zscore(df_with_trends)
    classifier, classifier_metrics = train_trending_classifier(df_with_zscore)
    top_posts = get_top_trending_posts(df_with_zscore, results_df, top_n=5)

    # Concise trend analysis
    print(
        f"  Classifier: Acc={classifier_metrics.get('accuracy', 0)*100:.1f}%, "
        f"F1={classifier_metrics.get('f1_score', 0):.4f}"
    )
    total_top_posts = sum(len(posts) for posts in top_posts.values())
    print(f"  Top trending posts: {total_top_posts} across platforms")

    # Save training results
    print("\n💾 Saving training results...")
    results_path = save_training_results(
        results_df, output_dir, models_dict=selected_models
    )
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
        train_results_df, rf_models = load_training_results(
            results_path, from_db=True, load_models=True
        )
        print(f"✓ Loaded training results: {len(train_results_df)} segments")
        if rf_models:
            print(f"✓ Loaded {len(rf_models)} Random Forest models")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease run 'train' mode first to generate training results.")
        return
    except Exception:
        # Fallback if models not available
        try:
            train_results_df = load_training_results(results_path, from_db=True)
            rf_models = None
            print(f"✓ Loaded training results: {len(train_results_df)} segments")
            print("⚠️  Models not available (using weight-based prediction)")
        except Exception as e2:
            print(f"❌ {e2}")
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
            rf_models=rf_models,
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


def calculate_correlation(col1: str, col2: str, data_source: str = "train") -> None:
    """
    Calculate and display correlation between two columns.

    Parameters
    ----------
    col1 : str
        First column name.
    col2 : str
        Second column name.
    data_source : str, default "train"
        Data source to use: "train" or "test".
    """
    print("\n" + "=" * 80)
    print(f"📊 CORRELATION ANALYSIS: {col1} vs {col2}")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent
    db_path = project_root / "data" / f"{data_source}.db"
    table_name = f"{data_source}_data_raw"

    # Load data
    print(f"\n📖 Loading {data_source} data...")
    try:
        with tqdm(
            total=100,
            desc="Reading database",
            bar_format="{l_bar}{bar}| {n_fmt}%",
        ) as pbar:
            for _ in range(0, 100, 25):
                time.sleep(0.05)
                pbar.update(25)

            df = read_from_sqlite(db_path=str(db_path), table_name=table_name)
            pbar.update(100 - pbar.n)

        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(f"\nPlease ensure you have run:" f"\n  1. python import_{data_source}.py")
        return

    # Check if columns exist
    if col1 not in df.columns:
        print(f"❌ Column '{col1}' not found in data")
        print(f"Available columns: {', '.join(df.columns[:10])}...")
        return

    if col2 not in df.columns:
        print(f"❌ Column '{col2}' not found in data")
        print(f"Available columns: {', '.join(df.columns[:10])}...")
        return

    # Calculate correlation
    print("\n🔍 Calculating correlation...")
    correlation = df[col1].corr(df[col2])

    # Display results
    print("\n" + "-" * 80)
    print("CORRELATION RESULTS")
    print("-" * 80)
    print(f"Column 1: {col1}")
    print(f"Column 2: {col2}")
    print(f"Correlation Coefficient: {correlation:.4f}")

    # Interpret correlation
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if correlation > 0 else "negative"
    print(f"Interpretation: {strength} {direction} correlation")

    # Additional statistics
    print("\n" + "-" * 80)
    print("STATISTICAL SUMMARY")
    print("-" * 80)
    print(f"{col1}:")
    print(f"  Mean: {df[col1].mean():.2f}")
    print(f"  Std:  {df[col1].std():.2f}")
    print(f"  Min:  {df[col1].min():.2f}")
    print(f"  Max:  {df[col1].max():.2f}")
    print(f"\n{col2}:")
    print(f"  Mean: {df[col2].mean():.2f}")
    print(f"  Std:  {df[col2].std():.2f}")
    print(f"  Min:  {df[col2].min():.2f}")
    print(f"  Max:  {df[col2].max():.2f}")

    print("\n" + "=" * 80)
    print("✅ CORRELATION ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


def correlation_likes_reach():
    """Calculate correlation between Likes and Reach."""
    calculate_correlation("Likes", "Reach", data_source="train")


def correlation_comments_reach():
    """Calculate correlation between Comments and Reach."""
    calculate_correlation("Comments", "Reach", data_source="train")


def correlation_shares_reach():
    """Calculate correlation between Shares and Reach."""
    calculate_correlation("Shares", "Reach", data_source="train")


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
        print("  3. Correlation - Likes vs Reach")
        print("  4. Correlation - Comments vs Reach")
        print("  5. Correlation - Shares vs Reach")
        print("  q. Quit - Exit the program")
        print()

        choice = input("Enter choice (1-5, or 'q' to quit): ").strip().lower()

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
        elif choice == "3":
            correlation_likes_reach()
            print("\n" + "-" * 80)
            print("Returning to main menu...")
            print("-" * 80)
        elif choice == "4":
            correlation_comments_reach()
            print("\n" + "-" * 80)
            print("Returning to main menu...")
            print("-" * 80)
        elif choice == "5":
            correlation_shares_reach()
            print("\n" + "-" * 80)
            print("Returning to main menu...")
            print("-" * 80)
        elif choice == "q" or choice == "quit":
            print("\n👋 Exiting...")
            print("=" * 80 + "\n")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5, or 'q'.")
            print()


if __name__ == "__main__":
    main()
