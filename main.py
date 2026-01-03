"""Main script for engagement weight analysis with train/test modes.

This module provides the main entry point for the PopWeight analysis system.
It offers an interactive menu for:
- Training models on engagement data
- Testing/validating models on test data
- Analyzing correlations between engagement metrics and reach

The script is organized into modular functions for each major operation,
making it easy to maintain and extend.
"""

from pathlib import Path

from analysis import (
    analyze_correlation,
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
from utils import (
    load_training_results,
    save_training_results,
)
from utils.data_loading import load_processed_data


def train_model() -> None:
    """
    Train model on training data and find best options.

    This function performs the complete training workflow:
    1. Loads preprocessed training data
    2. Performs cross-sectional analysis with Linear and RF models
    3. Selects the best performing model
    4. Generates visualizations (heatmaps, facet grids)
    5. Provides statistical insights
    6. Identifies trending posts
    7. Saves training results for later validation

    The function compares Linear Regression and Random Forest models,
    selecting the one with the highest R² score for each segment.

    Raises
    ------
    ValueError
        If data cannot be loaded or no valid segments are found.
    """
    print("\n" + "=" * 80)
    print("🎓 TRAINING MODE: Learning Engagement Weights")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent

    # Load preprocessed training data
    try:
        df, _ = load_processed_data(project_root, "train", "processed")
    except ValueError as e:
        print(f"\n{e}")
        return

    # Step 1: Cross-Sectional Analysis
    # Train both Linear Regression and Random Forest models
    # to compare performance and select the best approach
    print("\n" + "=" * 80)
    print("📊 STEP 1: Cross-Sectional Analysis")
    print("=" * 80)

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

    # Compare models and select the best performing one
    # Selection is based on mean R² score across all segments
    results_df, selected_models = _select_best_model(
        results_df_linear, results_df_rf, rf_models
    )

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


def _select_best_model(results_df_linear, results_df_rf, rf_models) -> tuple:
    """
    Select the best performing model based on R² scores.

    Parameters
    ----------
    results_df_linear : pd.DataFrame
        Results from Linear Regression model.
    results_df_rf : pd.DataFrame
        Results from Random Forest model.
    rf_models : dict
        Dictionary of trained Random Forest models.

    Returns
    -------
    tuple
        Tuple of (best_results_df, selected_models) where selected_models
        is None for Linear Regression or the rf_models dict for RF.
    """
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
            return results_df_rf, rf_models
        else:
            print(" → Using Linear")
            return results_df_linear, None
    elif len(results_df_rf) > 0:
        print("✓ Using Random Forest (Linear failed)")
        return results_df_rf, rf_models
    else:
        print("✓ Using Linear Regression")
        return results_df_linear, None


def test_model() -> None:
    """
    Test model on test data and provide validation analysis.

    This function performs the complete validation workflow:
    1. Loads training results (weights and optionally models)
    2. Loads preprocessed test data
    3. Optionally loads training data for feature range validation
    4. Validates model performance on test set
    5. Generates validation visualizations and metrics

    The function handles cases where Random Forest models may not be
    available, falling back to weight-based predictions.

    Raises
    ------
    FileNotFoundError
        If training results cannot be found.
    ValueError
        If test data cannot be loaded.
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING MODE: Validating on Test Data")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent
    output_dir = project_root / "outputs"

    # Load training results (weights and optionally models)
    train_results_df, rf_models = _load_training_results(output_dir)
    if train_results_df is None:
        return

    # Load test data
    try:
        test_df, _ = load_processed_data(project_root, "test", "processed")
    except ValueError as e:
        print(f"\n{e}")
        return

    # Optionally load training data for feature range validation
    # This helps identify if test data has values outside training range
    train_df = _load_training_data_for_validation(project_root)

    # Perform validation analysis
    print("\n" + "=" * 80)
    print("🔍 VALIDATION ANALYSIS")
    print("=" * 80)

    print("\nValidating model on test data...")
    validation_results = validate_model(
        train_results_df,
        test_df,
        train_df=train_df,
        output_dir=output_dir,
        clip_outliers=True,
        rf_models=rf_models,
    )

    # Display validation results
    print_validation_results(validation_results)

    # Display summary
    _print_validation_summary(output_dir)


def _load_training_results(output_dir: Path) -> tuple:
    """
    Load training results from saved files.

    Parameters
    ----------
    output_dir : Path
        Directory containing training results.

    Returns
    -------
    tuple
        Tuple of (train_results_df, rf_models) or (None, None) on error.
    """
    print("\n📖 Loading training results...")
    try:
        results_path = output_dir / "training_results.db"
        train_results_df, rf_models = load_training_results(
            results_path, from_db=True, load_models=True
        )
        print(f"✓ Loaded training results: {len(train_results_df)} segments")
        if rf_models:
            print(f"✓ Loaded {len(rf_models)} Random Forest models")
        return train_results_df, rf_models
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease run 'train' mode first to generate training results.")
        return None, None
    except Exception:
        # Fallback if models not available
        try:
            results_path = output_dir / "training_results.db"
            train_results_df = load_training_results(results_path, from_db=True)
            print(f"✓ Loaded training results: {len(train_results_df)} segments")
            print("⚠️  Models not available (using weight-based prediction)")
            return train_results_df, None
        except Exception as e2:
            print(f"❌ {e2}")
            return None, None


def _load_training_data_for_validation(project_root: Path):
    """
    Load training data for feature range validation.

    Parameters
    ----------
    project_root : Path
        Root directory of the project.

    Returns
    -------
    pd.DataFrame | None
        Training DataFrame if loaded successfully, None otherwise.
    """
    print("\n📖 Loading training data for feature range validation...")
    try:
        train_df, _ = load_processed_data(project_root, "train", "processed")
        print(f"✓ Loaded training data: {len(train_df):,} rows")
        return train_df
    except Exception as e:
        print(f"⚠️  Warning: Could not load training data: {e}")
        print("   Validation will proceed without feature range checking.")
        return None


def _print_validation_summary(output_dir: Path) -> None:
    """
    Print validation summary with file locations.

    Parameters
    ----------
    output_dir : Path
        Directory containing validation outputs.
    """
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Validation results saved to: {output_dir}")
    print("  • Prediction vs Actual: prediction_vs_actual.png")
    print("  • Confusion Matrix: confusion_matrix.png")
    print("=" * 80 + "\n")


def correlation_likes_reach() -> None:
    """
    Calculate and display correlation between Likes and Reach.

    This is a convenience wrapper for the correlation analysis module.
    """
    analyze_correlation("Likes", "Reach", data_source="train")


def correlation_comments_reach() -> None:
    """
    Calculate and display correlation between Comments and Reach.

    This is a convenience wrapper for the correlation analysis module.
    """
    analyze_correlation("Comments", "Reach", data_source="train")


def correlation_shares_reach() -> None:
    """
    Calculate and display correlation between Shares and Reach.

    This is a convenience wrapper for the correlation analysis module.
    """
    analyze_correlation("Shares", "Reach", data_source="train")


def _display_menu() -> None:
    """
    Display the main menu options to the user.

    This function prints a formatted menu with all available options
    for the engagement weight analysis system.
    """
    print("\nSelect mode:")
    print("  1. Train - Learn weights from training data")
    print("  2. Test  - Validate weights on test data")
    print("  3. Correlation - Likes vs Reach")
    print("  4. Correlation - Comments vs Reach")
    print("  5. Correlation - Shares vs Reach")
    print("  q. Quit - Exit the program")
    print()


def _handle_menu_choice(choice: str) -> bool:
    """
    Handle user menu choice and execute corresponding action.

    Parameters
    ----------
    choice : str
        User's menu choice (1-5, 'q', 'train', 'test', etc.).

    Returns
    -------
    bool
        True if the program should continue, False if it should exit.
    """
    choice = choice.strip().lower()

    # Map choices to functions
    menu_actions = {
        "1": train_model,
        "train": train_model,
        "2": test_model,
        "test": test_model,
        "3": correlation_likes_reach,
        "4": correlation_comments_reach,
        "5": correlation_shares_reach,
    }

    # Handle quit
    if choice in ("q", "quit"):
        print("\n👋 Exiting...")
        print("=" * 80 + "\n")
        return False

    # Execute action if valid choice
    if choice in menu_actions:
        menu_actions[choice]()
        print("\n" + "-" * 80)
        print("Returning to main menu...")
        print("-" * 80)
        return True

    # Invalid choice
    print("❌ Invalid choice. Please enter 1-5, or 'q'.")
    print()
    return True


def main() -> None:
    """
    Main entry point for the engagement weight analysis system.

    This function provides an interactive menu-driven interface that allows
    users to:
    - Train models on engagement data
    - Validate models on test data
    - Analyze correlations between engagement metrics and reach

    The function runs in a loop until the user chooses to quit.

    Examples
    --------
    Run the main script:
        python main.py

    The script will display a menu and wait for user input.
    """
    print("\n" + "=" * 80)
    print("🔬 ENGAGEMENT WEIGHT ANALYSIS SYSTEM")
    print("=" * 80)

    # Main loop - continue until user quits
    while True:
        _display_menu()
        choice = input("Enter choice (1-5, or 'q' to quit): ")

        # Handle choice and check if we should continue
        if not _handle_menu_choice(choice):
            break


if __name__ == "__main__":
    main()
