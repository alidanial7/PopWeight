"""Training workflow for engagement weight analysis.

This module handles the complete training workflow including:
- Loading training data
- Training Linear and Random Forest models
- Model selection and comparison
- Generating visualizations
- Statistical insights
- Trend detection
- Saving training results
"""

from pathlib import Path

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
    train_trending_classifier,
)
from utils import save_training_results
from utils.data_loading import load_processed_data


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
    # workflows/ is one level down from project root
    project_root = Path(__file__).parent.parent

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

    # Step 4: Statistical Insights
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

    # Display trend analysis summary
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

    # Display completion summary
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
