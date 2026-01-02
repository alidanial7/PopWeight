"""Statistical insights generation for engagement weight analysis."""

import pandas as pd


def generate_statistical_insights(results_df: pd.DataFrame) -> dict[str, any]:
    """
    Generate statistical insights from engagement weight analysis.

    Identifies high-impact interactions and key patterns across platforms
    and post types.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results DataFrame from analyze_engagement_weights.

    Returns
    -------
    dict[str, any]
        Dictionary containing various statistical insights.
    """
    insights = {}

    # Insight 1: High-Impact Interaction - Comments vs Shares by Platform
    insights["high_impact_interactions"] = {}

    for platform in results_df["Platform"].unique():
        platform_data = results_df[results_df["Platform"] == platform]

        if len(platform_data) == 0:
            continue

        # Find post type with highest Comments weight
        max_comments_idx = platform_data["Beta_Comments"].idxmax()
        max_comments_row = platform_data.loc[max_comments_idx]

        # Find post type with highest Shares weight
        max_shares_idx = platform_data["Gamma_Shares"].idxmax()
        max_shares_row = platform_data.loc[max_shares_idx]

        insights["high_impact_interactions"][platform] = {
            "most_comments_dependent": {
                "post_type": max_comments_row["Post Type"],
                "beta_weight": round(max_comments_row["Beta_Comments"], 4),
            },
            "most_shares_dependent": {
                "post_type": max_shares_row["Post Type"],
                "gamma_weight": round(max_shares_row["Gamma_Shares"], 4),
            },
        }

    # Insight 2: Overall weight statistics by platform
    insights["platform_statistics"] = {}

    for platform in results_df["Platform"].unique():
        platform_data = results_df[results_df["Platform"] == platform]

        if len(platform_data) == 0:
            continue

        insights["platform_statistics"][platform] = {
            "avg_alpha_likes": round(platform_data["Alpha_Likes"].mean(), 4),
            "avg_beta_comments": round(platform_data["Beta_Comments"].mean(), 4),
            "avg_gamma_shares": round(platform_data["Gamma_Shares"].mean(), 4),
            "avg_r_squared": round(platform_data["R_Squared"].mean(), 4),
            "total_segments": len(platform_data),
        }

    # Insight 3: Best performing segments (highest R-squared)
    top_n = min(5, len(results_df))
    top_segments = results_df.nlargest(top_n, "R_Squared")[
        ["Platform", "Post Type", "R_Squared", "N_Samples"]
    ].to_dict("records")

    insights["top_performing_segments"] = top_segments

    # Insight 4: Weight dominance patterns
    insights["dominance_patterns"] = {}

    for platform in results_df["Platform"].unique():
        platform_data = results_df[results_df["Platform"] == platform]

        if len(platform_data) == 0:
            continue

        # Count which weight type dominates most often
        platform_data = platform_data.copy()
        platform_data["dominant_weight"] = platform_data[
            ["Alpha_Likes", "Beta_Comments", "Gamma_Shares"]
        ].idxmax(axis=1)

        dominance_counts = platform_data["dominant_weight"].value_counts().to_dict()

        insights["dominance_patterns"][platform] = {
            "likes_dominant": dominance_counts.get("Alpha_Likes", 0),
            "comments_dominant": dominance_counts.get("Beta_Comments", 0),
            "shares_dominant": dominance_counts.get("Gamma_Shares", 0),
        }

    return insights


def print_statistical_insights(insights: dict[str, any]) -> None:
    """
    Print statistical insights in a readable format.

    Parameters
    ----------
    insights : dict[str, any]
        Insights dictionary from generate_statistical_insights.
    """
    print("\n" + "=" * 80)
    print("📊 STATISTICAL INSIGHTS")
    print("=" * 80)

    # High-Impact Interactions
    print("\n🔍 High-Impact Interactions (Comments vs Shares by Platform):")
    print("-" * 80)
    for platform, data in insights["high_impact_interactions"].items():
        print(f"\n{platform}:")
        comments_info = data["most_comments_dependent"]
        shares_info = data["most_shares_dependent"]
        print(
            f"  • Most Comments-Dependent: {comments_info['post_type']} "
            f"(β = {comments_info['beta_weight']})"
        )
        print(
            f"  • Most Shares-Dependent: {shares_info['post_type']} "
            f"(γ = {shares_info['gamma_weight']})"
        )

    # Platform Statistics
    print("\n📈 Platform Statistics:")
    print("-" * 80)
    for platform, stats in insights["platform_statistics"].items():
        print(f"\n{platform}:")
        print(f"  • Average Alpha (Likes): {stats['avg_alpha_likes']}")
        print(f"  • Average Beta (Comments): {stats['avg_beta_comments']}")
        print(f"  • Average Gamma (Shares): {stats['avg_gamma_shares']}")
        print(f"  • Average R²: {stats['avg_r_squared']}")
        print(f"  • Total Segments: {stats['total_segments']}")

    # Top Performing Segments
    print("\n🏆 Top 5 Performing Segments (by R²):")
    print("-" * 80)
    for idx, segment in enumerate(insights["top_performing_segments"], 1):
        print(
            f"{idx}. {segment['Platform']} - {segment['Post Type']}: "
            f"R² = {segment['R_Squared']:.4f} "
            f"(n = {segment['N_Samples']})"
        )

    # Dominance Patterns
    print("\n🎯 Weight Dominance Patterns by Platform:")
    print("-" * 80)
    for platform, patterns in insights["dominance_patterns"].items():
        print(f"\n{platform}:")
        print(f"  • Likes (α) Dominant: {patterns['likes_dominant']} segments")
        print(f"  • Comments (β) Dominant: {patterns['comments_dominant']} segments")
        print(f"  • Shares (γ) Dominant: {patterns['shares_dominant']} segments")

    print("\n" + "=" * 80 + "\n")
