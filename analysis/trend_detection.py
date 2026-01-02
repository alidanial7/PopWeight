"""Trend detection module for identifying trending posts."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def identify_trending_posts(
    df: pd.DataFrame,
    platform_col: str = "Platform",
    post_type_col: str = "Post Type",
    reach_col: str = "Reach_log",
    percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Identify trending posts as top percentile within each Platform-PostType.

    A post is considered 'Trending' if its Reach_log is in the top percentile
    (default 90th percentile = top 10%) of its specific Platform-PostType category.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with engagement features.
    platform_col : str, default "Platform"
        Name of the platform column.
    post_type_col : str, default "Post Type"
        Name of the post type column.
    reach_col : str, default "Reach_log"
        Name of the reach column to use for trending detection.
    percentile : float, default 90.0
        Percentile threshold (90.0 = top 10%).

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Is_Trending' column (1 for trending, 0 otherwise).
    """
    df = df.copy()

    # Initialize Is_Trending column
    df["Is_Trending"] = 0

    # Get unique combinations
    platforms = df[platform_col].unique()
    post_types = df[post_type_col].unique()

    for platform in platforms:
        for post_type in post_types:
            # Filter segment
            mask = (df[platform_col] == platform) & (df[post_type_col] == post_type)
            segment_indices = df[mask].index

            if len(segment_indices) == 0:
                continue

            # Calculate threshold for this segment
            segment_reach = df.loc[segment_indices, reach_col]
            threshold = np.percentile(segment_reach, percentile)

            # Mark trending posts
            trending_mask = df.loc[segment_indices, reach_col] >= threshold
            df.loc[segment_indices[trending_mask], "Is_Trending"] = 1

    return df


def train_trending_classifier(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "Is_Trending",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> tuple[RandomForestClassifier, dict[str, any]]:
    """
    Train a Random Forest classifier to predict trending posts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and Is_Trending column.
    feature_cols : list[str], optional
        List of feature columns to use. If None, uses engagement features.
    target_col : str, default "Is_Trending"
        Name of the target column.
    test_size : float, default 0.2
        Proportion of data to use for testing.
    random_state : int, default 42
        Random state for reproducibility.
    n_estimators : int, default 100
        Number of trees in the Random Forest.

    Returns
    -------
    tuple[RandomForestClassifier, dict[str, any]]
        Trained classifier and performance metrics.
    """
    from sklearn.model_selection import train_test_split

    # Default feature columns (engagement features)
    if feature_cols is None:
        feature_cols = [
            "Likes_log_log",
            "Comments_log_log",
            "Shares_log_log",
        ]

    # Check if all columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Prepare features and target
    X = df[feature_cols].values
    y = df[target_col].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train Random Forest
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    # Evaluate model
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
        if len(np.unique(y_test)) > 1
        else 0.0,
        "feature_importance": dict(
            zip(feature_cols, classifier.feature_importances_, strict=False)
        ),
    }

    return classifier, metrics


def calculate_engagement_zscore(
    df: pd.DataFrame,
    platform_col: str = "Platform",
    post_type_col: str = "Post Type",
    engagement_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate Z-score of engagement for each post.

    Z-score shows how many standard deviations a post's engagement stands
    above/below the mean for its Platform-PostType category.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with engagement features.
    platform_col : str, default "Platform"
        Name of the platform column.
    post_type_col : str, default "Post Type"
        Name of the post type column.
    engagement_cols : list[str], optional
        List of engagement columns. If None, uses default engagement features.

    Returns
    -------
    pd.DataFrame
        DataFrame with added Z-score columns for each engagement metric.
    """
    df = df.copy()

    # Default engagement columns
    if engagement_cols is None:
        engagement_cols = ["Likes_log_log", "Comments_log_log", "Shares_log_log"]

    # Check if columns exist
    missing_cols = [col for col in engagement_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Initialize Z-score columns
    for col in engagement_cols:
        df[f"{col}_zscore"] = 0.0

    # Get unique combinations
    platforms = df[platform_col].unique()
    post_types = df[post_type_col].unique()

    for platform in platforms:
        for post_type in post_types:
            # Filter segment
            mask = (df[platform_col] == platform) & (df[post_type_col] == post_type)
            segment_indices = df[mask].index

            if len(segment_indices) < 2:  # Need at least 2 samples for std
                continue

            # Calculate Z-scores for this segment
            for col in engagement_cols:
                segment_values = df.loc[segment_indices, col]
                mean_val = segment_values.mean()
                std_val = segment_values.std()

                if std_val > 0:  # Avoid division by zero
                    zscores = (segment_values - mean_val) / std_val
                    df.loc[segment_indices, f"{col}_zscore"] = zscores

    return df


def get_top_trending_posts(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    platform_col: str = "Platform",
    top_n: int = 5,
) -> dict[str, list[dict[str, any]]]:
    """
    Get top N trending posts for each platform with explanations.

    Explains why posts are outliers based on their engagement weights (α, β, γ).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with posts, trending labels, and Z-scores.
    results_df : pd.DataFrame
        Results DataFrame from analyze_engagement_weights with weights.
    platform_col : str, default "Platform"
        Name of the platform column.
    top_n : int, default 5
        Number of top trending posts to return per platform.

    Returns
    -------
    dict[str, list[dict[str, any]]]
        Dictionary mapping platform to list of top trending posts with explanations.
    """
    top_posts = {}

    platforms = df[platform_col].unique()

    for platform in platforms:
        platform_data = df[df[platform_col] == platform].copy()

        # Filter only trending posts
        trending_posts = platform_data[platform_data["Is_Trending"] == 1].copy()

        if len(trending_posts) == 0:
            top_posts[platform] = []
            continue

        # Sort by Reach_log (descending) and get top N
        trending_posts = trending_posts.nlargest(top_n, "Reach_log")

        platform_posts = []

        for _, post in trending_posts.iterrows():
            post_type = post["Post Type"]

            # Get weights for this Platform-PostType combination
            weights_row = results_df[
                (results_df["Platform"] == platform)
                & (results_df["Post Type"] == post_type)
            ]

            if len(weights_row) == 0:
                # Use default weights if not found
                alpha, beta, gamma = 0.33, 0.33, 0.34
            else:
                alpha = weights_row.iloc[0]["Alpha_Likes"]
                beta = weights_row.iloc[0]["Beta_Comments"]
                gamma = weights_row.iloc[0]["Gamma_Shares"]

            # Get Z-scores
            likes_zscore = post.get("Likes_log_log_zscore", 0.0)
            comments_zscore = post.get("Comments_log_log_zscore", 0.0)
            shares_zscore = post.get("Shares_log_log_zscore", 0.0)

            # Calculate weighted engagement score
            weighted_score = (
                alpha * likes_zscore + beta * comments_zscore + gamma * shares_zscore
            )

            # Determine which engagement type is most important
            if alpha >= beta and alpha >= gamma:
                primary_driver = "Likes"
                primary_weight = alpha
            elif beta >= gamma:
                primary_driver = "Comments"
                primary_weight = beta
            else:
                primary_driver = "Shares"
                primary_weight = gamma

            # Create explanation
            max_zscore = max(likes_zscore, comments_zscore, shares_zscore)
            explanation = (
                f"This post is trending because it has a high weighted engagement "
                f"score ({weighted_score:.2f} std devs above mean). "
                f"The {platform}-{post_type} segment prioritizes {primary_driver} "
                f"(weight: {primary_weight:.3f}), and this post's {primary_driver} "
                f"engagement is {max_zscore:.2f} standard deviations above average."
            )

            post_info = {
                "Post_ID": post.get("Post ID", "Unknown"),
                "Post_Type": post_type,
                "Reach_log": round(post["Reach_log"], 4),
                "Likes_zscore": round(likes_zscore, 2),
                "Comments_zscore": round(comments_zscore, 2),
                "Shares_zscore": round(shares_zscore, 2),
                "Weighted_Score": round(weighted_score, 2),
                "Alpha_Likes": round(alpha, 4),
                "Beta_Comments": round(beta, 4),
                "Gamma_Shares": round(gamma, 4),
                "Primary_Driver": primary_driver,
                "Explanation": explanation,
            }

            platform_posts.append(post_info)

        top_posts[platform] = platform_posts

    return top_posts


def print_trending_analysis(
    top_posts: dict[str, list[dict[str, any]]],
    classifier_metrics: dict[str, any],
) -> None:
    """
    Print trending analysis results in a readable format.

    Parameters
    ----------
    top_posts : dict[str, list[dict[str, any]]]
        Dictionary of top trending posts per platform.
    classifier_metrics : dict[str, any]
        Classifier performance metrics.
    """
    print("\n" + "=" * 80)
    print("📈 TREND DETECTION ANALYSIS")
    print("=" * 80)

    # Classifier Performance
    print("\n🤖 Random Forest Classifier Performance:")
    print("-" * 80)
    print(f"  • Accuracy: {classifier_metrics['accuracy']:.4f}")
    print(f"  • Precision: {classifier_metrics['precision']:.4f}")
    print(f"  • Recall: {classifier_metrics['recall']:.4f}")
    print(f"  • F1-Score: {classifier_metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC: {classifier_metrics['roc_auc']:.4f}")

    print("\n📊 Feature Importance:")
    print("-" * 80)
    for feature, importance in sorted(
        classifier_metrics["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  • {feature}: {importance:.4f}")

    # Top Trending Posts
    print("\n🔥 Top 5 Trending Posts by Platform:")
    print("=" * 80)

    for platform, posts in top_posts.items():
        if len(posts) == 0:
            print(f"\n{platform}: No trending posts found")
            continue

        print(f"\n{platform}:")
        print("-" * 80)

        for idx, post in enumerate(posts, 1):
            print(f"\n  {idx}. Post ID: {post['Post_ID']}")
            print(f"     Post Type: {post['Post_Type']}")
            print(f"     Reach (log): {post['Reach_log']:.4f}")
            print("     Z-Scores:")
            print(f"       • Likes: {post['Likes_zscore']:.2f}σ")
            print(f"       • Comments: {post['Comments_zscore']:.2f}σ")
            print(f"       • Shares: {post['Shares_zscore']:.2f}σ")
            print(f"     Weighted Engagement Score: {post['Weighted_Score']:.2f}σ")
            print("     Segment Weights:")
            print(f"       • Alpha (Likes): {post['Alpha_Likes']:.4f}")
            print(f"       • Beta (Comments): {post['Beta_Comments']:.4f}")
            print(f"       • Gamma (Shares): {post['Gamma_Shares']:.4f}")
            print(f"     Primary Driver: {post['Primary_Driver']}")
            print(f"     Explanation: {post['Explanation']}")

    print("\n" + "=" * 80 + "\n")
