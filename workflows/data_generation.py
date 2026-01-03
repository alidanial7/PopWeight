"""Data generation workflow.

This module contains the workflow for generating synthetic social media
engagement datasets.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


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


def _get_sample_count() -> int:
    """
    Get number of samples from user input.

    Returns
    -------
    int
        Number of samples to generate.
    """
    while True:
        try:
            count_str = input(
                "\nEnter number of samples to generate "
                "(default: 30000, press Enter for default): "
            ).strip()

            # Use default if empty
            if not count_str:
                return 30000

            count = int(count_str)

            if count > 0:
                return count
            else:
                _print_error("Please enter a positive number")
        except ValueError:
            _print_error("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled by user.")
            raise


def generate_data(n_samples: int | None = None) -> None:
    """
    Generate realistic synthetic social media engagement dataset.

    If n_samples is not provided, prompts the user interactively for the
    number of samples to generate.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate. If None, prompts user interactively.
    """
    _print_header("📊 Generating Realistic Synthetic Data")

    project_root = Path(__file__).parent.parent
    output_path = project_root / "data" / "social_media_engagement_data.xlsx"

    # Initial setup
    if n_samples is None:
        n_samples = _get_sample_count()

    np.random.seed(42)
    platforms = ["Facebook", "Instagram", "LinkedIn", "Twitter"]
    post_types = ["Image", "Video", "Link"]
    sentiments = ["Positive", "Negative", "Neutral"]

    # Start date for generating varied timestamps
    start_date = datetime(2025, 1, 1)
    data = []

    info_msg = f"Generating {n_samples:,} samples with platform-specific logic..."
    _print_info(info_msg)

    with tqdm(total=n_samples, desc="Creating samples") as pbar:
        for _ in range(n_samples):
            platform = np.random.choice(platforms)
            post_type = np.random.choice(post_types)
            sentiment = np.random.choice(sentiments)

            # Generate random timestamp throughout the year
            random_days = np.random.randint(0, 365)
            random_hours = np.random.randint(0, 24)
            post_time = start_date + timedelta(days=random_days, hours=random_hours)

            # Generate initial engagement metrics
            likes = np.random.randint(10, 5000)
            comments = np.random.randint(5, 1200)
            shares = np.random.randint(1, 800)

            # Platform-specific reach calculations
            # Modified algorithm to break likes dominance
            if platform == "Instagram":
                # Instagram: Still like-focused but with good comment impact
                base_reach = (likes * 1.8) + (comments * 2.5) + (shares * 1.5)
            elif platform == "Twitter":
                # Twitter: Shares (Retweets) are king
                base_reach = (likes * 0.5) + (comments * 1.0) + (shares * 12.0)
            elif platform == "LinkedIn":
                # LinkedIn: Comments drive network spread
                base_reach = (likes * 0.7) + (comments * 10.0) + (shares * 3.0)
            else:  # Facebook
                # Facebook: Balance between shares and likes
                base_reach = (likes * 1.0) + (comments * 2.0) + (shares * 5.0)

            # Content type attractiveness multiplier
            if post_type == "Video":
                type_multiplier = 2.0
            elif post_type == "Image":
                type_multiplier = 1.2
            else:  # Link
                type_multiplier = 0.8

            # Add noise for natural variance
            noise = np.random.normal(500, 200)

            reach = int((base_reach * type_multiplier) + noise)
            reach = max(100, reach)

            impressions = int(reach * np.random.uniform(1.2, 2.5))

            data.append(
                {
                    "Platform": platform,
                    "Post Type": post_type,
                    "Sentiment": sentiment,
                    "Likes": likes,
                    "Comments": comments,
                    "Shares": shares,
                    "Reach": reach,
                    "Impressions": impressions,
                    "Audience Age": np.random.randint(18, 65),
                    "Post Timestamp": post_time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            pbar.update(1)

    df = pd.DataFrame(data)

    # Save to Excel
    output_path.parent.mkdir(exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")

    dataset_msg = f"Dataset created: {len(df):,} rows × {len(df.columns)} columns"
    _print_success(dataset_msg)
    _print_success(f"Saved to: {output_path}")
    print(f"\n{'=' * 70}\n")
