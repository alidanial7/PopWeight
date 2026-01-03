import os

import numpy as np
import pandas as pd


def generate_realistic_dataset(n_samples=30000):
    np.random.seed(42)

    platforms = ["Facebook", "Instagram", "LinkedIn", "Twitter"]
    post_types = ["Image", "Video", "Link"]
    sentiments = ["Positive", "Negative", "Neutral"]

    data = []

    for _ in range(n_samples):
        platform = np.random.choice(platforms)
        post_type = np.random.choice(post_types)
        sentiment = np.random.choice(sentiments)

        likes = np.random.randint(50, 5000)
        comments = np.random.randint(10, 1000)
        shares = np.random.randint(5, 500)

        if platform == "Instagram":
            base_reach = (likes * 1.5) + (comments * 2.0) + (shares * 1.2)
        elif platform == "Twitter":
            base_reach = (likes * 0.8) + (comments * 1.5) + (shares * 5.0)
        elif platform == "LinkedIn":
            base_reach = (likes * 1.0) + (comments * 6.0) + (shares * 2.5)
        else:  # Facebook
            base_reach = (likes * 1.2) + (comments * 2.5) + (shares * 2.0)

        multiplier = 1.5 if post_type == "Video" else 1.0
        noise = np.random.normal(1000, 500)

        reach = int((base_reach * multiplier) + noise)
        reach = max(500, reach)  # حداقل ریچ

        # Impressions معمولاً 1.2 تا 2 برابر Reach است
        impressions = int(reach * np.random.uniform(1.1, 2.0))

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
                "Post Timestamp": "2025-01-01 10:00:00",  # به عنوان نمونه
            }
        )

    df = pd.DataFrame(data)

    os.makedirs("data", exist_ok=True)
    df.to_excel("data/social_media_engagement_data.xlsx", index=False)
    print("✅ Realistic dataset created successfully in 'data/' folder!")


if __name__ == "__main__":
    generate_realistic_dataset()
