# Data-Driven Engagement Weighting for Content Popularity Prediction

This repository implements a novel framework for predicting social media popularity by **learning dynamic engagement weights** instead of using fixed formulas.

## 📌 Research Motivation

In most social media studies, Engagement Rate (E) is calculated using a fixed linear combination:
$E = \text{Likes} + \text{Comments} + \text{Shares}$

However, this assumes every interaction has equal value. This project proposes a data-driven approach where weights ($\alpha, \beta, \gamma$) are learned automatically to reflect the true importance of each interaction type:
$$E = (\alpha \times \text{Likes}) + (\beta \times \text{Comments}) + (\gamma \times \text{Shares})$$

## ✨ Key Features

- **Dynamic Weighting:** Automatically learns coefficients for Likes, Comments, and Shares.
- **Log-Log Normalization:** Handles the high skewness of social media engagement data (inspired by DFW-PP).
- **Comparison Engine:** Built-in benchmarking against fixed-weight baseline models.
- **Context-Awareness:** Evaluates how weights change across different post types (Video vs. Image).

## 📂 Dataset Overview: Social Media Engagement

This project utilizes the **Social Media Engagement Dataset** (`data/train.xlsx`), which provides a comprehensive collection of metrics for analyzing how users interact with content across various social platforms.

### 📊 Feature Descriptions

The dataset consists of **18 columns**, which can be categorized into four main groups:

#### 1. Content Identification & Metadata

- **Platform:** The social media network where the post was published. The dataset includes posts from four major platforms: **Facebook**, **Instagram**, **Twitter**, and **LinkedIn**, providing a diverse cross-platform perspective on engagement patterns.

- **Post ID:** A unique identifier (UUID format) for each individual post, enabling precise tracking and referencing of specific content pieces.

- **Post Type:** The format of the content. The dataset contains three main content types: **Image**, **Video**, and **Link** posts, allowing analysis of how different content formats drive engagement differently.

- **Post Content:** The actual text content, captions, or hashtags used in the post. This field contains the raw textual content that accompanies the media, which can be analyzed for sentiment, keywords, or content themes.

- **Post Timestamp:** The exact date and time when the content was uploaded. The dataset spans from **March 2021 to March 2024**, providing a three-year longitudinal view of social media engagement trends.

#### 2. Engagement & Performance Metrics (Target Variables)

These metrics form the core of the popularity prediction task and are used to learn the optimal engagement weights:

- **Likes:** Total number of "like" interactions received. This is the most common form of engagement and typically has the highest volume.

- **Comments:** Total number of user comments under the post. Comments represent deeper engagement as they require more user effort than likes.

- **Shares:** Number of times the post was reshared or retweeted. Shares indicate the highest level of engagement, as users are actively promoting the content to their own networks.

- **Impressions:** Total number of times the post was displayed on users' screens (including multiple views by the same user). This metric reflects the potential audience size.

- **Reach:** Total number of unique users who saw the post. Unlike impressions, reach counts each user only once, providing a measure of unique audience exposure.

- **Engagement Rate:** A calculated metric representing the level of interaction relative to the audience size. This is typically computed as `(Likes + Comments + Shares) / Reach` or similar formulas, and serves as a normalized measure of content performance.

#### 3. Audience Demographics

These features describe the characteristics of the users who engaged with the content:

- **Audience Age:** The predominant age group (numerical value) of the users interacting with the content. This helps understand which demographic segments are most responsive to different types of posts.

- **Audience Gender:** The primary gender distribution of the engaged audience. The dataset includes three categories: **Male**, **Female**, and **Other**, reflecting diverse audience compositions.

- **Audience Location:** Geographical data indicating where the majority of the audience is located. The dataset includes a wide range of countries and regions, from major markets to smaller nations, enabling geographic analysis of engagement patterns.

- **Audience Interests:** Categorized interests or keywords that describe the engaged users' preferences. These can include topics like technology, fashion, sports, or other thematic categories that help understand audience alignment with content.

#### 4. Contextual & Marketing Data

These optional fields provide additional context about the content's purpose and origin:

- **Campaign ID:** Identifier for specific marketing campaigns the post belongs to. This field links posts that are part of coordinated marketing efforts, allowing analysis of campaign-level performance.

- **Sentiment:** The emotional tone of the content or user feedback. The dataset includes three sentiment categories: **Positive**, **Neutral**, and **Negative**, which can influence how audiences respond to content.

- **Influencer ID:** Identifier for the content creator or influencer associated with the post. This field helps track performance across different creators and understand how creator characteristics impact engagement.

### 📈 Data Characteristics

The dataset contains **999 social media posts** with **18 features** covering content metadata, engagement metrics, audience demographics, and contextual information. The data spans multiple platforms (Facebook, Instagram, Twitter, LinkedIn) and content types (Image, Video, Link), providing a comprehensive foundation for learning dynamic engagement weights that can adapt to different contexts.

**Data Structure:**

- **Numerical Features:** Engagement metrics (Likes, Comments, Shares, Impressions, Reach) and Audience Age are stored as integers, while Engagement Rate is a floating-point value.
- **Categorical Features:** Platform, Post Type, Audience Gender, Sentiment, and other text-based fields are stored as strings/objects.
- **Temporal Feature:** Post Timestamp is stored as a datetime object, enabling time-series analysis and temporal pattern detection.

## 🔧 Development Guidelines

### Type Safety

- **Always use type hints** for function parameters, return types, and class attributes
- Use `typing` module for complex types (e.g., `List`, `Dict`, `Optional`, `Union`)
- Prefer type annotations over comments for type information
- Use `mypy` or similar type checkers to validate type safety before committing
