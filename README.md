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

## 📊 Dataset
The project uses the **Social Media Engagement Report** (from Kaggle), which includes:
- Post Metadata (Type, Time, Date)
- Interactions (Likes, Comments, Shares)
- Target Metric (Reach/Impressions)

