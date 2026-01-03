# Analysis Documentation

The `analysis/` package contains modules for model training, validation, visualization,
insights generation, trend detection, and correlation analysis.

## 📦 Available Modules

### Models (`models.py`)

#### `analyze_engagement_weights(df, platform_col, post_type_col, min_samples, show_progress, model_type, target_col)`

Perform cross-sectional analysis of engagement weights.

Trains separate models (Linear Regression or Random Forest) for each
Platform-PostType combination and extracts normalized weights.

**Parameters:**
- `df` (pd.DataFrame): Preprocessed DataFrame with engagement features
- `platform_col` (str, default "Platform"): Name of the platform column
- `post_type_col` (str, default "Post Type"): Name of the post type column
- `min_samples` (int, default 10): Minimum number of samples required per segment
- `show_progress` (bool, default False): Whether to show progress bar
- `model_type` (str, default "linear"): Model type: "linear" or "rf"
- `target_col` (str, default "Reach_log"): Target column name

**Returns:**
- `tuple[pd.DataFrame, dict | None]`: Results table with columns: Platform, Post Type,
  Alpha_Likes, Beta_Comments, Gamma_Shares, R_Squared, N_Samples, Model_Type, and
  optionally a dictionary of trained models (for RF).

**Example:**
```python
from analysis import analyze_engagement_weights

results_df, models_dict = analyze_engagement_weights(
    df, model_type="linear", show_progress=True
)
```

#### `extract_weights_table(results_df)`

Extract weights table from analysis results.

**Example:**
```python
from analysis import extract_weights_table

weights_df = extract_weights_table(results_df)
```

### Validation (`validation.py`)

#### `validate_model(train_results_df, test_df, weights_df, verbose)`

Complete model validation workflow.

**Parameters:**
- `train_results_df` (pd.DataFrame): Training results DataFrame
- `test_df` (pd.DataFrame): Test DataFrame
- `weights_df` (pd.DataFrame): Weights DataFrame
- `verbose` (bool, default False): Whether to print detailed output

**Returns:**
- `dict`: Validation metrics dictionary

**Example:**
```python
from analysis import validate_model

metrics = validate_model(train_results_df, test_df, weights_df)
```

#### `calculate_predicted_engagement_score(df, weights_df, platform_col, post_type_col)`

Calculate predicted engagement scores using learned weights.

**Example:**
```python
from analysis import calculate_predicted_engagement_score

df["Predicted_Engagement_Score"] = calculate_predicted_engagement_score(
    df, weights_df
)
```

#### `calculate_regression_metrics(df, actual_col, predicted_col)`

Calculate regression metrics (MAE, RMSE, R²).

**Example:**
```python
from analysis import calculate_regression_metrics

metrics = calculate_regression_metrics(
    df, actual_col="Reach_log", predicted_col="Predicted_Engagement_Score"
)
```

#### `calculate_classification_metrics(df, actual_col, predicted_col, threshold)`

Calculate classification metrics (accuracy, precision, recall, F1).

**Example:**
```python
from analysis import calculate_classification_metrics

metrics = calculate_classification_metrics(
    df, actual_col="Reach_log", predicted_col="Predicted_Engagement_Score"
)
```

#### `plot_prediction_vs_actual(df, actual_col, predicted_col, save_path)`

Create scatter plot of predictions vs actual values.

**Example:**
```python
from analysis import plot_prediction_vs_actual

plot_prediction_vs_actual(
    df, save_path="outputs/prediction_vs_actual.png"
)
```

#### `plot_confusion_matrix(df, actual_col, predicted_col, threshold, save_path)`

Create confusion matrix visualization.

**Example:**
```python
from analysis import plot_confusion_matrix

plot_confusion_matrix(df, save_path="outputs/confusion_matrix.png")
```

#### `diagnose_validation_issues(train_results_df, test_df, weights_df, verbose)`

Diagnose potential validation issues.

**Checks:**
- Platform-PostType coverage mismatches
- Feature range outliers
- Missing weight mappings

**Example:**
```python
from analysis import diagnose_validation_issues

diagnose_validation_issues(train_results_df, test_df, weights_df)
```

### Visualizations (`visualizations.py`)

#### `create_heatmap(results_df, save_path)`

Create heatmap visualization of engagement weights.

**Example:**
```python
from analysis import create_heatmap

create_heatmap(results_df, save_path="outputs/gamma_heatmap.png")
```

#### `create_facet_grid_chart(results_df, save_path)`

Create facet grid chart showing weights across platforms and post types.

**Example:**
```python
from analysis import create_facet_grid_chart

create_facet_grid_chart(
    results_df, save_path="outputs/weights_facet_grid.png"
)
```

### Insights (`insights.py`)

#### `generate_statistical_insights(results_df)`

Generate statistical insights from training results.

**Returns:**
- `dict`: Dictionary containing statistical summaries

**Example:**
```python
from analysis import generate_statistical_insights

insights = generate_statistical_insights(results_df)
```

#### `print_statistical_insights(results_df)`

Print statistical insights in a formatted way.

**Example:**
```python
from analysis import print_statistical_insights

print_statistical_insights(results_df)
```

### Trend Detection (`trend_detection.py`)

#### `identify_trending_posts(df, platform_col, post_type_col, reach_col, percentile)`

Identify trending posts based on reach percentile.

**Example:**
```python
from analysis import identify_trending_posts

df_trending = identify_trending_posts(df, percentile=90.0)
```

#### `train_trending_classifier(df, platform_col, post_type_col)`

Train a classifier to predict trending posts.

**Example:**
```python
from analysis import train_trending_classifier

classifier, metrics = train_trending_classifier(df)
```

#### `get_top_trending_posts(df, results_df, platform_col, top_n)`

Get top N trending posts per platform-post type combination.

**Example:**
```python
from analysis import get_top_trending_posts

top_posts = get_top_trending_posts(df, results_df, top_n=5)
```

#### `print_trending_analysis(top_posts, classifier_metrics)`

Print trending analysis results.

**Example:**
```python
from analysis import print_trending_analysis

print_trending_analysis(top_posts, classifier_metrics)
```

### Correlation (`correlation.py`)

#### `analyze_correlation(col1, col2, data_source)`

Calculate and display correlation between two columns.

**Parameters:**
- `col1` (str): First column name
- `col2` (str): Second column name
- `data_source` (str, default "train"): Data source to use: "train" or "test"

**Example:**
```python
from analysis import analyze_correlation

analyze_correlation("Likes", "Reach", data_source="train")
analyze_correlation("Comments", "Reach", data_source="train")
analyze_correlation("Shares", "Reach", data_source="train")
```

#### `calculate_correlation(col1, col2, data_source)`

Calculate correlation coefficient between two columns.

**Returns:**
- `dict`: Dictionary containing correlation results

**Example:**
```python
from analysis.correlation import calculate_correlation

results = calculate_correlation("Likes", "Reach", data_source="train")
```

#### `interpret_correlation_strength(correlation)`

Interpret the strength and direction of a correlation coefficient.

**Returns:**
- `tuple[str, str]`: Tuple of (strength, direction) strings

**Example:**
```python
from analysis.correlation import interpret_correlation_strength

strength, direction = interpret_correlation_strength(0.75)
# Returns: ("strong", "positive")
```

## 🔄 Usage Examples

### Complete Analysis Workflow

```python
from analysis import (
    analyze_engagement_weights,
    extract_weights_table,
    create_heatmap,
    create_facet_grid_chart,
    generate_statistical_insights,
    print_statistical_insights,
    identify_trending_posts,
    train_trending_classifier,
    validate_model,
    analyze_correlation,
)

# 1. Analyze engagement weights
results_df, models_dict = analyze_engagement_weights(df, show_progress=True)

# 2. Extract weights table
weights_df = extract_weights_table(results_df)

# 3. Create visualizations
create_heatmap(results_df, save_path="outputs/gamma_heatmap.png")
create_facet_grid_chart(
    results_df, save_path="outputs/weights_facet_grid.png"
)

# 4. Generate insights
print_statistical_insights(results_df)

# 5. Identify trending posts
df_trending = identify_trending_posts(df, percentile=90.0)
classifier, metrics = train_trending_classifier(df)

# 6. Validate model
metrics = validate_model(train_results_df, test_df, weights_df)

# 7. Analyze correlations
analyze_correlation("Likes", "Reach", data_source="train")
```

