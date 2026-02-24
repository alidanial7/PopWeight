# PopWeight — Implementation Plan (README)

This document specifies the step-by-step engineering tasks to implement the **PopWeight** pipeline in Python, using an Excel dataset located at:

- `./data/social_media_engagement_data.xlsx`

The Excel file contains at least these sheets:

- `Working File` (primary sheet to use)
- `social_media_engagement_data` (raw reference; not required for modeling)
- `Countries or Areas` (lookup table; optional for future work)

The goal is to:

1. **Learn platform- and post-type-specific interaction weights** (\alpha, \beta, \gamma) such that engagement interactions predict **log(ER_proxy)** where ER_proxy = (Likes+Comments+Shares)/Reach.
2. Use the learned weights to compute an **Engagement Score** and build a **Trending classifier** (binary) where trending is defined by **engagement rate proxy (ER_proxy) percentile within each segment**.

> IMPORTANT: **Do not use `Engagement Rate`** in training or validation. It is derived from Reach and interaction counts and may cause leakage.

---

## 0) Repository Structure (Required)

```
project/
  main.py
  data/
    social_media_engagement_data.xlsx
  popweight/
    __init__.py
    config.py
    io_excel.py
    schema.py
    cleaning.py
    features.py
    splits.py
    weights.py
    scoring.py
    trending.py
    models.py
    evaluation.py
    reporting.py
    storage.py
    diagnostics.py
  tests/
    test_trending.py
  outputs/
    (generated artifacts)
  logs/
    (optional)
```

### Output artifacts (minimum)

- `outputs/results.sqlite` (SQLite database containing processed data, learned weights, and metrics)
- `outputs/weights.csv` (learned weights per segment)
- `outputs/metrics_regression.csv` (regression metrics per split and aggregate)
- `outputs/metrics_classification.csv` (classification metrics per split and aggregate)
- Optional plots: `outputs/pred_vs_actual.png`, `outputs/confusion_matrix.png`, `outputs/weights_heatmap.png`

---

## 1) Global Configuration (TASK)

### Objective

Centralize all settings in one place.

### Implement

Create `popweight/config.py` with:

- `DATA_PATH = "data/social_media_engagement_data.xlsx"`
- `SHEET_NAME = "Working File"`
- `SQLITE_PATH = "outputs/results.sqlite"`
- `RANDOM_SEEDS = [0,1,2,3,4,5,6,7,8,9]` (default 10 repeats)
- `TRAIN_RATIO = 0.8`
- `TREND_PERCENTILE = 0.9` (top 10% per segment)
- Outlier controls:
  - `REMOVE_TOP_REACH_PERCENTILE = 0.995` (optional)
  - `MIN_REACH = 1`

- Feature transform controls:
  - `USE_DOUBLE_LOG = True`

- Segment definition:
  - `SEGMENT_KEYS = ["Platform", "Post Type"]`
  - `MIN_SEGMENT_SAMPLES = 20`

### Inputs

None.

### Outputs

Python config module.

---

## 2) Excel Ingestion (TASK)

### Objective

Load the dataset from the Excel file, using the `Working File` sheet.

### Implement

`popweight/io_excel.py`:

- `load_working_file(path: str, sheet: str) -> pandas.DataFrame`
  - Read Excel with `openpyxl` engine
  - Preserve column names exactly as found
  - Trim whitespace from string columns (including trailing spaces in headers)

### Inputs

- `data/social_media_engagement_data.xlsx`
- `sheet = "Working File"`

### Outputs

- DataFrame `df_raw`

---

## 3) Column Normalization + Schema Validation (TASK)

### Objective

Ensure required columns exist and are consistently named.

### Required columns (must exist)

- `Platform`
- `Post Type`
- `Likes`
- `Comments`
- `Shares`
- `Reach`
- `Weekday Type`
- `Time Periods`
- `Age Group`
- `Sentiment`

### Implement

`popweight/schema.py`:

- `normalize_columns(df) -> df`
  - Strip header whitespace
  - Normalize common variants (e.g., double spaces)

- `validate_required_columns(df) -> None`
  - Raise a clear exception listing missing columns

### Inputs

- `df_raw`

### Outputs

- `df_schema_ok`

---

## 4) Data Cleaning + Integrity Rules (TASK)

### Objective

Clean numeric columns and enforce integrity constraints.

### Rules

- Convert `Likes`, `Comments`, `Shares`, `Reach` to numeric (coerce errors to NaN)
- Drop rows with NaN in any of: Likes, Comments, Shares, Reach
- Remove rows where:
  - `Reach < MIN_REACH`
  - `Likes < 0` or `Comments < 0` or `Shares < 0`

- Optional outlier removal:
  - Remove rows with `Reach` above `REMOVE_TOP_REACH_PERCENTILE` (global or per segment; choose global for simplicity)

### Implement

`popweight/cleaning.py`:

- `clean_core_columns(df, config) -> (df_clean, report_dict)`
  - `report_dict` includes counts of dropped rows by reason

### Inputs

- `df_schema_ok`

### Outputs

- `df_clean`
- `cleaning_report.json` (stored in SQLite and/or `outputs/cleaning_report.json`)

---

## 5) Feature Engineering (TASK)

### Objective

Create transformed interaction features and modeling-ready fields.

### Transforms

For each row:

- `Likes_ll = log(log(Likes + 1) + 1)`
- `Comments_ll = log(log(Comments + 1) + 1)`
- `Shares_ll = log(log(Shares + 1) + 1)`
- `Reach_log = log(Reach + 1)`

Additionally:

- `Segment = Platform + "__" + Post Type` (string key)
- Convert categorical columns to clean categories (strip whitespace):
  - `Platform`, `Post Type`, `Weekday Type`, `Time Periods`, `Age Group`, `Sentiment`

### Implement

`popweight/features.py`:

- `add_transforms(df) -> df`
- `add_segment_key(df, keys=["Platform","Post Type"]) -> df`

### Inputs

- `df_clean`

### Outputs

- `df_features`

---

## 6) SQLite Storage (TASK — Recommended)

### Objective

Persist intermediate results for faster iteration and easy querying.

### Tables

- `raw_working_file` (optional)
- `clean_data`
- `features_data`
- `weights` (learned weights per segment per split)
- `regression_metrics` (per split)
- `classification_metrics` (per split)
- `aggregate_metrics` (summary)

### Implement

`popweight/storage.py`:

- `init_db(sqlite_path) -> None`
- `write_df(table_name, df) -> None`
- `read_df(table_name, where=None) -> df`

### Inputs

- `df_clean`, `df_features`, later results

### Outputs

- `outputs/results.sqlite`

---

## 7) Repeated Train/Test Split with Segment Coverage (TASK)

### Objective

Create repeated train/test splits while ensuring segments are present.

### Requirements

- Use `TRAIN_RATIO`.
- Repeat for each seed in `RANDOM_SEEDS`.
- Ensure every segment in test exists in train (segment coverage). If not, re-sample or drop those test rows (prefer re-sample up to N tries).

### Implement

`popweight/splits.py`:

- `make_splits(df, seeds, train_ratio) -> List[Split]`
  - Each Split contains `seed`, `train_df`, `test_df`

### Inputs

- `df_features`

### Outputs

- List of splits

---

## 8) Weight Learning per Segment (Regression) (TASK)

### Objective

Learn (\alpha, \beta, \gamma) per segment to predict `log(ER_proxy)` (aligned with Trending).

### Model

- `Eng = Likes + Comments + Shares`
- `ER_proxy = Eng / Reach` (safe division: Reach.clip(lower=1), ER_proxy.clip(lower=1e-10))
- y = `log(ER_proxy)`
- X = [`Likes_ll`, `Comments_ll`, `Shares_ll`]

- Store `alpha`, `beta`, `gamma`, and `intercept`.
- Also store regression training diagnostics per segment:
  - `n_train_rows`
  - `r2_train` against log(ER_proxy)

### Implement

`popweight/weights.py`:

- `fit_segment_weights(train_df, seed=None, min_segment_samples=20) -> weights_df`
  - Iterate over segments
  - Handle small segments: if a segment has < `MIN_SEGMENT_SAMPLES`,
    fall back to **global weights** (one model on all train data)
  - Must be deterministic and logged

### Inputs

- `train_df` from a split (with Segment, Likes_ll, Comments_ll, Shares_ll, Likes, Comments, Shares, Reach)

### Outputs

- `weights_df` with columns:
  - `seed`, `Platform`, `Post Type`, `Segment`, `alpha`, `beta`, `gamma`, `intercept`, `n_train`, `r2_train`, `strategy`

---

## 9) Scoring: Predict `log(ER_proxy)` via Engagement Score (TASK)

### Objective

Compute predicted engagement-rate-proxy signal per row using learned weights.

### Formula

For each row in train/test within a segment:

`Score = intercept + alpha*Likes_ll + beta*Comments_ll + gamma*Shares_ll`

Interpretation:

- `Score` is the model’s prediction of `log(ER_proxy)`.

### Implement

`popweight/scoring.py`:

- `apply_scores(df, weights_df) -> df_scored`
  - Join weights by (`Platform`, `Post Type`) and compute Score
  - If weights missing for a segment, apply fallback strategy (must be defined)

### Inputs

- `test_df`
- `weights_df` for the same seed

### Outputs

- `test_scored_df` containing `Score`

---

## 10) Regression Evaluation (TASK)

### Objective

Evaluate how well the Score predicts `log(ER_proxy)`.

### Metrics

Compute on test set per split:

- R²
- MAE
- RMSE
- Pearson correlation

### Implement

`popweight/evaluation.py`:

- `regression_metrics(y_true, y_pred) -> dict`
- `evaluate_regression(test_scored_df) -> metrics_row` (computes log(ER_proxy) from Likes, Comments, Shares, Reach)

### Inputs

- `test_scored_df` with Likes, Comments, Shares, Reach and Score

### Outputs

- `regression_metrics` table rows per seed

---

## 11) Trending Label Construction (TASK)

### Objective

Create a binary label `Trending` based on **engagement rate proxy (ER_proxy)**, not Reach or Score.

### Definition

- `Eng = Likes + Comments + Shares`
- `ER_proxy = Eng / Reach` (safe division: Reach.clip(lower=1e-10); rows with Reach less than MIN_REACH removed in cleaning)

Within each segment, compute threshold on **train only** per seed:

- `threshold = quantile(ER_proxy, TREND_PERCENTILE)`

Label:

- `Trending = 1 if ER_proxy >= threshold else 0`

For rows whose segment has no threshold (e.g., unseen segment in test), use the **median threshold across all segments** for that seed.

> Eng and ER_proxy are computed inside `trending.py`; the pipeline must have `Likes`, `Comments`, `Shares`, `Reach` available before trend labeling.

### Implement

`popweight/trending.py`:

- `compute_segment_thresholds(train_df, percentile) -> thresholds_df` (returns Segment, threshold for ER_proxy)
- `apply_trending_label(df, thresholds_df) -> df_labeled`

### Inputs

- `train_df` (for thresholds; must have Likes, Comments, Shares, Reach, Segment)
- `df` to label (must have same columns)

### Outputs

- `df_labeled` with `Trending` (trending rate ~10% in train with TREND_PERCENTILE=0.9)

---

## 12) Trending Classifier Training (TASK)

### Objective

Train a classifier to predict `Trending`.

### Features

- `Score` (must)
- `Eng_log = log(Likes + Comments + Shares + 1)` (computed inside models)
- `Platform`, `Post Type`
- `Weekday Type`, `Time Periods`
- `Age Group`
- `Sentiment`

### Model options

- Primary: Gradient Boosting classifier (sklearn GradientBoostingClassifier)

### Threshold selection (leakage prevention)

- Split training set into train_sub (80%) and val_sub (20%) stratified by Trending
- Fit classifier on train_sub only
- Choose probability threshold that maximizes F1 on val_sub (scan 0.01 to 0.99)
- Apply that fixed threshold to test set
- Do not select threshold on the same data used to fit the classifier

### Implement

`popweight/models.py`:

- `train_trending_classifier(train_scored_labeled_df, val_ratio=0.2) -> (model, threshold)`
- `predict_trending(model, test_scored_labeled_df, threshold) -> y_pred, y_prob, threshold`

Note:

- For the classifier training data, you need to score and label the **train** set as well (not just test).
- DataFrame must have Likes, Comments, Shares for Eng_log computation.

### Inputs

- `train_df` -> (weights) -> `train_scored_df` -> (thresholds) -> `train_scored_labeled_df`

### Outputs

- trained classifier model and chosen threshold
- predictions on test using the fixed threshold

---

## 13) Classification Evaluation (TASK)

### Objective

Measure performance of trending prediction.

### Metrics

On test set per split:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Implement

`popweight/evaluation.py`:

- `classification_metrics(y_true, y_pred) -> dict`

### Inputs

- `test_scored_labeled_df` with `Trending`
- predicted `y_pred`

### Outputs

- `classification_metrics` table rows per seed

---

## 14) Baseline Comparison (TASK — Required for Paper)

### Objective

Compare PopWeight against a fixed-weight baseline.

### Baseline score

- `Score_baseline = Likes_ll + Comments_ll + Shares_ll` (equal weights; no intercept)

### Steps

For each split:

- Compute `Score_baseline` for train/test
- Evaluate regression metrics vs `log(ER_proxy)` (same target as PopWeight)
- Train the same classifier using `Score_baseline` instead of `Score`
- Evaluate classification metrics

### Outputs

- Baseline metrics tables
- Comparison summary table:
  - `delta_accuracy`, `delta_f1`, etc.

---

## 15) Aggregation + Reporting (TASK)

### Objective

Aggregate metrics across repeats and produce final outputs.

### Implement

`popweight/reporting.py`:

- `aggregate_regression_metrics(metrics_list) -> regression_df`
- `aggregate_classification_metrics(metrics_list) -> classification_df`
- `aggregate_weights(weights_list) -> weights_df`
- `save_reports(regression_df, classification_df, weights_df, output_dir) -> None`

### Required outputs

- Mean and standard deviation across seeds for:
  - Regression metrics
  - Classification metrics

- Save:
  - `outputs/metrics_regression.csv`
  - `outputs/metrics_classification.csv`
  - `outputs/weights.csv` (optionally averaged across seeds)

### Optional plots

- Predicted vs Actual (`Score` vs `log(ER_proxy)`)
- Confusion matrix
- Heatmap of `gamma` (Shares weight) by Platform × Post Type

---

## 16) Diagnostics (TASK)

### Objective

Detect common pipeline failures early.

### Checks

- Missing segments in train or test
- Too-small segments
- NaNs after transforms
- Unexpected category values (e.g., trailing spaces)

### Implement

`popweight/diagnostics.py`:

- `run_diagnostics(df_features, splits, weights_df) -> report`

### Output

- `outputs/diagnostics_report.json`

---

## 17) Orchestration via `main.py` (TASK)

### Objective

Provide a simple CLI menu or command-based runner.

### Minimum commands

1. `load` — read Excel and validate schema
2. `clean` — clean core columns
3. `features` — add transforms + segment key
4. `split` — build repeated splits
5. `fit_weights` — learn weights per segment per split
6. `score` — compute Score on train/test
7. `trend_label` — build trending labels
8. `train_classifier` — train + evaluate classifier
9. `baseline` — run baseline pipeline
10. `report` — aggregate and export CSVs
11. `diagnostics` — run diagnostics

### Input

- `data/social_media_engagement_data.xlsx`

### Output

- All outputs under `outputs/`

---

## Notes / Decisions (Must be Implemented)

### A) Engagement Rate

- Do not use it anywhere in training or evaluation.

### B) Leakage Prevention

- Trending thresholds (ER_proxy per segment) must be computed using **train only** per split.
- Classifier probability threshold must be chosen on a **validation split** (stratified 80/20), not on the training data used to fit the classifier.

### C) Small Segment Strategy

**Implemented**: Global fallback weights. When a segment has fewer than MIN_SEGMENT_SAMPLES rows, use weights learned from all train data.

### D) Reproducibility

- All repeats must be controlled by explicit random seeds.
- Store seeds and split sizes in SQLite.

---

## Acceptance Criteria

Implementation is considered complete when:

1. Running the pipeline produces `outputs/results.sqlite` with the required tables.
2. `outputs/weights.csv` exists and contains (\alpha, \beta, \gamma) per `Platform × Post Type`.
3. `outputs/metrics_regression.csv` and `outputs/metrics_classification.csv` exist and include per-seed metrics plus aggregated mean/std.
4. Baseline results are produced and comparable to PopWeight.
5. Diagnostics report is generated and indicates no fatal issues.
