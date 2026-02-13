# PopWeight — Implementation Plan (README)

This document specifies the step-by-step engineering tasks to implement the **PopWeight** pipeline in Python, using an Excel dataset located at:

- `./data/social_media_engagement_data.xlsx`

The Excel file contains at least these sheets:

- `Working File` (primary sheet to use)
- `social_media_engagement_data` (raw reference; not required for modeling)
- `Countries or Areas` (lookup table; optional for future work)

The goal is to:

1. **Learn platform- and post-type-specific interaction weights** (\alpha, \beta, \gamma) such that engagement interactions predict **Reach**.
2. Use the learned weights to compute an **Engagement Score** and build a **Trending classifier** (binary) where trending is defined by **Reach percentile within each segment**.

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
    storage.py
    diagnostics.py
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

Learn (\alpha, \beta, \gamma) per segment to predict `Reach_log`.

### Model

- Linear Regression per segment:
  - X = [`Likes_ll`, `Comments_ll`, `Shares_ll`]
  - y = `Reach_log`

- Store `alpha`, `beta`, `gamma`, and `intercept`.
- Also store regression training diagnostics per segment:
  - `n_train_rows`
  - `r2_train` (optional)

### Implement

`popweight/weights.py`:

- `fit_segment_weights(train_df) -> weights_df`
  - Iterate over segments
  - Handle small segments:
    - If a segment has < `MIN_SEGMENT_SAMPLES` (define, e.g., 20), either:
      - skip and mark as insufficient
      - or fall back to platform-only weights
      - or fall back to global weights

    - Must be deterministic and logged.

### Inputs

- `train_df` from a split

### Outputs

- `weights_df` with columns:
  - `seed`, `Platform`, `Post Type`, `Segment`, `alpha`, `beta`, `gamma`, `intercept`, `n_train`

---

## 9) Scoring: Predict `Reach_log` via Engagement Score (TASK)

### Objective

Compute predicted reach signal per row using learned weights.

### Formula

For each row in train/test within a segment:

`Score = intercept + alpha*Likes_ll + beta*Comments_ll + gamma*Shares_ll`

Interpretation:

- `Score` is the model’s prediction of `Reach_log`.

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

Evaluate how well the Score predicts `Reach_log`.

### Metrics

Compute on test set per split:

- R²
- MAE
- RMSE
- Pearson correlation

### Implement

`popweight/evaluation.py`:

- `regression_metrics(y_true, y_pred) -> dict`
- `evaluate_regression(test_scored_df) -> metrics_row`

### Inputs

- `test_scored_df` with `Reach_log` and `Score`

### Outputs

- `regression_metrics` table rows per seed

---

## 11) Trending Label Construction (TASK)

### Objective

Create a binary label `Trending` based on **Reach**, not Score.

### Definition

Within each segment, compute threshold:

- `thr = percentile(Reach, TREND_PERCENTILE)` using **train** or **combined train+test** for stability.

Label:

- `Trending = 1 if Reach >= thr else 0`

> Recommended: compute thresholds on **train only** per seed to avoid test leakage.

### Implement

`popweight/trending.py`:

- `compute_segment_thresholds(train_df, percentile) -> thresholds_df`
- `apply_trending_label(df, thresholds_df) -> df_labeled`

### Inputs

- `train_df` (for thresholds)
- `test_scored_df` (to label)

### Outputs

- `test_labeled_df` with `Trending`

---

## 12) Trending Classifier Training (TASK)

### Objective

Train a classifier to predict `Trending`.

### Features (recommended)

- `Score` (must)
- `Platform`, `Post Type`
- `Weekday Type`, `Time Periods`
- `Age Group`
- `Sentiment`

### Model options

- Primary: Gradient Boosting classifier (e.g., XGBoost if available; otherwise sklearn’s GradientBoostingClassifier)

### Implement

`popweight/models.py`:

- `train_trending_classifier(train_scored_labeled_df) -> model`
- `predict_trending(model, test_scored_labeled_df) -> y_pred, y_prob`

Note:

- For the classifier training data, you need to score and label the **train** set as well (not just test).

### Inputs

- `train_df` -> (weights) -> `train_scored_df` -> (thresholds) -> `train_scored_labeled_df`

### Outputs

- trained classifier model (optionally persisted)
- predictions on test

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
- Evaluate regression metrics vs `Reach_log`
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

### Required outputs

- Mean and standard deviation across seeds for:
  - Regression metrics
  - Classification metrics

- Save:
  - `outputs/metrics_regression.csv`
  - `outputs/metrics_classification.csv`
  - `outputs/weights.csv` (optionally averaged across seeds)

### Optional plots

- Predicted vs Actual (`Score` vs `Reach_log`)
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

- Trending thresholds must be computed using **train only** per split.

### C) Small Segment Strategy

Define and implement one strategy:

- Global fallback weights (learn one model using all train data)
- Platform-only fallback (learn weights per Platform)
- Drop the segment (exclude from evaluation)

This must be logged and consistent.

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
