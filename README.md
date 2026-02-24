# PopWeight — Complete Project Documentation

**A pipeline for learning engagement weights and predicting "Trending" social media posts.**

This document explains the entire project in detail. You can understand what the code does **without reading a single line of source code**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Complete Pipeline Pseudocode](#3-complete-pipeline-pseudocode)
4. [Repository Structure](#4-repository-structure)
5. [Configuration](#5-configuration)
6. [Module-by-Module Explanation](#6-module-by-module-explanation)
7. [CLI Commands](#7-cli-commands)
8. [Output Files](#8-output-files)
9. [Data Flow Diagram](#9-data-flow-diagram)
10. [Important Design Decisions](#10-important-design-decisions)

---

## 1. Overview

### What Does PopWeight Do?

1. **Learns segment-specific weights** (α, β, γ) so that a weighted combination of Likes, Comments, and Shares predicts **log(ER_proxy)**, where:
   - `Eng` = total engagement = Likes + Comments + Shares
   - `ER_proxy = Eng / Reach` (engagement rate proxy). This normalization
     allows fair comparison across posts with different exposure levels.
   - Each segment = a unique (Platform, Post Type) combination (e.g., Facebook__Image, Instagram__Video)

2. **Computes a Score** per post: `Score = intercept + α·Likes_ll + β·Comments_ll + γ·Shares_ll` (this is the model's prediction of log(ER_proxy)).

3. **Builds a binary Trending classifier** that predicts whether a post will be in the top 10% by ER_proxy within its segment. The classifier uses **Score** (primary learned signal) and **Eng_log** (auxiliary: log of total engagement) plus categorical features (Platform, Post Type, Weekday Type, etc.).

### Input Data

- **Source**: `data/social_media_engagement_data.xlsx` (sheet: "Working File")
- **Required columns**: Platform, Post Type, Likes, Comments, Shares, Reach, Weekday Type, Time Periods, Age Group, Sentiment
- Each row = one social media post

### Output Artifacts

- `outputs/results.sqlite` — SQLite database with all intermediate tables
- `outputs/metrics_regression.csv` — R², MAE, RMSE, Pearson per seed
- `outputs/metrics_classification.csv` — Accuracy, Precision, Recall, F1 per seed
- `outputs/weights.csv` — Learned (α, β, γ, intercept) per segment per seed
- `outputs/diagnostics_report.json` — Pipeline health check

---

## 2. Quick Start

```bash
# Run full pipeline (load → clean → features → ... → report → diagnostics)
python main.py all

# Or run individual commands:
python main.py load clean features split fit_weights report
```

---

## 3. Complete Pipeline Pseudocode

This pseudocode describes **exactly** what happens when you run `python main.py all`. No code reading required.

**Notation**: Subscript `_i` denotes the value for row i (e.g., Likes_i = Likes for row i).

```
═══════════════════════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════════════════════
  Dataset D (Excel: data/social_media_engagement_data.xlsx, sheet "Working File")
  Each row i: (Likes_i, Comments_i, Shares_i, Reach_i, Platform_i, Post_Type_i,
               Weekday_Type_i, Time_Periods_i, Age_Group_i, Sentiment_i)

  Config: TRAIN_RATIO=0.8, TREND_PERCENTILE=0.9, RANDOM_SEEDS=[0..9]
          MIN_REACH=1, REMOVE_TOP_REACH_PERCENTILE=0.995
          MIN_SEGMENT_SAMPLES=20

═══════════════════════════════════════════════════════════════════════════
STEP 1: LOAD & VALIDATE (io_excel.py, schema.py)
═══════════════════════════════════════════════════════════════════════════
  1.1 Read Excel with openpyxl engine
  1.2 Strip whitespace from column names and string values
  1.3 Normalize column names (collapse multiple spaces to single space)
  1.4 Validate: all of [Platform, Post Type, Likes, Comments, Shares, Reach,
       Weekday Type, Time Periods, Age Group, Sentiment] must exist
  1.5 Save to SQLite table "raw_working_file"

═══════════════════════════════════════════════════════════════════════════
STEP 2: CLEAN (cleaning.py)
═══════════════════════════════════════════════════════════════════════════
  2.1 Convert Likes, Comments, Shares, Reach to numeric (non-numeric → NaN)
  2.2 Drop rows with NaN in any of these four columns
  2.3 Drop rows where Reach < MIN_REACH (e.g., 1)
  2.4 Drop rows where Likes < 0 or Comments < 0 or Shares < 0
  2.5 Drop rows where Reach > quantile(Reach, 0.995)  // top 0.5% outlier
  2.6 Return df_clean and report (counts of dropped rows)
  2.7 Save to SQLite table "clean_data"

  Pseudocode:
    for col in [Likes, Comments, Shares, Reach]:
        df[col] = to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[Likes, Comments, Shares, Reach])
    df = df[df.Reach >= MIN_REACH]
    df = df[(df.Likes >= 0) & (df.Comments >= 0) & (df.Shares >= 0)]
    thr = df.Reach.quantile(0.995)
    df = df[df.Reach <= thr]

═══════════════════════════════════════════════════════════════════════════
STEP 3: FEATURE TRANSFORMS (features.py)
═══════════════════════════════════════════════════════════════════════════
  3.1 add_transforms:
      For each row i:
          Likes_ll_i  = log(log(Likes_i  + 1) + 1)
          Comments_ll_i = log(log(Comments_i + 1) + 1)
          Shares_ll_i = log(log(Shares_i + 1) + 1)
          Reach_log_i = log(Reach_i + 1)
      Why double-log: Likes/Comments/Shares are heavy-tailed (few viral posts,
      many low-engagement). log(log(x+1)+1) compresses the scale, reduces outlier
      impact, and gives a more linear relationship for the regression.

  3.2 add_segment_key:
      Segment_i = Platform_i + "__" + Post_Type_i   // e.g., "Facebook__Image"
      Strip whitespace from: Platform, Post Type, Weekday Type, Time Periods,
                             Age Group, Sentiment

  3.3 Save to SQLite table "features_data"

═══════════════════════════════════════════════════════════════════════════
STEP 4: SPLIT (splits.py)
═══════════════════════════════════════════════════════════════════════════
  For each seed in RANDOM_SEEDS (0..9):
    4.1 Split D into D_train (80%) and D_test (20%)
        - Stratify by Segment (so each segment appears proportionally in both)
        - random_state = seed (+ try_num if retry)
    4.2 Check: every segment in D_test must also appear in D_train
    4.3 If any test segment is missing from train:
        - Retry up to MAX_SPLIT_TRIES (20) times with different random_state
        - If still failing: drop test rows whose segment is not in train
    4.4 Store Split(seed, train_df, test_df)

  Result: List of 10 Split objects.

═══════════════════════════════════════════════════════════════════════════
STEP 5: LEARN WEIGHTS (weights.py)
═══════════════════════════════════════════════════════════════════════════
  Target variable (for each row i):
    Eng_i = Likes_i + Comments_i + Shares_i   // total engagement
    reach_safe_i = max(Reach_i, 1)
    ER_proxy_i = (Eng_i / reach_safe_i).clip(lower=1e-10)
    y_i = log(ER_proxy_i)

  Features: X_i = [Likes_ll_i, Comments_ll_i, Shares_ll_i]

  5.1 Fit GLOBAL model on ALL of D_train:
      y = b0 + α_global·Likes_ll + β_global·Comments_ll + γ_global·Shares_ll
      (Use this as fallback for small segments)

  5.2 For each segment k = (Platform, Post Type):
      Dk_train = rows in D_train with Segment == k
      if |Dk_train| < MIN_SEGMENT_SAMPLES (20):
          Wk = (α_global, β_global, γ_global, b0)   // global fallback
          strategy = "global_fallback"
      else:
          Fit: y = bk + αk·Likes_ll + βk·Comments_ll + γk·Shares_ll
          r2_train = R²(y_true, y_pred) on Dk_train
          Wk = (αk, βk, γk, bk)
          strategy = "segment"

  5.3 Store weights_df with columns: seed, Platform, Post Type, Segment,
      alpha, beta, gamma, intercept, n_train, r2_train, strategy

═══════════════════════════════════════════════════════════════════════════
STEP 6: COMPUTE SCORE (scoring.py)
═══════════════════════════════════════════════════════════════════════════
  For each row i in D_train and D_test:
    k = segment(Platform_i, Post_Type_i)
    (α, β, γ, b) = Wk   // from weights; if missing, use fallback
    Score_i = b + α·Likes_ll_i + β·Comments_ll_i + γ·Shares_ll_i

  Fallback (when segment k has no weights in weights_df — e.g., weights
  filtered by a different seed, or segment absent from train):
  Use mean(α), mean(β), mean(γ), mean(intercept) across all learned
  segment weights. This gives a reasonable default Score for every row
  without failing; the mean approximates "average" platform behavior.
  In normal runs, this is rare due to the split coverage check, but it
  keeps scoring robust.

═══════════════════════════════════════════════════════════════════════════
STEP 7: TRENDING LABELS (trending.py)
═══════════════════════════════════════════════════════════════════════════
  7.1 compute_segment_thresholds (on D_train only):
      For each segment k:
          For each row i in Dk_train:
              Eng_i = Likes_i + Comments_i + Shares_i
              reach_safe_i = max(Reach_i, 1)
              ER_proxy_i = Eng_i / reach_safe_i
          Tk = quantile(ER_proxy in Dk_train, TREND_PERCENTILE)  // e.g., 90th

  7.2 apply_trending_label (on D_train and D_test):
      For each row i:
          Eng_i = Likes_i + Comments_i + Shares_i
          ER_proxy_i = Eng_i / max(Reach_i, 1)
          k = segment(Platform_i, Post_Type_i)
          if k has threshold Tk:
              yi = 1 if ER_proxy_i >= Tk else 0
          else:
              T_fallback = median(all Tk)
              yi = 1 if ER_proxy_i >= T_fallback else 0

  Result: ~10% of rows in each segment have yi=1 (Trending).

═══════════════════════════════════════════════════════════════════════════
STEP 8: TRENDING CLASSIFIER (models.py)
═══════════════════════════════════════════════════════════════════════════
  Features for classifier:
    - Score (primary, from Step 6 — learned prediction of log(ER_proxy))
    - Eng_log (auxiliary: log(Likes_i + Comments_i + Shares_i + 1))
    - Platform, Post Type, Weekday Type, Time Periods, Age Group, Sentiment
      (OneHot encoded)

  8.1 Split D_train into train_sub (80%) and val_sub (20%)
      Stratify by y (Trending). If stratification fails, use random split.

  8.2 Train GradientBoostingClassifier on (X_train_sub, y_train_sub)
      - Numeric: Score (primary), Eng_log (auxiliary, passthrough)
      - Categorical: OneHotEncoder(handle_unknown="ignore")

  8.3 Threshold selection on val_sub (NOT on train_sub — prevents leakage):
      y_prob_val = model.predict_proba(X_val)[:, 1]
      best_F1 = -1
      for τ in 0.01, 0.02, ..., 0.99:
          y_pred = (y_prob_val >= τ)
          F1 = F1_score(y_val, y_pred)
          if F1 > best_F1: best_F1=F1; τ* = τ

  8.4 On D_test:
      y_prob_test = model.predict_proba(X_test)[:, 1]
      y_pred_test = 1 if y_prob_test >= τ* else 0

  Returns: (model, τ*)

═══════════════════════════════════════════════════════════════════════════
STEP 9: EVALUATION (evaluation.py)
═══════════════════════════════════════════════════════════════════════════
  Regression (Score vs log(ER_proxy) on test):
    y_true = compute_log_ER_proxy(test_df)   // from Likes, Comments, Shares, Reach
    y_pred = Score
    Compute: R², MAE, RMSE, Pearson
    Why: R² measures variance explained; MAE/RMSE measure prediction error;
    Pearson measures linear correlation. Together they answer: how well does
    our learned Score predict the true log(ER_proxy)?

  Classification (y_pred vs y_true Trending on test):
    Compute: Accuracy, Precision, Recall, F1, Confusion Matrix
    Why: With ~10% positives, accuracy alone is misleading (predicting all
    zeros gives 90%). Precision = when we predict Trending, how often right?
    Recall = of all true Trending, how many did we find? F1 balances both.

═══════════════════════════════════════════════════════════════════════════
STEP 10: REPORT (reporting.py)
═══════════════════════════════════════════════════════════════════════════
  Run Steps 5–9 for each seed. Aggregate:
    - metrics_regression: per-seed rows + mean row + std row
    - metrics_classification: per-seed rows + mean row + std row
    - weights: concatenate all weights from all seeds

  Save: metrics_regression.csv, metrics_classification.csv, weights.csv

═══════════════════════════════════════════════════════════════════════════
STEP 11: DIAGNOSTICS (diagnostics.py)
═══════════════════════════════════════════════════════════════════════════
  Check:
    - NaNs in Likes_ll, Comments_ll, Shares_ll, Reach_log, Segment
    - Trailing whitespace in categorical columns
    - Test segments missing from train
    - Segments with < MIN_SEGMENT_SAMPLES rows
    - Feature segments without learned weights

  Save: diagnostics_report.json (fatal_issues, warnings, ok)
═══════════════════════════════════════════════════════════════════════════
```

---

## 4. Repository Structure

```
PopWeight/
├── main.py                    # CLI entry point; runs commands
├── data/
│   └── social_media_engagement_data.xlsx
├── popweight/
│   ├── __init__.py
│   ├── config.py              # DATA_PATH, TRAIN_RATIO, TREND_PERCENTILE, etc.
│   ├── io_excel.py            # load_working_file()
│   ├── schema.py              # normalize_columns(), validate_required_columns()
│   ├── cleaning.py            # clean_core_columns()
│   ├── features.py            # add_transforms(), add_segment_key()
│   ├── splits.py              # make_splits()
│   ├── weights.py             # fit_segment_weights(), compute_log_ER_proxy()
│   ├── scoring.py             # apply_scores(), apply_baseline_scores()
│   ├── trending.py            # compute_segment_thresholds(), apply_trending_label()
│   ├── models.py              # train_trending_classifier(), predict_trending()
│   ├── evaluation.py          # regression_metrics(), evaluate_regression(),
│   │                          # classification_metrics()
│   ├── reporting.py           # aggregate_*, save_reports()
│   ├── storage.py             # init_db(), write_df(), read_df()
│   ├── diagnostics.py         # run_diagnostics(), save_diagnostics_report()
│   └── baseline.py            # run_baseline_evaluation(), compare_with_popweight()
├── tests/
│   └── test_trending.py       # Sanity checks for trending labels
└── outputs/                   # Generated: .csv, .sqlite, .json
```

---

## 5. Configuration

File: `popweight/config.py`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| DATA_PATH | "data/social_media_engagement_data.xlsx" | Input Excel path |
| SHEET_NAME | "Working File" | Sheet to load |
| SQLITE_PATH | "outputs/results.sqlite" | SQLite database path |
| RANDOM_SEEDS | [0,1,2,3,4,5,6,7,8,9] | Seeds for repeated splits |
| TRAIN_RATIO | 0.8 | 80% train, 20% test |
| TREND_PERCENTILE | 0.9 | Top 10% per segment = Trending |
| MIN_REACH | 1 | Drop rows with Reach < 1 |
| REMOVE_TOP_REACH_PERCENTILE | 0.995 | Drop top 0.5% by Reach (outliers) |
| SEGMENT_KEYS | ["Platform", "Post Type"] | Columns that define segment |
| MIN_SEGMENT_SAMPLES | 20 | Min rows for segment-specific weights; else global fallback |

**Why MIN_SEGMENT_SAMPLES = 20?** We fit 4 coefficients (intercept + α, β, γ) per segment. With fewer than ~20 rows, linear regression becomes unstable (high variance, overfitting). Rule of thumb: at least 5× the number of predictors.

*Example*: LinkedIn__Video has 15 rows in train → use global weights (from all train). Facebook__Image has 6,500 rows → fit segment-specific weights. The learned (α, β, γ) for Facebook__Image can differ from LinkedIn__Video because engagement patterns vary by platform and post type. With MIN=5, we'd fit on 15 rows (risky); with MIN=100, a small segment like LinkedIn__Video would never get its own weights.

---

## 6. Module-by-Module Explanation

### 6.1 io_excel.py

**Function**: `load_working_file(path, sheet)`

**What it does**:
1. Reads Excel file with `pandas.read_excel(engine="openpyxl")`
2. Trims whitespace from column names
3. For every string column, applies `str.strip()` to each value
4. Returns DataFrame (no validation yet)

**Called by**: `cmd_load` in main.py

---

### 6.2 schema.py

**Functions**: `normalize_columns(df)`, `validate_required_columns(df)`

**normalize_columns**:
- Replaces multiple consecutive spaces with single space in column names
- Strips leading/trailing whitespace from column names

**validate_required_columns**:
- Raises `ValueError` if any of these are missing: Platform, Post Type, Likes, Comments, Shares, Reach, Weekday Type, Time Periods, Age Group, Sentiment

**Called by**: `cmd_load` (after load_working_file)

---

### 6.3 cleaning.py

**Function**: `clean_core_columns(df, config)`

**What it does** (in order):
1. Convert Likes, Comments, Shares, Reach to numeric (errors→NaN)
2. Drop rows with NaN in any of these
3. Drop rows where Reach < MIN_REACH
4. Drop rows where Likes<0 or Comments<0 or Shares<0
5. Compute thr = quantile(Reach, REMOVE_TOP_REACH_PERCENTILE) and drop rows with Reach > thr
6. Return (df_clean, report_dict) where report has initial_rows, dropped_*, final_rows

**Called by**: `cmd_clean`

---

### 6.4 features.py

**add_transforms(df)**:
```
For Likes, Comments, Shares:
  x = col + 1
  out_col = log(log(x) + 1)
Reach_log = log(Reach + 1)
```
Why double-log: compresses heavy-tailed engagement; reduces outlier impact;
better linear fit for regression.

**add_segment_key(df, keys=["Platform","Post Type"])**:
- For each row i: Segment_i = Platform_i + "__" + Post_Type_i
- Strips whitespace from Platform, Post Type, Weekday Type, Time Periods, Age Group, Sentiment

**Called by**: `cmd_features`

---

### 6.5 splits.py

**make_splits(df, seeds, train_ratio)**:
- For each seed: call `train_test_split` with stratify=Segment
- Up to MAX_SPLIT_TRIES (20) attempts to get segment coverage
- If test has segment not in train: either retry (new random_state) or drop those test rows
- Returns list of `Split(seed, train_df, test_df)`

**Called by**: `cmd_split`, `cmd_fit_weights`, `cmd_report`, `cmd_train_classifier`, `cmd_baseline`, `cmd_diagnostics`

---

### 6.6 weights.py

**compute_log_ER_proxy(df)**:
- Eng = total engagement = Likes + Comments + Shares
- reach_safe = Reach.clip(lower=1)
- ER_proxy = (Eng/reach_safe).clip(lower=1e-10)
- return log(ER_proxy)

**fit_segment_weights(train_df, seed, min_segment_samples)**:
- Fit global linear regression on all train (fallback for small segments)
- For each segment: if n < min_segment_samples, use global weights; else fit
  segment-specific regression
- Target: log(ER_proxy). Features: Likes_ll, Comments_ll, Shares_ll
- Returns DataFrame with one row per segment

**Called by**: `cmd_fit_weights`, `cmd_report`

---

### 6.7 scoring.py

**apply_scores(df, weights_df)**:
- Merge df with weights by (Platform, Post Type) to look up (α, β, γ, intercept)
- **Fallback for missing weights**: If a row's (Platform, Post Type) has no
  matching weights (e.g., weights_df filtered by seed, or segment unseen in
  train), use mean(α), mean(β), mean(γ), mean(intercept) across all segment
  weights. This ensures every row gets a Score; the mean approximates average
  learned behavior across segments. In normal runs, this is rare due to the
  split coverage check, but it keeps scoring robust.
- Score = intercept + α*Likes_ll + β*Comments_ll + γ*Shares_ll

**apply_baseline_scores(df)**:
- Score_baseline = Likes_ll + Comments_ll + Shares_ll (equal weights, no intercept)

**Called by**: `cmd_score`, `cmd_train_classifier`, `cmd_report`, baseline.py

---

### 6.8 trending.py

**compute_segment_thresholds(train_df, percentile)**:
- Add Eng = Likes + Comments + Shares; ER_proxy = Eng/Reach (Reach clipped at 1)
- Group by Segment, compute quantile(ER_proxy, percentile)
- Returns DataFrame: Segment, threshold

**apply_trending_label(df, thresholds_df)**:
- Add ER_proxy
- Merge with thresholds by Segment
- If segment missing: use median(threshold)
- Trending = 1 if ER_proxy >= threshold else 0

**Called by**: `cmd_trend_label`, `cmd_train_classifier`, `cmd_report`, baseline.py

---

### 6.9 models.py

**train_trending_classifier(train_scored_labeled_df, val_ratio=0.2)**:
1. Add Eng_log = log(Likes+Comments+Shares+1) (auxiliary feature)
2. Extract features: Score (primary), Eng_log (auxiliary), Platform, Post Type,
   Weekday Type, Time Periods, Age Group, Sentiment
3. Split train into train_sub (80%) and val_sub (20%) stratified by Trending
4. Fit GradientBoostingClassifier on train_sub
5. Find threshold τ that maximizes F1 on val_sub (scan 0.01..0.99)
6. Return (model, τ)

**predict_trending(model, test_df, threshold)**:
- Add Eng_log, extract features, predict_proba, apply threshold
- Return (y_pred, y_prob, threshold)

**Called by**: `cmd_train_classifier`, `cmd_report`, baseline.py

---

### 6.10 evaluation.py

**evaluate_regression(test_scored_df, score_col="Score")**:
- y_true = compute_log_ER_proxy(test_scored_df)
- y_pred = test_scored_df[score_col]
- Returns {r2, mae, rmse, pearson, seed?}
- Why: R² = variance explained; MAE/RMSE = error magnitude; Pearson =
  linear correlation. Answers: how well does Score predict log(ER_proxy)?

**classification_metrics(y_true, y_pred)**:
- Returns {accuracy, precision, recall, f1, confusion_matrix}
- Why: With ~10% Trending, accuracy is misleading alone. Precision = of
  predicted Trending, fraction correct. Recall = of true Trending, fraction
  found. F1 balances precision and recall for imbalanced classification.

**Called by**: `cmd_train_classifier`, `cmd_report`, baseline.py

---

### 6.11 reporting.py

**aggregate_regression_metrics(metrics_list)**: per-seed rows + mean + std
**aggregate_classification_metrics(metrics_list)**: per-seed rows + mean + std (drops confusion_matrix)
**aggregate_weights(weights_list)**: concatenate all weights DataFrames
**save_reports(reg_df, cls_df, weights_df, output_dir)**: write CSVs to output_dir

**Called by**: `cmd_report`

---

### 6.12 storage.py

**init_db(sqlite_path)**: set database path, create parent dir
**write_df(table_name, df)**: replace table with df
**read_df(table_name, where?)**: read table, optional WHERE

**Tables used**: raw_working_file, clean_data, features_data, weights, regression_metrics, classification_metrics

**Called by**: All commands that persist or load data

---

### 6.13 diagnostics.py

**run_diagnostics(df_features, splits, weights_df, min_segment_samples)**:
- Check NaNs in transform columns
- Check trailing whitespace in categoricals
- Check test segments missing from train
- Check segments with < min_segment_samples rows
- Check feature segments missing from weights
- Returns {fatal_issues, warnings, checks, ok}

**save_diagnostics_report(report, path)**: write JSON

**Called by**: `cmd_diagnostics`

---

### 6.14 baseline.py

**Purpose**: Reference model with equal weights (α=β=γ=1). No segment-specific learning.

**run_baseline_evaluation(split, trend_percentile, seed)**:
1. Same trending labels as PopWeight (ER_proxy thresholds)
2. Score_baseline = Likes_ll + Comments_ll + Shares_ll (equal weights, no intercept)
3. Evaluate regression: Score_baseline vs log(ER_proxy)
4. Train same classifier with Score_baseline instead of Score
5. Evaluate classification
6. Return {baseline_regression, baseline_classification, seed}

**Called by**: `cmd_baseline`

---

## 7. CLI Commands

Run: `python main.py <command> [command ...]` or `python main.py all`

| Command | Dependencies | What It Does |
|---------|--------------|--------------|
| load | — | Load Excel, normalize, validate, save raw |
| clean | load | Clean data, save clean_data |
| features | clean | Add transforms + segment, save features_data |
| split | features | Create 10 train/test splits |
| fit_weights | split | Learn weights (α, β, γ) specific to each segment per seed, save |
| score | fit_weights | Compute Score for seed 0 (verification) |
| trend_label | split | Build Trending labels for seed 0 |
| train_classifier | fit_weights | Train classifier to predict Trending (Score + features), eval seed 0 |
| baseline | split | Run baseline (equal-weight score + classifier) |
| report | features | Full pipeline per seed, aggregate metrics and weights, save CSVs |
| diagnostics | fit_weights | Run health checks, save diagnostics_report.json |

**Dependencies**: If you run `report`, it will auto-run `features` first (and thus load, clean). `report` does NOT use pre-computed weights from `fit_weights`; it recomputes everything internally.

**Usage tips** (for newcomers):
- **fit_weights**: Run after `split`. Learns (α, β, γ) per segment; saves to DB. Use when you want to inspect or reuse weights before running the full classifier.
- **train_classifier**: Run after `fit_weights`. Trains Trending classifier on Score + Eng_log + categoricals, evaluates on test (seed 0). Quick way to check classifier performance on a single split.
- **report**: Run after `features` (skips fit_weights). Runs the full pipeline for all 10 seeds, aggregates metrics and weights, writes CSVs. Use for final evaluation and reproducibility.

---

## 8. Output Files

### outputs/results.sqlite

SQLite database with tables:
- **raw_working_file**: Loaded Excel (after normalize)
- **clean_data**: After cleaning
- **features_data**: After transforms + segment key
- **weights**: Learned weights (when fit_weights or report runs)
- **regression_metrics**: Per-seed regression metrics (when report runs)
- **classification_metrics**: Per-seed classification metrics (when report runs)

### outputs/metrics_regression.csv

Columns: r2, mae, rmse, pearson, seed  
Rows: one per seed (0..9) + "mean" + "std"

### outputs/metrics_classification.csv

Columns: accuracy, precision, recall, f1, seed  
Rows: one per seed + mean + std

### outputs/weights.csv

Columns: seed, Platform, Post Type, Segment, alpha, beta, gamma, intercept, n_train, r2_train, strategy  
One row per segment per seed.

### outputs/diagnostics_report.json

```json
{
  "fatal_issues": [],
  "warnings": [],
  "checks": { "nan_after_transforms": "none", ... },
  "ok": true
}
```

---

## 9. Data Flow Diagram

```
Excel (Working File)
    │
    ▼
load ──► raw_working_file
    │
    ▼
clean ──► clean_data
    │
    ▼
features ──► features_data (Likes_ll, Comments_ll, Shares_ll, Reach_log, Segment, ...)
    │
    ▼
split ──► [Split(seed, train_df, test_df)] × 10
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
fit_weights ──► weights_df         baseline: Score_baseline = L_ll + C_ll + S_ll
    │                                  │
    ▼                                  │
apply_scores ──► train_scored, test_scored  (Score column)
    │                                  │
    ▼                                  │
compute_segment_thresholds ──► thresholds_df (Segment, threshold)
    │                                  │
    ▼                                  ▼
apply_trending_label ──► train_labeled, test_labeled (Trending column)
    │                                  │
    ▼                                  ▼
train_trending_classifier ──► (model, threshold)
    │
    ▼
predict_trending ──► y_pred, y_prob
    │
    ▼
evaluate_regression, classification_metrics
    │
    ▼
aggregate + save_reports ──► metrics_*.csv, weights.csv
```

---

## 10. Important Design Decisions

### A) Why log(ER_proxy) instead of Reach?

- **Trending** is defined by ER_proxy (engagement rate) within each segment
- Aligning the regression target (log(ER_proxy)) with the label (Trending by ER_proxy) improves classifier performance
- Prevents degenerate solutions where the model collapses to predicting
  the majority class

### B) Why TREND_PERCENTILE (e.g., 90%)?

- **Trending** = top performers within each segment. 90th percentile means ~10% of
  posts per (Platform, Post Type) are labeled "Trending".
- Per-segment: a post competes only with similar content (e.g., Facebook Image
  vs Facebook Image), not across platforms.
- Configurable: TREND_PERCENTILE in config lets you tune (e.g., 0.95 for top 5%).

### C) Leakage Prevention

- **Trending thresholds**: Computed on train only; never use test data
- **Classifier threshold**: Chosen on a validation split (20% of train), not on the 80% used to fit the classifier
- **Test set**: Never used for any threshold or weight learning

### D) Small Segment Fallback

- If a segment has fewer than MIN_SEGMENT_SAMPLES (20) rows, use **global weights** (one model on all train data)
- No platform-only fallback; only global
- **If too low (e.g., 5)**: segment-specific weights would overfit; R² would be unreliable. **If too high (e.g., 100)**: rare segments would always fall back, losing platform/post-type nuance.

### E) Do Not Use "Engagement Rate"

- The raw dataset may have an "Engagement Rate" column
- Do **not** use it in training or evaluation (risk of leakage)
- We compute ER_proxy ourselves from Likes, Comments, Shares, Reach

### F) Classifier Features: Score (Primary) vs Eng_log (Auxiliary)

- **Score** is the primary feature — the learned prediction of log(ER_proxy) from segment-specific weights
- **Eng_log** = log(Likes + Comments + Shares + 1) is an auxiliary feature
- Eng_log can help capture raw engagement magnitude, while Score carries the main PopWeight signal

### G) Baseline as Reference

- **Baseline** = equal weights: Score_baseline = Likes_ll + Comments_ll + Shares_ll (α=β=γ=1, no intercept). Treats Likes, Comments, Shares equally; ignores segment.
- **PopWeight** = learned (α, β, γ) per (Platform, Post Type); intercept per segment.

**Why is the baseline an important reference?** It answers: *"Does learning segment-specific weights actually help?"* The baseline is the simplest plausible predictor using the same inputs. It sets a floor: PopWeight must beat it to justify the added complexity. In practice, PopWeight typically achieves higher R² and F1 because it learns (e.g.) that Comments weigh more for LinkedIn Links than for Facebook Images. If PopWeight barely beats baseline, the extra modeling may not be worth it.

### H) Reproducibility

- All splits use explicit random seeds
- GradientBoostingClassifier uses random_state=0
- Results are reproducible across runs

---

## Acceptance Criteria

The implementation is complete when:

1. `python main.py all` produces `outputs/results.sqlite` with required tables
2. `outputs/weights.csv` exists with (α, β, γ) per Platform × Post Type per seed
3. `outputs/metrics_regression.csv` and `outputs/metrics_classification.csv` exist with per-seed + mean/std
4. Baseline results are produced and comparable
5. Diagnostics report indicates no fatal issues (ok: true)
