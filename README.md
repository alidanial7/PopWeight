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
- **Correlation Analysis:** Built-in correlation analysis between engagement metrics (Likes, Comments, Shares) and Reach.
- **Optimized Storage:** Preprocessing automatically filters to only essential columns, reducing database size and improving performance.

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

### Code Quality & Formatting

This project uses **[Ruff](https://docs.astral.sh/ruff/)** for linting and code formatting, integrated with **pre-commit** hooks to ensure consistent code quality.

#### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

#### Usage

- **Automatic**: Pre-commit hooks run automatically on every `git commit`, checking and fixing code issues
- **Manual linting**: Run `ruff check .` to check for linting issues
- **Manual formatting**: Run `ruff format .` to format code
- **Check all files**: Run `pre-commit run --all-files` to check the entire codebase

#### Configuration

Ruff configuration is defined in `pyproject.toml` with the following settings:

- Line length: 88 characters
- Enabled rule sets: pycodestyle, pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade
- Import sorting configured for the `utils` package

#### Common Ruff Errors to Avoid

- **E501**: Line too long (>88 characters) - Break long lines into multiple lines
- **B007**: Unused loop variable - Use `_` for unused loop variables
- **F401**: Unused imports - Remove unused imports
- Always run `ruff check .` before committing to catch these issues early

## 📦 Project Structure

```
PopWeight/
├── main.py                 # Main interactive menu (single entry point)
├── workflows/              # All workflow modules (unified interface)
│   ├── data_preparation.py # Data generation, splitting, import, preprocessing
│   ├── training.py         # Model training workflow
│   ├── validation.py       # Model validation workflow
│   ├── correlation.py      # Correlation analysis workflows
│   └── diagnostics.py      # Diagnostic tools
├── analysis/               # Analysis modules
│   ├── models.py           # Model training and weight extraction
│   ├── validation.py       # Validation and metrics
│   ├── visualizations.py   # Plotting and charts
│   ├── insights.py         # Statistical insights
│   ├── trend_detection.py  # Trending post detection
│   └── correlation.py      # Correlation analysis
├── utils/                  # Utility functions
│   ├── data_loader.py      # Excel file loading utilities
│   ├── data_loading.py    # Data loading with progress
│   ├── database.py        # SQLite database operations
│   ├── preprocessing.py   # Data preprocessing functions
│   └── model_storage.py    # Model saving and loading
├── data/                   # Data files directory
│   ├── train.xlsx          # Training dataset (Excel)
│   ├── test.xlsx           # Test dataset (Excel)
│   ├── train.db            # Training SQLite database
│   │   ├── train_data_raw          # Raw training data
│   │   └── train_data_processed   # Preprocessed training data
│   └── test.db             # Test SQLite database
│       ├── test_data_raw           # Raw test data
│       └── test_data_processed     # Preprocessed test data
├── outputs/                # Generated outputs
│   ├── training_results.db # Saved training results
│   ├── gamma_heatmap.png   # Weight heatmap visualization
│   ├── weights_facet_grid.png # Facet grid visualization
│   ├── prediction_vs_actual.png # Validation visualization
│   └── confusion_matrix.png # Classification metrics
├── requirements.txt        # Python dependencies
└── pyproject.toml          # Project configuration (Ruff, etc.)
```

**Note:** All operations are accessible through `main.py` interactive menu or by
importing from the `workflows` package. Standalone script files have been removed
in favor of the unified workflow system.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Menu

The easiest way to use the system is through the interactive menu:

```bash
python main.py
```

This will display a menu with all available operations. Follow the menu options
in order for a complete workflow.

### 3. Complete Workflow

**Using the Interactive Menu (Recommended):**

1. Run `python main.py`
2. Follow the menu options in order:
   - **Option 1**: Generate Data (interactive prompt for sample count)
   - **Option 2**: Split Data (interactive prompt for train percentage)
   - **Option 3**: Import Train
   - **Option 4**: Import Test
   - **Option 5**: Preprocess Train
   - **Option 6**: Preprocess Test
   - **Option 7**: Train (to learn weights)
   - **Option 8**: Test (to validate)

**Using Workflows Programmatically:**

```python
from workflows import (
    generate_data,
    split_data,
    import_train_data,
    import_test_data,
    preprocess_train,
    preprocess_test,
    train_model,
    test_model,
)

# Complete pipeline
generate_data()  # Interactive prompt for sample count
split_data()  # Interactive prompt for train percentage
import_train_data()
import_test_data()
preprocess_train()
preprocess_test()
train_model()
test_model()
```

### 4. Data Import and Preprocessing Details

**Import Workflows:**

- Load data from Excel files (`data/train.xlsx` and `data/test.xlsx`)
- Save raw data to separate SQLite databases (`data/train.db` and `data/test.db`)
- Create tables: `train_data_raw` and `test_data_raw`
- Display progress bars and detailed logging
- Create databases automatically if they don't exist

**Preprocessing Workflows:**

- Read from raw data tables (`train_data_raw` and `test_data_raw`)
- Apply comprehensive preprocessing transformations
- Filter to essential columns only
- Save processed data to separate tables (`train_data_processed` and `test_data_processed`)
- Display detailed progress and transformation summaries

## 📝 Workflow Usage Guide

All operations are available through the unified workflow system. You can access
them either through the interactive menu or by importing them programmatically.

### Using Workflows via Menu (Recommended)

The easiest and recommended approach is to use the interactive menu:

```bash
python main.py
```

The menu provides:

- **Organized sections**: Data Preparation, Analysis, Utilities
- **Clear descriptions**: Each option explains what it does
- **Interactive prompts**: For parameters like sample count and split percentage
- **Progress indicators**: Visual feedback for long-running operations
- **Error handling**: Clear error messages and recovery suggestions

### Using Workflows Programmatically

All workflows can be imported and used in your own Python scripts:

```python
from workflows import (
    generate_data,
    split_data,
    import_train_data,
    import_test_data,
    preprocess_train,
    preprocess_test,
    train_model,
    test_model,
    correlation_likes_reach,
    run_diagnostics,
)

# Example: Complete data preparation pipeline
generate_data()  # Interactive prompt for sample count
split_data()  # Interactive prompt for train percentage
import_train_data()
import_test_data()
preprocess_train()
preprocess_test()
```

**Note:** Some workflows (like `generate_data()` and `split_data()`) will prompt
interactively for parameters if called without arguments. You can also provide
parameters directly for programmatic use.

### Workflow Features

**Data Import Workflows (`import_train_data`, `import_test_data`):**

- Read from Excel files (`data/train.xlsx`, `data/test.xlsx`)
- Save to SQLite databases (`data/train.db`, `data/test.db`)
- Create tables: `train_data_raw`, `test_data_raw`
- Progress bars with `tqdm`
- Detailed logging and error handling
- Automatic directory creation

**Preprocessing Workflows (`preprocess_train`, `preprocess_test`):**

- Read from raw data tables
- Apply comprehensive preprocessing transformations
- Filter to essential columns only
- Save to processed tables
- Detailed progress indicators

**Preprocessing Steps Applied:**

1. **Missing Value Handling**: Fills numerical columns with median, categorical with 'None'
2. **Log-Log Normalization**: Applies `log(log(x + 1) + 1)` to Likes, Comments, Shares
3. **Temporal Feature Extraction**: Extracts Hour_of_day, Day_of_week, Is_Weekend
4. **One-Hot Encoding**: Encodes Platform, Post Type, and Sentiment
5. **Feature Scaling**: Applies StandardScaler to Audience Age
6. **Target Transformation**: Applies log transformation to Reach
7. **Column Filtering**: Automatically filters to only essential columns

**Essential Columns Kept:**

- Grouping columns: `Platform`, `Post Type`
- Feature columns: `Likes_log_log`, `Comments_log_log`, `Shares_log_log`
- Target columns: `Reach_log`, `Engagement_Rate`
- Optional features: `Engagement_Density`
- One-hot encoded columns: `Platform_*`, `Post_Type_*`, `Sentiment_*`

**Note:** The preprocessing pipeline ensures consistency between training and test data transformations.

## 🗄️ Database Structure

The project uses **4 separate tables** to maintain raw and processed data:

### Training Database (`data/train.db`)

1. **`train_data_raw`** - Raw training data imported from Excel

   - Contains original columns from `data/train.xlsx`
   - No preprocessing applied
   - Created by: `workflows.data_preparation.import_train_data()` workflow
   - Accessible via menu option 3 or programmatically

2. **`train_data_processed`** - Preprocessed training data
   - Contains only essential columns needed for analysis
   - All preprocessing transformations applied
   - Unused columns automatically filtered out for efficiency
   - Created by: `workflows.data_preparation.preprocess_train()` workflow
   - Accessible via menu option 5 or programmatically

### Test Database (`data/test.db`)

3. **`test_data_raw`** - Raw test data imported from Excel

   - Contains original columns from `data/test.xlsx`
   - No preprocessing applied
   - Created by: `workflows.data_preparation.import_test_data()` workflow
   - Accessible via menu option 4 or programmatically

4. **`test_data_processed`** - Preprocessed test data
   - Contains only essential columns needed for analysis
   - All preprocessing transformations applied
   - Unused columns automatically filtered out for efficiency
   - Created by: `workflows.data_preparation.preprocess_test()` workflow
   - Accessible via menu option 6 or programmatically

## 🛠️ Utilities and Analysis Modules

For detailed documentation on utility functions and analysis modules, see:

- **[Utils Documentation](utils/README.md)** - Database operations, data loading, preprocessing, and model storage utilities
- **[Analysis Documentation](analysis/README.md)** - Model training, validation, visualization, insights, trend detection, and correlation analysis

## ⌨️ Interactive Menu Interface

The project provides an interactive menu-driven interface through `main.py` that
consolidates all operations in one place. All workflows are accessible through
the main menu, making it easy to perform data preparation, analysis, and diagnostics.

### Running the Main Menu

```bash
python main.py
```

This will display an interactive menu with all available operations organized
into sections:

**📊 Data Preparation:**

- Generate Data - Create synthetic dataset
- Split Data - Split base data into train/test
- Import Train - Import training data to database
- Import Test - Import test data to database
- Preprocess Train - Preprocess training data
- Preprocess Test - Preprocess test data

**🔬 Analysis:**

- Train - Learn weights from training data
- Test - Validate weights on test data
- Correlation - Likes vs Reach
- Correlation - Comments vs Reach
- Correlation - Shares vs Reach

**🔍 Utilities:**

- Diagnostics - Run validation diagnostics

### Workflow Modules

All workflows are implemented as modules in the `workflows/` package and can
also be imported and used programmatically:

```python
from workflows import (
    generate_data,
    split_data,
    import_train_data,
    import_test_data,
    preprocess_train,
    preprocess_test,
    train_model,
    test_model,
    run_diagnostics,
)

# Use workflows programmatically
generate_data(n_samples=10000)
import_train_data()
preprocess_train()
train_model()
```

## 🔄 Workflows Documentation

For complete workflows documentation, see **[Workflows README](workflows/README.md)**.

The `workflows/` package contains modular workflow functions for all major
operations. Each workflow is self-contained and can be used independently.

### Quick Reference

**Data Preparation Workflows:**

- `generate_data()` - Generate synthetic dataset (interactive prompt)
- `split_data()` - Split data into train/test (interactive prompt)
- `import_train_data()` - Import training data to database
- `import_test_data()` - Import test data to database
- `preprocess_train()` - Preprocess training data
- `preprocess_test()` - Preprocess test data

**Analysis Workflows:**

- `train_model()` - Train models and learn weights
- `test_model()` - Validate models on test data
- `correlation_likes_reach()` - Analyze Likes vs Reach correlation
- `correlation_comments_reach()` - Analyze Comments vs Reach correlation
- `correlation_shares_reach()` - Analyze Shares vs Reach correlation
- `run_diagnostics()` - Run diagnostic checks

For detailed documentation with parameters, examples, and usage, see
**[Workflows README](workflows/README.md)**.

## 🔍 Data Exploration & Analysis

Use `main.py` to explore the loaded data and perform analysis:

```bash
python main.py
```

### Main Menu Options

The script provides an interactive menu with the following options:

1. **Train** - Learn weights from training data

   - Performs cross-sectional analysis
   - Trains Linear Regression and Random Forest models
   - Generates visualizations (heatmaps, facet grids)
   - Identifies trending posts
   - Saves training results

2. **Test** - Validate weights on test data

   - Loads learned weights from training
   - Validates on test set
   - Generates prediction vs actual visualizations
   - Creates confusion matrices
   - Provides validation metrics

3. **Correlation - Likes vs Reach**

   - Calculates Pearson correlation coefficient
   - Displays correlation strength and direction
   - Shows statistical summary for both metrics

4. **Correlation - Comments vs Reach**

   - Calculates Pearson correlation coefficient
   - Displays correlation strength and direction
   - Shows statistical summary for both metrics

5. **Correlation - Shares vs Reach**
   - Calculates Pearson correlation coefficient
   - Displays correlation strength and direction
   - Shows statistical summary for both metrics

**Correlation Analysis Features:**

- Loads data from training database
- Calculates correlation coefficient with interpretation
- Provides statistical summaries (mean, std, min, max)
- Categorizes correlation strength (negligible, weak, moderate, strong, very strong)
- Indicates direction (positive/negative)

**Example Output:**

```
📊 CORRELATION ANALYSIS: Likes vs Reach
================================================================================

📖 Loading train data...
✓ Loaded 999 rows × 18 columns

🔍 Calculating correlation...

--------------------------------------------------------------------------------
CORRELATION RESULTS
--------------------------------------------------------------------------------
Column 1: Likes
Column 2: Reach
Correlation Coefficient: 0.8234
Interpretation: very strong positive correlation

--------------------------------------------------------------------------------
STATISTICAL SUMMARY
--------------------------------------------------------------------------------
Likes:
  Mean: 1250.45
  Std:  2340.12
  Min:  0.00
  Max:  15000.00

Reach:
  Mean: 8500.23
  Std:  12000.45
  Min:  100.00
  Max:  50000.00
```
