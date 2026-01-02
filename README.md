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
├── data/                    # Data files directory
│   ├── train.xlsx          # Training dataset (Excel)
│   ├── test.xlsx           # Test dataset (Excel)
│   ├── train.db            # Training SQLite database
│   │   ├── train_data_raw          # Raw training data
│   │   └── train_data_processed   # Preprocessed training data
│   └── test.db             # Test SQLite database
│       ├── test_data_raw           # Raw test data
│       └── test_data_processed     # Preprocessed test data
├── commands/               # Command-line interface modules
│   ├── import_excel.py     # Generic Excel import command
│   ├── import_train.py     # Training data import command
│   ├── import_test.py      # Test data import command
│   ├── preprocess_train.py # Training data preprocessing command
│   └── preprocess_test.py  # Test data preprocessing command
├── utils/                  # Utility functions
│   ├── data_loader.py      # Excel file loading utilities
│   ├── database.py         # SQLite database operations
│   └── preprocessing.py    # Data preprocessing functions
├── import_train.py         # Training data import script
├── import_test.py          # Test data import script
├── preprocess_train.py      # Training data preprocessing script
├── preprocess_test.py       # Test data preprocessing script
├── main.py                 # Main exploration script
├── requirements.txt        # Python dependencies
└── pyproject.toml          # Project configuration (Ruff, etc.)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Import Raw Data to Database

Import training and test data from Excel files into SQLite databases:

```bash
# Import training data (creates train_data_raw table)
python import_train.py

# Import test data (creates test_data_raw table)
python import_test.py
```

Both scripts will:
- Load data from Excel files (`data/train.xlsx` and `data/test.xlsx`)
- Save raw data to separate SQLite databases (`data/train.db` and `data/test.db`)
- Create tables: `train_data_raw` and `test_data_raw`
- Display progress bars and detailed logging
- Create databases automatically if they don't exist

### 3. Preprocess Data

Apply data preprocessing transformations to prepare data for modeling:

```bash
# Preprocess training data (creates train_data_processed table)
python preprocess_train.py

# Preprocess test data (creates test_data_processed table)
python preprocess_test.py
```

The preprocessing scripts will:
- Read from raw data tables (`train_data_raw` and `test_data_raw`)
- Apply comprehensive preprocessing transformations
- Save processed data to separate tables (`train_data_processed` and `test_data_processed`)
- Display detailed progress and transformation summaries

## 📝 Data Import Scripts

### Import Training Data (`import_train.py`)

Simple script to import raw training data from Excel to SQLite.

**Usage:**
```bash
python import_train.py
```

**Features:**
- Reads from: `data/train.xlsx`
- Saves to: `data/train.db` (table: `train_data_raw`)
- Beautiful progress bars with `tqdm`
- Detailed logging and error handling
- Automatic directory creation

**Output:**
- Progress indicators for reading Excel and writing to database
- Summary of loaded data (rows, columns, column preview)
- Success confirmation with database location

### Import Test Data (`import_test.py`)

Simple script to import raw test data from Excel to SQLite.

**Usage:**
```bash
python import_test.py
```

**Features:**
- Reads from: `data/test.xlsx`
- Saves to: `data/test.db` (table: `test_data_raw`)
- Same beautiful UI and progress tracking as training script
- Separate database to keep train/test data isolated

## 🔧 Data Preprocessing Scripts

### Preprocess Training Data (`preprocess_train.py`)

Applies comprehensive data preprocessing transformations to training data.

**Usage:**
```bash
python preprocess_train.py
```

**Features:**
- Reads from: `data/train.db` (table: `train_data_raw`)
- Saves to: `data/train.db` (table: `train_data_processed`)
- Applies all preprocessing transformations (see below)
- Beautiful progress bars and step-by-step indicators
- Detailed transformation summaries

**Preprocessing Steps:**
1. **Missing Value Handling**: Fills numerical columns with median, categorical with 'None'
2. **Log-Log Normalization**: Applies `log(log(x + 1) + 1)` to Likes, Comments, Shares
3. **Temporal Feature Extraction**: Extracts Hour_of_day, Day_of_week, Is_Weekend from Post Timestamp
4. **One-Hot Encoding**: Encodes Platform, Post Type, and Sentiment
5. **Feature Scaling**: Applies StandardScaler to Audience Age
6. **Target Transformation**: Applies log transformation to Reach (target variable)

**Output:**
- Progress indicators for each preprocessing step
- Summary of original vs processed data (rows, columns)
- Number of new columns added
- Success confirmation with database location

### Preprocess Test Data (`preprocess_test.py`)

Applies the same preprocessing transformations to test data.

**Usage:**
```bash
python preprocess_test.py
```

**Features:**
- Reads from: `data/test.db` (table: `test_data_raw`)
- Saves to: `data/test.db` (table: `test_data_processed`)
- Same preprocessing pipeline as training data
- Beautiful UI and progress tracking

**Note:** The preprocessing pipeline ensures consistency between training and test data transformations.

## 🗄️ Database Structure

The project uses **4 separate tables** to maintain raw and processed data:

### Training Database (`data/train.db`)

1. **`train_data_raw`** - Raw training data imported from Excel
   - Contains original columns from `data/train.xlsx`
   - No preprocessing applied
   - Created by: `import_train.py`

2. **`train_data_processed`** - Preprocessed training data
   - Contains all original columns plus new features
   - All preprocessing transformations applied
   - Created by: `preprocess_train.py`

### Test Database (`data/test.db`)

3. **`test_data_raw`** - Raw test data imported from Excel
   - Contains original columns from `data/test.xlsx`
   - No preprocessing applied
   - Created by: `import_test.py`

4. **`test_data_processed`** - Preprocessed test data
   - Contains all original columns plus new features
   - All preprocessing transformations applied
   - Created by: `preprocess_test.py`

## 🛠️ Database Utilities

The project includes utilities for working with SQLite databases in `utils/database.py`:

### Functions

#### `save_to_sqlite(df, db_path, table_name, if_exists)`
Save a pandas DataFrame to a SQLite database.

**Parameters:**
- `df`: pandas DataFrame to save
- `db_path`: Path to SQLite database file
- `table_name`: Name of the table (default: `"social_media_data"`)
- `if_exists`: Behavior if table exists - `"fail"`, `"replace"`, or `"append"` (default: `"replace"`)

**Example:**
```python
from utils import save_to_sqlite
import pandas as pd

df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
save_to_sqlite(df, "data.db", table_name="my_table")
```

#### `read_from_sqlite(db_path, table_name, query)`
Read data from a SQLite database into a pandas DataFrame.

**Parameters:**
- `db_path`: Path to SQLite database file
- `table_name`: Name of the table to read (default: `"social_media_data"`)
- `query`: Optional SQL query (if provided, `table_name` is ignored)

**Example:**
```python
from utils import read_from_sqlite

# Read entire table
df = read_from_sqlite("data/train.db", table_name="train_data_raw")

# Read processed data
df = read_from_sqlite("data/train.db", table_name="train_data_processed")

# Read with custom query
df = read_from_sqlite(
    "data/train.db",
    query="SELECT * FROM train_data_raw WHERE Likes > 100"
)
```

#### `list_tables(db_path)`
List all tables in a SQLite database.

**Example:**
```python
from utils import list_tables

tables = list_tables("data/train.db")
print(tables)  # ['train_data_raw', 'train_data_processed']
```

## 🔄 Data Preprocessing Pipeline

The preprocessing pipeline in `utils/preprocessing.py` applies the following transformations:

### 1. Missing Value Handling
- **Numerical columns**: Filled with median value
- **Categorical columns**: Filled with 'None'

### 2. Log-Log Normalization
Applies the formula `x' = log(log(x + 1) + 1)` to handle skewness in:
- `Likes`
- `Comments`
- `Shares`

Creates new columns: `Likes_log_log`, `Comments_log_log`, `Shares_log_log`

### 3. Temporal Feature Extraction
Extracts features from `Post Timestamp`:
- `Hour_of_day`: Hour of the day (0-23)
- `Day_of_week`: Day of the week (0=Monday, 6=Sunday)
- `Is_Weekend`: Boolean (1 if Saturday/Sunday, 0 otherwise)

### 4. One-Hot Encoding
Encodes categorical variables:
- `Platform`: Creates columns like `Platform_Facebook`, `Platform_Instagram`, etc.
- `Post Type`: Creates columns like `Post_Type_Image`, `Post_Type_Video`, etc.
- `Sentiment`: Creates columns like `Sentiment_Positive`, `Sentiment_Neutral`, etc.

### 5. Feature Scaling
- Applies `StandardScaler` to `Audience Age` (normalizes to mean=0, std=1)

### 6. Target Variable Transformation
- Applies log transformation to `Reach`: `Reach_log = log(Reach + 1)`

**Note:** Original columns are preserved alongside new features, allowing you to reference both raw and processed data.

## ⌨️ Command-Line Interface

The project includes command-line commands for data import operations:

### Available Commands

#### `import-train`
Import raw training data from Excel to SQLite database.

**Usage:**
```bash
# Using default paths (data/train.xlsx -> data/train.db)
python -m commands.import_train

# With custom paths
python -m commands.import_train --excel data/train.xlsx --db data/train.db

# With custom table name
python -m commands.import_train --table my_train_table

# Append to existing table
python -m commands.import_train --if-exists append
```

**Options:**
- `--excel`: Path to Excel file (default: `data/train.xlsx`)
- `--db`: Path to database file (default: `data/train.db`)
- `--table`: Table name (default: `train_data_raw`)
- `--sheet`: Excel sheet name/index (default: first sheet)
- `--if-exists`: Behavior if table exists - `fail`, `replace`, or `append` (default: `replace`)

#### `import-test`
Import raw test data from Excel to SQLite database.

**Usage:**
```bash
# Using default paths (data/test.xlsx -> data/test.db)
python -m commands.import_test

# With custom paths
python -m commands.import_test --excel data/test.xlsx --db data/test.db
```

**Options:** Same as `import-train` command (default table: `test_data_raw`)

#### `preprocess-train`
Preprocess training data from SQLite database.

**Usage:**
```bash
# Using default paths
python -m commands.preprocess_train

# With custom paths
python -m commands.preprocess_train --db data/train.db --table train_data_raw
```

**Options:**
- `--db`: Path to database file (default: `data/train.db`)
- `--table`: Input table name (default: `train_data_raw`)
- `--output-table`: Output table name (default: `train_data_processed`)
- `--if-exists`: Behavior if output table exists (default: `replace`)

#### `preprocess-test`
Preprocess test data from SQLite database.

**Usage:**
```bash
# Using default paths
python -m commands.preprocess_test

# With custom paths
python -m commands.preprocess_test --db data/test.db --table test_data_raw
```

**Options:** Same as `preprocess-train` command (defaults: `test_data_raw` → `test_data_processed`)

#### `import-excel`
Generic Excel import command (requires all parameters).

**Usage:**
```bash
python -m commands.import_excel --excel data/file.xlsx --db data/database.db
```

**Options:**
- `--excel`: Path to Excel file (required)
- `--db`: Path to database file (required)
- `--table`: Table name (default: `social_media_data`)
- `--sheet`: Excel sheet name/index (default: first sheet)
- `--if-exists`: Behavior if table exists (default: `replace`)

### After Package Installation

If you install the package, these commands are available directly:

```bash
import-train --excel data/train.xlsx --db data/train.db
import-test --excel data/test.xlsx --db data/test.db
import-excel --excel data/file.xlsx --db data/database.db
```

## 🔍 Data Exploration

Use `main.py` to explore the loaded data:

```bash
python main.py
```

This script will:
- Load data from Excel
- Display column information and data types
- Save data to SQLite database
- List tables in the database
- Read data back from database
- Show example queries
