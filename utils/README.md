# Utils Documentation

The `utils/` package provides utility functions for data loading, database operations,
preprocessing, and model storage.

## 📦 Available Modules

### Database Utilities (`database.py`)

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

### Data Loading Utilities (`data_loader.py`)

#### `load_social_media_data(file_path, sheet_name)`

Load social media engagement data from Excel file.

**Parameters:**
- `file_path` (str, optional): Path to the Excel file. If None, uses default:
  `data/social_media_engagement_data.xlsx`
- `sheet_name` (str, optional): Name or index of the sheet to read. If None,
  reads the first sheet.

**Returns:**
- `pd.DataFrame`: DataFrame containing the social media engagement data.

**Example:**
```python
from utils import load_social_media_data

# Use default path
df = load_social_media_data()

# Specify custom path
df = load_social_media_data(file_path="custom/path/data.xlsx")

# Specify sheet name
df = load_social_media_data(sheet_name="Sheet1")
```

### Data Loading with Progress (`data_loading.py`)

#### `load_processed_data(project_root, data_source, data_type)`

Load processed data from SQLite database with progress indicator.

**Parameters:**
- `project_root` (Path): Root directory of the project
- `data_source` (str): Data source name: "train" or "test"
- `data_type` (str, default "processed"): Type of data to load: "processed" or "raw"

**Returns:**
- `tuple`: Tuple containing (DataFrame, None) on success

**Example:**
```python
from pathlib import Path
from utils.data_loading import load_processed_data

project_root = Path(__file__).parent.parent
df, _ = load_processed_data(project_root, "train", "processed")
```

### Preprocessing Utilities (`preprocessing.py`)

#### `preprocess_train_data(df)`

Preprocess training data with all transformations.

**Parameters:**
- `df` (pd.DataFrame): Raw training DataFrame

**Returns:**
- `pd.DataFrame`: Preprocessed DataFrame

**Preprocessing Steps:**
1. Missing value handling (median for numerical, 'None' for categorical)
2. Log-log normalization: `log(log(x + 1) + 1)` for Likes, Comments, Shares
3. Temporal feature extraction from Post Timestamp
4. One-hot encoding for Platform, Post Type, Sentiment
5. Feature scaling for Audience Age
6. Target transformation: `log(Reach + 1)`

**Example:**
```python
from utils import preprocess_train_data

df_processed = preprocess_train_data(df_raw)
```

#### `preprocess_test_data(df, scaler)`

Preprocess test data using the same transformations as training.

**Parameters:**
- `df` (pd.DataFrame): Raw test DataFrame
- `scaler` (StandardScaler, optional): Fitted scaler from training data

**Returns:**
- `pd.DataFrame`: Preprocessed DataFrame

**Example:**
```python
from utils import preprocess_test_data

df_processed = preprocess_test_data(df_raw, scaler=scaler)
```

#### `filter_essential_columns(df)`

Filter DataFrame to only include columns needed for analysis.

**Keeps:**
- Grouping columns: `Platform`, `Post Type`
- Feature columns: `Likes_log_log`, `Comments_log_log`, `Shares_log_log`
- Target columns: `Reach_log`, `Engagement_Rate`
- Optional features: `Engagement_Density`
- One-hot encoded columns: `Platform_*`, `Post_Type_*`, `Sentiment_*`

**Example:**
```python
from utils import filter_essential_columns

df_filtered = filter_essential_columns(df_processed)
```

#### `preprocess_data(df, target_col, fit_scaler, scaler)`

Generic preprocessing function that applies all transformations.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `target_col` (str, optional): Target column name
- `fit_scaler` (bool, default True): Whether to fit a new scaler
- `scaler` (StandardScaler, optional): Pre-fitted scaler

**Returns:**
- `tuple`: (preprocessed_df, scaler)

**Example:**
```python
from utils import preprocess_data

df_processed, scaler = preprocess_data(df_raw, target_col="Reach")
```

### Model Storage Utilities (`model_storage.py`)

#### `save_training_results(results_df, output_dir, filename, models_dict)`

Save training results (weights) and models to file for later use.

**Parameters:**
- `results_df` (pd.DataFrame): Results DataFrame with learned weights
- `output_dir` (Path): Directory to save results
- `filename` (str, default "training_results"): Base filename (without extension)
- `models_dict` (dict, optional): Dictionary of trained models (for Random Forest)

**Returns:**
- `Path`: Path to saved CSV file

**Saves:**
- CSV file: `{filename}.csv`
- SQLite database: `{filename}.db`
- Pickle file (if models provided): `{filename}_models.pkl`

**Example:**
```python
from pathlib import Path
from utils import save_training_results

output_dir = Path("outputs")
save_training_results(results_df, output_dir, models_dict=models_dict)
```

#### `load_training_results(results_path, from_db, load_models)`

Load training results from file.

**Parameters:**
- `results_path` (Path): Path to results file (CSV or DB)
- `from_db` (bool, default True): Whether to load from database (True) or CSV (False)
- `load_models` (bool, default False): Whether to load saved models

**Returns:**
- `pd.DataFrame` or `tuple`: Results DataFrame, or (DataFrame, models_dict) if load_models=True

**Example:**
```python
from pathlib import Path
from utils import load_training_results

results_path = Path("outputs/training_results.db")
results_df = load_training_results(results_path, from_db=True)

# Load with models
results_df, models_dict = load_training_results(
    results_path, from_db=True, load_models=True
)
```

## 🔄 Data Preprocessing Pipeline

The preprocessing pipeline applies the following transformations:

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

### 7. Column Filtering
After preprocessing, the pipeline automatically filters to only essential columns:
- **Grouping columns**: `Platform`, `Post Type`
- **Feature columns**: `Likes_log_log`, `Comments_log_log`, `Shares_log_log`
- **Target columns**: `Reach_log`, `Engagement_Rate`
- **Optional features**: `Engagement_Density` (if available)
- **One-hot encoded columns**: All `Platform_*`, `Post_Type_*`, `Sentiment_*` columns

This optimization reduces database size and improves query performance by removing
unused columns like original engagement metrics, metadata (Post ID, Post Content,
Post Timestamp), audience demographics, and temporal features that aren't used in
the analysis.

**Note:** Raw data is preserved in `*_data_raw` tables. Only processed tables are
filtered.

