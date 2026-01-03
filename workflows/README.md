# Workflows Documentation

The `workflows/` package contains modular workflow functions for all major
operations. Each workflow is self-contained and can be used independently.

## 📦 Available Workflows

### Data Preparation Workflows (`data_preparation.py`)

#### `generate_data(n_samples=None)`

Generate a realistic synthetic social media engagement dataset.

**Parameters:**
- `n_samples` (int, optional): Number of samples to generate. If None (default),
  prompts user interactively for the number of samples.

**Features:**
- Interactive prompt for number of samples (default: 30000 if Enter pressed)
- Creates platform-specific engagement patterns
- Generates realistic reach calculations based on engagement
- Includes multiple platforms (Facebook, Instagram, Twitter, LinkedIn)
- Supports different post types (Image, Video, Link)
- Saves to `data/social_media_engagement_data.xlsx`
- Progress bar showing generation progress

**Usage:**
```python
from workflows import generate_data

# Interactive mode (prompts for number of samples)
generate_data()

# Programmatic mode (specify number directly)
generate_data(n_samples=10000)
```

**Interactive Prompt:**
When called without parameters, the function will prompt:
```
Enter number of samples to generate (default: 30000, press Enter for default):
```

#### `split_data()`

Split base data into train and test datasets with interactive percentage input.

**Features:**
- Interactive percentage input (0-100)
- Shuffles data before splitting
- Creates `data/train.xlsx` and `data/test.xlsx`
- Confirmation prompt before overwriting existing files

**Example:**
```python
from workflows import split_data

split_data()  # Interactive prompt for train percentage
```

#### `import_train_data()`

Import training data from Excel to SQLite database.

**Features:**
- Reads from `data/train.xlsx`
- Saves to `data/train.db` (table: `train_data_raw`)
- Progress indicators with tqdm
- Error handling and validation

**Example:**
```python
from workflows import import_train_data

import_train_data()
```

#### `import_test_data()`

Import test data from Excel to SQLite database.

**Features:**
- Reads from `data/test.xlsx`
- Saves to `data/test.db` (table: `test_data_raw`)
- Progress indicators with tqdm
- Error handling and validation

**Example:**
```python
from workflows import import_test_data

import_test_data()
```

#### `preprocess_train()`

Preprocess training data from SQLite database.

**Features:**
- Reads from `train_data_raw` table
- Applies all preprocessing transformations
- Filters to essential columns only
- Saves to `train_data_processed` table
- Detailed progress indicators

**Preprocessing Steps:**
1. Missing value handling
2. Log-log normalization
3. Temporal feature extraction
4. One-hot encoding
5. Feature scaling
6. Target transformation
7. Column filtering

**Example:**
```python
from workflows import preprocess_train

preprocess_train()
```

#### `preprocess_test()`

Preprocess test data from SQLite database.

**Features:**
- Reads from `test_data_raw` table
- Applies same preprocessing as training
- Filters to essential columns only
- Saves to `test_data_processed` table
- Detailed progress indicators

**Example:**
```python
from workflows import preprocess_test

preprocess_test()
```

### Training Workflow (`training.py`)

#### `train_model()`

Complete training workflow for learning engagement weights.

**Workflow Steps:**
1. Loads preprocessed training data
2. Trains Linear Regression and Random Forest models
3. Compares models and selects best performing
4. Generates visualizations (heatmaps, facet grids)
5. Provides statistical insights
6. Identifies trending posts
7. Saves training results

**Outputs:**
- `outputs/gamma_heatmap.png` - Heatmap of engagement weights
- `outputs/weights_facet_grid.png` - Facet grid visualization
- `outputs/training_results.db` - Saved weights and models
- `outputs/training_results.csv` - CSV export of results

**Example:**
```python
from workflows import train_model

train_model()
```

### Validation Workflow (`validation.py`)

#### `test_model()`

Complete validation workflow for testing learned weights.

**Workflow Steps:**
1. Loads training results (weights and models)
2. Loads preprocessed test data
3. Optionally loads training data for feature range validation
4. Validates model performance on test set
5. Generates validation visualizations
6. Displays validation metrics

**Outputs:**
- `outputs/prediction_vs_actual.png` - Prediction vs actual scatter plot
- `outputs/confusion_matrix.png` - Confusion matrix for classification

**Example:**
```python
from workflows import test_model

test_model()
```

### Correlation Workflows (`correlation.py`)

#### `correlation_likes_reach()`

Calculate and display correlation between Likes and Reach.

**Features:**
- Loads data from training database
- Calculates Pearson correlation coefficient
- Interprets correlation strength and direction
- Displays statistical summaries

**Example:**
```python
from workflows import correlation_likes_reach

correlation_likes_reach()
```

#### `correlation_comments_reach()`

Calculate and display correlation between Comments and Reach.

**Example:**
```python
from workflows import correlation_comments_reach

correlation_comments_reach()
```

#### `correlation_shares_reach()`

Calculate and display correlation between Shares and Reach.

**Example:**
```python
from workflows import correlation_shares_reach

correlation_shares_reach()
```

### Diagnostics Workflow (`diagnostics.py`)

#### `run_diagnostics()`

Run diagnostic checks on training and test data.

**Checks Performed:**
- Platform-PostType coverage mismatches
- Model performance issues
- Feature range outliers
- Missing weight mappings

**Example:**
```python
from workflows import run_diagnostics

run_diagnostics()
```

## 🔄 Complete Workflow Example

Here's a complete example of running all workflows in sequence:

**Using the Interactive Menu:**
```bash
python main.py
# Then select options 1-8 in order
```

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
    correlation_likes_reach,
    run_diagnostics,
)

# 1. Generate synthetic data (interactive prompt for sample count)
generate_data()

# 2. Split into train/test (interactive prompt for percentage)
split_data()

# 3. Import data to databases
import_train_data()
import_test_data()

# 4. Preprocess data
preprocess_train()
preprocess_test()

# 5. Train models
train_model()

# 6. Validate models
test_model()

# 7. Analyze correlations
correlation_likes_reach()

# 8. Run diagnostics
run_diagnostics()
```

**Note:** `generate_data()` and `split_data()` will prompt interactively for
parameters. You can also provide parameters directly:
- `generate_data(n_samples=10000)` - Specify sample count directly
- `split_data()` - Always prompts for train percentage (for safety)

