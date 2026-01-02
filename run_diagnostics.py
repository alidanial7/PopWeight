"""Quick diagnostic script to check validation issues."""

from pathlib import Path

from analysis.validation import diagnose_validation_issues, load_training_weights
from utils import load_training_results, read_from_sqlite

# Get project root
project_root = Path(__file__).parent
output_dir = project_root / "outputs"

# Load training results
print("Loading training results...")
results_path = output_dir / "training_results.db"
train_results_df = load_training_results(results_path, from_db=True)
print(f"✓ Loaded {len(train_results_df)} segments\n")

# Load test data
print("Loading test data...")
test_db_path = project_root / "data" / "test.db"
test_df = read_from_sqlite(db_path=str(test_db_path), table_name="test_data_processed")
print(f"✓ Loaded {len(test_df)} test rows\n")

# Load weights
weights_df = load_training_weights(train_results_df)

# Run diagnostics
diagnose_validation_issues(train_results_df, test_df, weights_df)
