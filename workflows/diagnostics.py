"""Diagnostics workflow for validation issues.

This module provides diagnostic tools to check for validation issues
between training and test data.
"""

from pathlib import Path

from analysis.validation import diagnose_validation_issues, load_training_weights
from utils import load_training_results, read_from_sqlite


def run_diagnostics() -> None:
    """
    Run diagnostic checks on training and test data.

    This function checks for:
    - Platform-PostType coverage mismatches
    - Model performance issues
    - Feature range outliers
    - Missing weight mappings
    """
    print("\n" + "=" * 80)
    print("🔍 RUNNING DIAGNOSTICS")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"

    # Load training results
    print("\n📖 Loading training results...")
    try:
        results_path = output_dir / "training_results.db"
        train_results_df = load_training_results(results_path, from_db=True)
        print(f"✓ Loaded {len(train_results_df)} segments\n")
    except Exception as e:
        print(f"❌ Error loading training results: {e}")
        print("\nPlease run 'train' mode first.")
        return

    # Load test data
    print("📖 Loading test data...")
    try:
        test_db_path = project_root / "data" / "test.db"
        test_df = read_from_sqlite(
            db_path=str(test_db_path), table_name="test_data_processed"
        )
        print(f"✓ Loaded {len(test_df)} test rows\n")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # Load weights
    print("📖 Loading training weights...")
    try:
        weights_df = load_training_weights(train_results_df)
        print("✓ Loaded training weights\n")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # Run diagnostics
    print("=" * 80)
    print("🔍 DIAGNOSTIC RESULTS")
    print("=" * 80)
    diagnose_validation_issues(train_results_df, test_df, weights_df, verbose=True)

    print("\n" + "=" * 80)
    print("✅ DIAGNOSTICS COMPLETE")
    print("=" * 80 + "\n")
