"""Validation workflow for engagement weight analysis.

This module handles the complete validation workflow including:
- Loading training results
- Loading test data
- Feature range validation
- Model validation
- Generating validation visualizations
- Displaying validation metrics
"""

from pathlib import Path

from analysis import print_validation_results, validate_model
from utils import load_training_results
from utils.data_loading import load_processed_data


def _load_training_results(output_dir: Path) -> tuple:
    """
    Load training results from saved files.

    Parameters
    ----------
    output_dir : Path
        Directory containing training results.

    Returns
    -------
    tuple
        Tuple of (train_results_df, rf_models) or (None, None) on error.
    """
    print("\n📖 Loading training results...")
    try:
        results_path = output_dir / "training_results.db"
        train_results_df, rf_models = load_training_results(
            results_path, from_db=True, load_models=True
        )
        print(f"✓ Loaded training results: {len(train_results_df)} segments")
        if rf_models:
            print(f"✓ Loaded {len(rf_models)} Random Forest models")
        return train_results_df, rf_models
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease run 'train' mode first to generate training results.")
        return None, None
    except Exception:
        # Fallback if models not available
        try:
            results_path = output_dir / "training_results.db"
            train_results_df = load_training_results(results_path, from_db=True)
            print(f"✓ Loaded training results: {len(train_results_df)} segments")
            print("⚠️  Models not available (using weight-based prediction)")
            return train_results_df, None
        except Exception as e2:
            print(f"❌ {e2}")
            return None, None


def _load_training_data_for_validation(project_root: Path):
    """
    Load training data for feature range validation.

    Parameters
    ----------
    project_root : Path
        Root directory of the project.

    Returns
    -------
    pd.DataFrame | None
        Training DataFrame if loaded successfully, None otherwise.
    """
    print("\n📖 Loading training data for feature range validation...")
    try:
        train_df, _ = load_processed_data(project_root, "train", "processed")
        print(f"✓ Loaded training data: {len(train_df):,} rows")
        return train_df
    except Exception as e:
        print(f"⚠️  Warning: Could not load training data: {e}")
        print("   Validation will proceed without feature range checking.")
        return None


def _print_validation_summary(output_dir: Path) -> None:
    """
    Print validation summary with file locations.

    Parameters
    ----------
    output_dir : Path
        Directory containing validation outputs.
    """
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Validation results saved to: {output_dir}")
    print("  • Prediction vs Actual: prediction_vs_actual.png")
    print("  • Confusion Matrix: confusion_matrix.png")
    print("=" * 80 + "\n")


def test_model() -> None:
    """
    Test model on test data and provide validation analysis.

    This function performs the complete validation workflow:
    1. Loads training results (weights and optionally models)
    2. Loads preprocessed test data
    3. Optionally loads training data for feature range validation
    4. Validates model performance on test set
    5. Generates validation visualizations and metrics

    The function handles cases where Random Forest models may not be
    available, falling back to weight-based predictions.

    Raises
    ------
    FileNotFoundError
        If training results cannot be found.
    ValueError
        If test data cannot be loaded.
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING MODE: Validating on Test Data")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"

    # Load training results (weights and optionally models)
    train_results_df, rf_models = _load_training_results(output_dir)
    if train_results_df is None:
        return

    # Load test data
    try:
        test_df, _ = load_processed_data(project_root, "test", "processed")
    except ValueError as e:
        print(f"\n{e}")
        return

    # Optionally load training data for feature range validation
    # This helps identify if test data has values outside training range
    train_df = _load_training_data_for_validation(project_root)

    # Perform validation analysis
    print("\n" + "=" * 80)
    print("🔍 VALIDATION ANALYSIS")
    print("=" * 80)

    print("\nValidating model on test data...")
    validation_results = validate_model(
        train_results_df,
        test_df,
        train_df=train_df,
        output_dir=output_dir,
        clip_outliers=True,
        rf_models=rf_models,
    )

    # Display validation results
    print_validation_results(validation_results)

    # Display summary
    _print_validation_summary(output_dir)
