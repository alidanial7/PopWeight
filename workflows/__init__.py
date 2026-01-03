"""Workflow modules for the PopWeight analysis system.

This package contains workflow modules for different operations:
- Data Preparation: Data generation, splitting, importing, preprocessing
- Training: Model training workflow
- Validation: Model validation workflow
- Correlation: Correlation analysis workflow
- Diagnostics: Diagnostic tools for validation issues
"""

from .correlation import (
    correlation_comments_reach,
    correlation_likes_reach,
    correlation_shares_reach,
)
from .data_generation import generate_data
from .data_preparation import (
    import_test_data,
    import_train_data,
    preprocess_test,
    preprocess_train,
    split_data,
)
from .diagnostics import run_diagnostics
from .training import train_model
from .validation import test_model

__all__ = [
    # Data preparation
    "generate_data",
    "split_data",
    "import_train_data",
    "import_test_data",
    "preprocess_train",
    "preprocess_test",
    # Analysis
    "train_model",
    "test_model",
    "correlation_likes_reach",
    "correlation_comments_reach",
    "correlation_shares_reach",
    # Diagnostics
    "run_diagnostics",
]
