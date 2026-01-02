"""Command-line commands for the PopWeight project."""

from .import_excel import import_excel_command
from .import_test import import_test_command
from .import_train import import_train_command

__all__ = [
    "import_excel_command",
    "import_train_command",
    "import_test_command",
]
