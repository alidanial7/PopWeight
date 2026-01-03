"""Main script for engagement weight analysis with train/test modes.

This module provides the main entry point for the PopWeight analysis system.
It offers an interactive menu for:
- Data preparation (generate, split, import, preprocess)
- Training models on engagement data
- Testing/validating models on test data
- Analyzing correlations between engagement metrics and reach
- Running diagnostics

The script delegates all workflow operations to specialized modules in the
workflows package, keeping the main file clean and focused on user interaction.
"""

from workflows import (
    correlation_comments_reach,
    correlation_likes_reach,
    correlation_shares_reach,
    generate_data,
    import_test_data,
    import_train_data,
    preprocess_test,
    preprocess_train,
    run_diagnostics,
    split_data,
    test_model,
    train_model,
)


def _display_menu() -> None:
    """
    Display the main menu options to the user.

    This function prints a formatted menu with all available options
    for the engagement weight analysis system.
    """
    print("\n" + "=" * 80)
    print("MAIN MENU")
    print("=" * 80)
    print("\n📊 Data Preparation:")
    print("  1. Generate Data - Create synthetic dataset")
    print("  2. Split Data - Split base data into train/test")
    print("  3. Import Train - Import training data to database")
    print("  4. Import Test - Import test data to database")
    print("  5. Preprocess Train - Preprocess training data")
    print("  6. Preprocess Test - Preprocess test data")
    print("\n🔬 Analysis:")
    print("  7. Train - Learn weights from training data")
    print("  8. Test - Validate weights on test data")
    print("  9. Correlation - Likes vs Reach")
    print(" 10. Correlation - Comments vs Reach")
    print(" 11. Correlation - Shares vs Reach")
    print("\n🔍 Utilities:")
    print(" 12. Diagnostics - Run validation diagnostics")
    print("\n  q. Quit - Exit the program")
    print()


def _handle_menu_choice(choice: str) -> bool:
    """
    Handle user menu choice and execute corresponding action.

    Parameters
    ----------
    choice : str
        User's menu choice (1-12, 'q', 'train', 'test', etc.).

    Returns
    -------
    bool
        True if the program should continue, False if it should exit.
    """
    choice = choice.strip().lower()

    # Map choices to workflow functions
    menu_actions = {
        # Data preparation
        "1": generate_data,
        "generate": generate_data,
        "2": split_data,
        "split": split_data,
        "3": import_train_data,
        "import_train": import_train_data,
        "4": import_test_data,
        "import_test": import_test_data,
        "5": preprocess_train,
        "preprocess_train": preprocess_train,
        "6": preprocess_test,
        "preprocess_test": preprocess_test,
        # Analysis
        "7": train_model,
        "train": train_model,
        "8": test_model,
        "test": test_model,
        "9": correlation_likes_reach,
        "10": correlation_comments_reach,
        "11": correlation_shares_reach,
        # Utilities
        "12": run_diagnostics,
        "diagnostics": run_diagnostics,
    }

    # Handle quit
    if choice in ("q", "quit"):
        print("\n👋 Exiting...")
        print("=" * 80 + "\n")
        return False

    # Execute action if valid choice
    if choice in menu_actions:
        menu_actions[choice]()
        print("\n" + "-" * 80)
        print("Returning to main menu...")
        print("-" * 80)
        return True

    # Invalid choice
    print("❌ Invalid choice. Please enter 1-12, or 'q'.")
    print()
    return True


def main() -> None:
    """
    Main entry point for the engagement weight analysis system.

    This function provides an interactive menu-driven interface that allows
    users to:
    - Train models on engagement data
    - Validate models on test data
    - Analyze correlations between engagement metrics and reach

    The function runs in a loop until the user chooses to quit.

    Examples
    --------
    Run the main script:
        python main.py

    The script will display a menu and wait for user input.
    """
    print("\n" + "=" * 80)
    print("🔬 ENGAGEMENT WEIGHT ANALYSIS SYSTEM")
    print("=" * 80)

    # Main loop - continue until user quits
    while True:
        _display_menu()
        choice = input("Enter choice (1-12, or 'q' to quit): ")

        # Handle choice and check if we should continue
        if not _handle_menu_choice(choice):
            break


if __name__ == "__main__":
    main()
