import sys
from pathlib import Path

import platformdirs


def get_site_packages() -> Path:
    return Path(platformdirs.user_data_dir("WhisperNow")) / "packages"


def setup_path() -> None:
    site_packages = get_site_packages()
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def check_dependencies_available() -> bool:
    setup_path()
    try:
        import importlib.util

        required_modules = ["torch", "nemo"]
        for module in required_modules:
            if importlib.util.find_spec(module) is None:
                return False
        return True
    except Exception:
        return False


def run_dependency_wizard() -> bool:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QMessageBox,
        QVBoxLayout,
        QWizard,
        QWizardPage,
    )

    from whispernow.core.deps import DependencyManager
    from whispernow.ui.dependency_setup_page import DependencySetupPage

    app = QApplication(sys.argv)
    app.setApplicationName("WhisperNow Setup")

    manager = DependencyManager()

    wizard = QWizard()
    wizard.setWindowTitle("WhisperNow - First Time Setup")
    wizard.setWizardStyle(QWizard.ModernStyle)
    wizard.setMinimumSize(550, 400)

    welcome_page = QWizardPage()
    welcome_page.setTitle("Welcome to WhisperNow")
    welcome_page.setSubTitle("First-time setup required")

    welcome_layout = QVBoxLayout(welcome_page)
    welcome_label = QLabel(
        "WhisperNow needs to download AI speech recognition components.\n\n"
        "This is a one-time setup that will:\n"
        "• Download PyTorch and speech recognition models\n"
        "• May take 5-15 minutes depending on your connection\n"
        "• Requires approximately 500 MB - 2.5 GB of disk space\n\n"
        "Click Next to begin."
    )
    welcome_label.setWordWrap(True)
    welcome_layout.addWidget(welcome_label)
    welcome_layout.addStretch()

    wizard.addPage(welcome_page)

    dep_page = DependencySetupPage(manager)
    wizard.addPage(dep_page)

    complete_page = QWizardPage()
    complete_page.setTitle("Setup Complete!")
    complete_page.setSubTitle("WhisperNow is ready to use.")

    complete_layout = QVBoxLayout(complete_page)
    complete_label = QLabel(
        "All components have been installed successfully.\n\n"
        "Click Finish to start WhisperNow."
    )
    complete_label.setWordWrap(True)
    complete_layout.addWidget(complete_label)
    complete_layout.addStretch()

    wizard.addPage(complete_page)

    result = wizard.exec()

    if result == QWizard.Accepted:
        return True
    else:
        return False


def main():
    if not check_dependencies_available():
        success = run_dependency_wizard()
        if not success:
            print("Setup cancelled or failed. Exiting.")
            sys.exit(1)

        setup_path()

        if not check_dependencies_available():
            print("Dependencies not found after installation. Please try again.")
            sys.exit(1)

    from whispernow.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
