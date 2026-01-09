"""
Pytest configuration for Qt-based tests.

Provides fixtures for proper Qt object cleanup between tests to prevent segfaults.
"""
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def cleanup_qt_objects(qtbot, request):
    """
    Auto-cleanup fixture that runs after each test to ensure Qt objects are
    properly destroyed before the next test starts.
    
    This prevents segmentation faults caused by dangling Qt object references.
    """
    yield
    
    # Process any pending events
    app = QApplication.instance()
    if app:
        app.processEvents()
