"""Pytest configuration for Qt-based tests."""
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def cleanup_qt_objects(qtbot, request):
    yield
    app = QApplication.instance()
    if app:
        app.processEvents()
