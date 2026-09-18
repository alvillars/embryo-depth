import os

import pytest

# Must be set before qtpy imports Qt, so the GUI tests run headless on a machine with no
# display and do not pop windows on one that has a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_app():
    """A single QApplication for the session.

    Qt allows only one, and destroying it mid-session breaks later widget construction.
    """
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
