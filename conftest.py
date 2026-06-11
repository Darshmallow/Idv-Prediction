import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that read every xlsx file (deselect with -m 'not slow')",
    )
