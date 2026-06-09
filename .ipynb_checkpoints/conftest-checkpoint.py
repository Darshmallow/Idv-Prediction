import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that read every xlsx file (deselect with -m 'not slow')",
    )
