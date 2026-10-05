import pytest

# Make all async test functions automatically treated as asyncio tests.
# This is required for pytest-asyncio >= 0.21 (strict/auto mode).
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
