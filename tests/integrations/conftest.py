"""
Conftest for integration tests.

This module provides fixtures and skip markers for integration tests
that require optional dependencies like nautilus_trader.
"""

import pytest


# Check if nautilus_trader is available
try:
    import nautilus_trader
    HAS_NAUTILUS = True
except ImportError:
    HAS_NAUTILUS = False


# Check if openbb is available
try:
    import openbb
    HAS_OPENBB = True
except ImportError:
    HAS_OPENBB = False


# Skip decorators for optional dependencies
requires_nautilus = pytest.mark.skipif(
    not HAS_NAUTILUS,
    reason="nautilus_trader not installed"
)

requires_openbb = pytest.mark.skipif(
    not HAS_OPENBB,
    reason="openbb not installed"
)


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests that require missing dependencies."""
    for item in items:
        # Check for nautilus-related tests
        if "nautilus" in item.nodeid.lower():
            if not HAS_NAUTILUS:
                item.add_marker(pytest.mark.skip(
                    reason="nautilus_trader not installed"
                ))

        # Check for end-to-end tests (which require both nautilus and openbb)
        if "end_to_end" in item.nodeid.lower():
            if not HAS_NAUTILUS:
                item.add_marker(pytest.mark.skip(
                    reason="nautilus_trader not installed (required for end-to-end tests)"
                ))
            elif not HAS_OPENBB:
                item.add_marker(pytest.mark.skip(
                    reason="openbb not installed (required for end-to-end tests)"
                ))

        # Check for openbb-related tests
        if "openbb" in item.nodeid.lower() and "mock" not in item.nodeid.lower():
            if not HAS_OPENBB:
                item.add_marker(pytest.mark.skip(
                    reason="openbb not installed"
                ))
