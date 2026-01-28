"""
Pytest configuration for reference comparison tests.

This configuration ensures test order and fail-fast behavior:
1. Normal spectra tests run first
2. If normal spectra fail, other tests are skipped
3. First failing spectrum is printed with full details
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options for test configuration."""
    parser.addoption(
        "--max-spectra",
        action="store",
        default="300",
        help="Maximum number of spectra to test for real data (default: 300)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Modify test collection to ensure proper ordering.

    Why: Normal spectra tests should run first. If they fail, we want to see
    the failure immediately before running more expensive tests.
    """
    # Define test order priority (lower = runs first)
    test_order = {
        "test_normal_spectra": 1,
        "test_problematic_spectra": 2,
        "test_real_data": 3,
    }

    def get_test_priority(item: pytest.Item) -> int:
        """Get priority for a test item based on its module name."""
        module_name = item.module.__name__
        for key, priority in test_order.items():
            if key in module_name:
                return priority
        return 99  # Unknown tests run last

    # Sort items by priority
    items.sort(key=get_test_priority)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """
    Hook to track test failures and potentially skip subsequent tests.

    Why: If normal spectra tests fail, we want to stop running other tests
    since they depend on the baseline behavior being correct.
    """
    outcome = yield
    report = outcome.get_result()

    # Track if a normal spectra test failed
    if report.when == "call" and report.failed:
        if "test_normal_spectra" in item.nodeid:
            # Mark that normal spectra failed
            item.session.normal_spectra_failed = True  # type: ignore


def pytest_runtest_setup(item: pytest.Item) -> None:
    """
    Hook to skip tests if normal spectra tests have failed.

    Why: No point running problematic or real data tests if basic normal
    spectra tests are failing.
    """
    # Skip problematic and real data tests if normal spectra failed
    if hasattr(item.session, "normal_spectra_failed"):
        if "test_problematic_spectra" in item.nodeid or "test_real_data" in item.nodeid:
            pytest.skip("Skipping due to normal spectra test failure")
