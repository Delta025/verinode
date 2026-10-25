"""
Pytest configuration for the Knowledge Graph AI test suite.

Defines custom markers and shared fixtures.
"""

import pytest


def pytest_addoption(parser):
    """Add command-line options for test configuration."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require a running Ollama server.",
    )
    parser.addoption(
        "--run-calibration",
        action="store_true",
        default=False,
        help="Run calibration tests that require models and datasets.",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a running Ollama server (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers",
        "calibration: marks calibration tests that require models and datasets (deselect with '-m \"not calibration\"')",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration/calibration tests unless their flags are passed."""
    run_integration = config.getoption("--run-integration")
    run_calibration = config.getoption("--run-calibration")

    skip_integration = pytest.mark.skip(
        reason="Needs --run-integration flag and a running Ollama server"
    )
    skip_calibration = pytest.mark.skip(
        reason="Needs --run-calibration flag (and possibly models/datasets)"
    )

    for item in items:
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
        if "calibration" in item.keywords and not run_calibration:
            item.add_marker(skip_calibration)
