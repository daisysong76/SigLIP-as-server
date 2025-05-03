"""Pytest configuration file."""
import pytest

def pytest_configure(config):
    """Configure pytest-asyncio to use function scope by default."""
    config.option.asyncio_default_fixture_loop_scope = "function" 