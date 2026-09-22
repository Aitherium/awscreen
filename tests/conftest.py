"""Pytest configuration for awscreen tests."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_PKG_ROOT = _Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))

import pytest  # noqa: E402

# Check if optional dependencies are available
try:
    import anthropic  # noqa: F401
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from PIL import Image  # noqa: F401
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


@pytest.fixture
def mock_finder():
    """Provide a Finder instance for testing (or skip if Anthropic unavailable)."""
    if not HAS_ANTHROPIC:
        pytest.skip("Anthropic not available")
    from awscreen import Finder
    return Finder(api_key="test-key")


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API."""
    return (
        '[{"x":10,"y":20,"width":100,"height":50,'
        '"confidence":0.95,"description":"button"}]'
    )
