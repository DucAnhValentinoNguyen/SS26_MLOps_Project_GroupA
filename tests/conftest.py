"""Pytest configuration and shared fixtures for the test suite."""

from unittest.mock import MagicMock
import sys

sys.modules["peft"] = MagicMock()
