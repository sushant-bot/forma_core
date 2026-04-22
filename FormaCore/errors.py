"""
FormaCore AI - errors.py
Shared exception types for product-level error handling.
"""
from __future__ import annotations


class FormaCoreError(Exception):
    """Base class for all FormaCore-specific failures."""


class ConfigurationError(FormaCoreError):
    """Raised when runtime configuration is invalid."""


class ParseError(FormaCoreError):
    """Raised when an input file cannot be parsed."""


class RoutingError(FormaCoreError):
    """Raised when routing cannot be completed reliably."""


class ExportError(FormaCoreError):
    """Raised when output artifact generation fails."""