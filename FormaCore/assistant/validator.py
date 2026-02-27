"""
FormaCore AI - assistant/validator.py
Strict validation of action JSON against registered schemas.
Rejects anything not explicitly allowed.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

from assistant.actions import ACTION_REGISTRY


class ValidationError:
    """A single validation issue."""

    def __init__(self, field: str, msg: str):
        self.field = field
        self.msg = msg

    def __repr__(self):
        return f"ValidationError({self.field}: {self.msg})"


class ValidationResult:
    """Result of validating an action."""

    def __init__(self, errors: List[ValidationError],
                 cleaned: Dict[str, Any]):
        self.errors = errors
        self.cleaned = cleaned

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def validate_action(action_id: str,
                    payload: Dict[str, Any]) -> ValidationResult:
    """
    Validate an action against its schema.

    Returns ValidationResult with:
      - valid: True if no errors
      - errors: list of ValidationError
      - cleaned: payload with defaults applied and types coerced
    """
    errors: List[ValidationError] = []
    cleaned: Dict[str, Any] = {}

    # Check action exists
    schema = ACTION_REGISTRY.get(action_id)
    if schema is None:
        errors.append(ValidationError(
            "action_id",
            f"Unknown action: '{action_id}'. "
            f"Allowed: {list(ACTION_REGISTRY.keys())}"
        ))
        return ValidationResult(errors, cleaned)

    fields = schema["fields"]

    # Check for unknown fields
    for key in payload:
        if key not in fields:
            errors.append(ValidationError(
                key, f"Unknown field '{key}' for action '{action_id}'"
            ))

    # Validate each field
    for name, spec in fields.items():
        value = payload.get(name)
        required = spec.get("required", False)
        field_type = spec.get("type", "str")

        # Missing required field
        if value is None:
            if required:
                errors.append(ValidationError(
                    name, f"Required field '{name}' is missing"
                ))
                continue
            else:
                # Apply default
                cleaned[name] = spec.get("default")
                continue

        # Type checking and coercion
        try:
            value = _coerce_type(value, field_type)
        except (ValueError, TypeError) as e:
            errors.append(ValidationError(
                name, f"Field '{name}' type error: expected "
                      f"{field_type}, got {type(value).__name__}"
            ))
            continue

        # Allowed values check
        if "allowed" in spec and value not in spec["allowed"]:
            errors.append(ValidationError(
                name, f"Field '{name}' value {value} not in "
                      f"allowed values: {spec['allowed']}"
            ))
            continue

        # Range checks
        if "min" in spec and isinstance(value, (int, float)):
            if value < spec["min"]:
                errors.append(ValidationError(
                    name, f"Field '{name}' value {value} below "
                          f"minimum {spec['min']}"
                ))
                continue

        if "max" in spec and isinstance(value, (int, float)):
            if value > spec["max"]:
                errors.append(ValidationError(
                    name, f"Field '{name}' value {value} above "
                          f"maximum {spec['max']}"
                ))
                continue

        cleaned[name] = value

    return ValidationResult(errors, cleaned)


def _coerce_type(value: Any, expected: str) -> Any:
    """Try to coerce value to expected type."""
    if expected == "str":
        return str(value)
    elif expected == "int":
        return int(value)
    elif expected == "float":
        return float(value)
    elif expected == "bool":
        return bool(value)
    elif expected == "list_str":
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        raise TypeError("Expected list")
    elif expected == "dict":
        if isinstance(value, dict):
            return value
        raise TypeError("Expected dict")
    else:
        return value
