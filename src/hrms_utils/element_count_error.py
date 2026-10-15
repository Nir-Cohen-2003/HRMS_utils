"""Validation shared by workflow controls and the native isotope binding."""

import math
import sys
from collections.abc import Mapping

ELEMENT_COUNT_ERROR_KEYS = frozenset({"C", "S", "Cl", "Br"})
MAXIMUM_ELEMENT_COUNT_ERROR = 2**31 - 1


def validate_carbon_tolerance(value: int | float | None) -> float | None:
    """Validate a nonnegative finite tolerance and normalize it for native serialization."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("carbon_tolerance must be a nonnegative finite int or float (not bool).")
    if value < 0:
        raise ValueError("carbon_tolerance must be a nonnegative finite int or float.")
    if isinstance(value, int) and value > int(sys.float_info.max):
        raise ValueError("carbon_tolerance must convert to a finite float.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(
            "carbon_tolerance must be a nonnegative finite int or float."
        )
    return normalized


def validate_element_count_errors(
    minimum_element_count_error: Mapping[str, int] | None,
    maximum_element_count_error: Mapping[str, int] | None,
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    """Copy validated integer count margins into native-serializable dictionaries."""
    validated: list[dict[str, int] | None] = []
    for name, values in (
        ("minimum_element_count_error", minimum_element_count_error),
        ("maximum_element_count_error", maximum_element_count_error),
    ):
        if values is None:
            validated.append(None)
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping of C, S, Cl, Br to integers.")
        for element, value in values.items():
            if element not in ELEMENT_COUNT_ERROR_KEYS:
                raise ValueError(
                    f"{name} keys must be exactly spelled C, S, Cl, Br; got {element!r}."
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= MAXIMUM_ELEMENT_COUNT_ERROR
            ):
                raise ValueError(
                    f"{name}[{element!r}] must be an integer in 0..{MAXIMUM_ELEMENT_COUNT_ERROR} (not bool)."
                )
        validated.append({element: int(value) for element, value in values.items()})
    minimum, maximum = validated
    if minimum is not None and maximum is not None:
        for element in minimum.keys() & maximum.keys():
            if minimum[element] > maximum[element]:
                raise ValueError(
                    f"minimum_element_count_error[{element!r}] must be <= maximum_element_count_error[{element!r}]."
                )
    return minimum, maximum
