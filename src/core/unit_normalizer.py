"""
Unit normalization for contradiction detection.

Canonicalizes common measurement units to a single standard form so that
identical measurements expressed in different units (e.g., "1.6 million
miles" vs "2.57 million km") are no longer flagged as contradictions.

Canonical forms:
- Distances → km
- Speeds → km/h
- Temperatures → °C
- Masses → kg
- Areas → km²

All conversions are pure text transforms — no external dependencies.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Number parsing helpers
# ------------------------------------------------------------------

# Multiplier words: "1.6 million" → 1_600_000
_MULTIPLIERS = {
    "hundred": 1e2,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}

# Regex for a number with optional commas and decimals
_NUMBER_RE = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"

# Optional multiplier word after number
_MULTIPLIER_RE = r"(?:\s+(hundred|thousand|million|billion|trillion))?"


def _parse_number(match_str: str, multiplier: Optional[str] = None) -> float:
    """Parse a number string (possibly with commas) and optional multiplier."""
    clean = match_str.replace(",", "")
    value = float(clean)
    if multiplier:
        value *= _MULTIPLIERS.get(multiplier.lower(), 1.0)
    return value


def _format_number(value: float) -> str:
    """Format a number compactly, using multiplier words for large values."""
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"{value / 1e12:.2f} trillion"
    if abs_val >= 1e9:
        return f"{value / 1e9:.2f} billion"
    if abs_val >= 1e6:
        return f"{value / 1e6:.2f} million"
    if abs_val >= 1e4:
        # For values >= 10,000, use integer if close to whole number
        if abs(value - round(value)) < 0.01:
            return str(int(round(value)))
        return f"{value:.2f}"
    if abs(value - round(value)) < 0.001:
        return str(int(round(value)))
    return f"{value:.2f}"


# ------------------------------------------------------------------
# Conversion factors (all → canonical unit)
# ------------------------------------------------------------------

# Distance → km
_DIST_TO_KM = {
    "mi": 1.60934,
    "mile": 1.60934,
    "miles": 1.60934,
    "meter": 0.001,
    "meters": 0.001,
    "metre": 0.001,
    "metres": 0.001,
    "m": 0.001,
    "ft": 0.0003048,
    "feet": 0.0003048,
    "foot": 0.0003048,
    "yd": 0.0009144,
    "yard": 0.0009144,
    "yards": 0.0009144,
    "au": 1.496e8,
}

# Speed → km/h
_SPEED_TO_KMH = {
    "mph": 1.60934,
    "mi/h": 1.60934,
    "m/s": 3.6,
    "ft/s": 1.09728,
    "knot": 1.852,
    "knots": 1.852,
    "kn": 1.852,
}

# Temperature conversions (special — not simple multiplication)
# Handled separately in _convert_temperature()

# Mass → kg
_MASS_TO_KG = {
    "lb": 0.453592,
    "lbs": 0.453592,
    "pound": 0.453592,
    "pounds": 0.453592,
    "oz": 0.0283495,
    "ounce": 0.0283495,
    "ounces": 0.0283495,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "tonne": 1000.0,
    "tonnes": 1000.0,
    "ton": 907.185,  # US short ton
    "tons": 907.185,
}

# Area → km²
_AREA_TO_KM2 = {
    "sq mi": 2.58999,
    "square mile": 2.58999,
    "square miles": 2.58999,
    "mi2": 2.58999,
    "mi²": 2.58999,
    "sq km": 1.0,
    "square km": 1.0,
    "square kilometer": 1.0,
    "square kilometers": 1.0,
    "square kilometre": 1.0,
    "square kilometres": 1.0,
    "km²": 1.0,
    "km2": 1.0,
    "acre": 4.04686e-3,
    "acres": 4.04686e-3,
    "hectare": 0.01,
    "hectares": 0.01,
    "ha": 0.01,
}


# ------------------------------------------------------------------
# Temperature conversion (non-linear)
# ------------------------------------------------------------------


def _convert_temperature(value: float, unit: str) -> tuple:
    """Convert a temperature to Celsius. Returns (value_celsius, '°C')."""
    unit_lower = unit.lower().strip("° ")
    if unit_lower in ("f", "fahrenheit"):
        return (value - 32) * 5 / 9, "°C"
    elif unit_lower in ("k", "kelvin"):
        return value - 273.15, "°C"
    # Already Celsius
    return value, "°C"


# ------------------------------------------------------------------
# Build regex patterns
# ------------------------------------------------------------------


def _build_unit_pattern(unit_dict: dict) -> str:
    """Build alternation pattern from unit dictionary keys, longest first."""
    units = sorted(unit_dict.keys(), key=len, reverse=True)
    escaped = [re.escape(u) for u in units]
    return "|".join(escaped)


# Temperature units (handled separately)
_TEMP_UNITS = (
    r"°F|°C|°K|degrees?\s+fahrenheit|degrees?\s+celsius|degrees?\s+kelvin"
    r"|fahrenheit|celsius|kelvin|\bK\b|\bF\b"
)

# Pre-built patterns
_DIST_PATTERN = _build_unit_pattern(_DIST_TO_KM)
_SPEED_PATTERN = _build_unit_pattern(_SPEED_TO_KMH)
_MASS_PATTERN = _build_unit_pattern(_MASS_TO_KG)
_AREA_PATTERN = _build_unit_pattern(_AREA_TO_KM2)


def _make_conversion_regex(unit_pattern: str) -> re.Pattern:
    """Build a compiled regex: NUMBER [multiplier] UNIT."""
    return re.compile(
        _NUMBER_RE + _MULTIPLIER_RE + r"\s*(" + unit_pattern + r")\b",
        re.IGNORECASE,
    )


_RE_DIST = _make_conversion_regex(_DIST_PATTERN)
_RE_SPEED = _make_conversion_regex(_SPEED_PATTERN)
_RE_MASS = _make_conversion_regex(_MASS_PATTERN)
_RE_AREA = _make_conversion_regex(_AREA_PATTERN)
_RE_TEMP = re.compile(
    _NUMBER_RE + r"\s*(" + _TEMP_UNITS + r")",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Core conversion functions
# ------------------------------------------------------------------


def _convert_match(match: re.Match, factor_dict: dict, canonical_unit: str) -> str:
    """Convert a regex match (number + optional multiplier + unit) to canonical form."""
    num_str = match.group(1)
    multiplier = match.group(2)  # may be None
    unit = match.group(3)

    value = _parse_number(num_str, multiplier)
    factor = factor_dict.get(unit.lower())
    if factor is None:
        # Already canonical or unknown
        return match.group(0)

    if factor == 1.0:
        # Already in canonical unit — just normalize the number format
        return f"{_format_number(value)} {canonical_unit}"

    converted = value * factor
    return f"{_format_number(converted)} {canonical_unit}"


def _convert_temp_match(match: re.Match) -> str:
    """Convert a temperature regex match to Celsius."""
    num_str = match.group(1)
    unit = match.group(2)

    value = _parse_number(num_str)

    # Detect unit type
    unit_clean = unit.lower().strip("° ")
    if unit_clean in ("c", "celsius") or unit.startswith("°C"):
        # Already Celsius
        return match.group(0)

    converted, canon = _convert_temperature(value, unit)
    return f"{_format_number(converted)} {canon}"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def normalize_number_formats(text: str) -> str:
    """
    Strip commas from numbers in text.

    ``"384,400 km"`` → ``"384400 km"``
    ``"1,234,567"`` → ``"1234567"``
    """

    def _strip_commas(m: re.Match) -> str:
        return m.group(0).replace(",", "")

    # Match numbers with commas (at least one comma group)
    return re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", _strip_commas, text)


def normalize_units(text: str) -> str:
    """
    Normalize measurement units in text to canonical forms.

    Converts distances to km, speeds to km/h, temperatures to °C,
    masses to kg, and areas to km². Also strips number commas.

    Args:
        text: Input text possibly containing measurements.

    Returns:
        Text with measurements converted to canonical units.
        Non-measurement text is unchanged.

    Examples:
        >>> normalize_units("The distance is 93 million miles")
        'The distance is 149.67 million km'
        >>> normalize_units("Speed: 515,000 mph")
        'Speed: 828812.10 km/h'
        >>> normalize_units("Temperature: 5778 K")
        'Temperature: 5505 °C'
    """
    if not text:
        return text

    # Apply conversions in order (areas first — "sq mi" before "mi")
    result = _RE_AREA.sub(lambda m: _convert_match(m, _AREA_TO_KM2, "km²"), text)
    result = _RE_SPEED.sub(lambda m: _convert_match(m, _SPEED_TO_KMH, "km/h"), result)
    result = _RE_DIST.sub(lambda m: _convert_match(m, _DIST_TO_KM, "km"), result)
    result = _RE_MASS.sub(lambda m: _convert_match(m, _MASS_TO_KG, "kg"), result)
    result = _RE_TEMP.sub(_convert_temp_match, result)

    # Normalize number formatting (strip commas)
    result = normalize_number_formats(result)

    return result
